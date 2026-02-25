//=============================================================================
// seq_cuda.cu - CUDA-Accelerated Two-Terminal Network Reliability
//
// Parallelization Strategy:
// 1. CPU maintains priority queue (sequential, small overhead)
// 2. GPU performs batched connectivity checks (parallel, main workload)
// 3. GPU performs batched path finding with branch detection
//
// Key optimizations:
// - Batched processing (BATCH_SIZE masks at once)
// - Pinned memory for fast CPU-GPU transfers
// - Asynchronous operations with streams
// - Warp-level primitives for small graphs
// - Coalesced memory access patterns
//=============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <vector>
#include <sys/resource.h>

#include "CUDA_Reliability.h"
#include "CUDA_Kernels.h"

using namespace std;

//=============================================================================
// Type Definitions (same as CPU version)
//=============================================================================
typedef double Probability;
typedef int NodeId;
typedef int EdgeId;

//=============================================================================
// Graph Reading (simplified from CPU version)
//=============================================================================
struct HostGraph {
    int N;                      // Number of nodes
    int M;                      // Number of edges
    int MaskSize;               // Unique edges
    char GT;                    // Graph type
    
    // CSR format
    vector<int> rowPtr;         // [N+1]
    vector<int> colIdx;         // [M]
    vector<int> edgeSeqNo;      // [M]
    vector<Probability> prob;   // [MaskSize]
    vector<double> logProb;     // [MaskSize]
    vector<double> logProbFail; // [MaskSize]
};

bool readGraph(const string& filename, HostGraph& g) {
    ifstream infile(filename);
    if (!infile) {
        cerr << "Cannot open file: " << filename << endl;
        return false;
    }
    
    infile >> g.N >> g.M >> g.GT;
    cout << "Reading graph: N=" << g.N << ", M=" << g.M << ", Type=" << g.GT << endl;
    
    g.MaskSize = (g.GT == 'U') ? g.M / 2 : g.M;
    
    // Temporary adjacency list
    vector<vector<pair<int, pair<int, double>>>> adj(g.N);  // [dst, (seqNo, prob)]
    
    int seqNo = 0;
    for (int i = 0; i < g.N; i++) {
        int degree;
        infile >> degree;
        
        for (int j = 0; j < degree; j++) {
            int dst;
            double p;
            infile >> dst >> p;
            adj[i].push_back({dst, {seqNo++, p}});
        }
    }
    
    // For undirected graphs, adjust sequence numbers
    if (g.GT == 'U') {
        int offset = adj[0].size();
        for (int i = 1; i < g.N; i++) {
            for (auto& edge : adj[i]) {
                int dst = edge.first;
                if (dst < i) {
                    // Find the reverse edge and use its seqNo
                    for (auto& revEdge : adj[dst]) {
                        if (revEdge.first == i) {
                            edge.second.first = revEdge.second.first;
                            break;
                        }
                    }
                } else {
                    edge.second.first = offset++;
                }
            }
        }
    }
    
    // Convert to CSR
    g.rowPtr.resize(g.N + 1);
    g.rowPtr[0] = 0;
    
    for (int i = 0; i < g.N; i++) {
        g.rowPtr[i + 1] = g.rowPtr[i] + adj[i].size();
        for (auto& edge : adj[i]) {
            g.colIdx.push_back(edge.first);
            g.edgeSeqNo.push_back(edge.second.first);
        }
    }
    
    // Build probability arrays
    g.prob.resize(g.MaskSize, 0.0);
    g.logProb.resize(g.MaskSize);
    g.logProbFail.resize(g.MaskSize);
    
    for (int i = 0; i < g.N; i++) {
        for (auto& edge : adj[i]) {
            int sn = edge.second.first;
            g.prob[sn] = edge.second.second;
        }
    }
    
    const double EPSILON = 1e-15;
    for (int i = 0; i < g.MaskSize; i++) {
        double p = g.prob[i];
        if (p < EPSILON) p = EPSILON;
        if (p > 1.0 - EPSILON) p = 1.0 - EPSILON;
        g.logProb[i] = log(p);
        g.logProbFail[i] = log(1.0 - p);
    }
    
    return true;
}

//=============================================================================
// Priority Queue Entry
//=============================================================================
struct MaskEntry {
    vector<unsigned char> mask;
    double logProb;
    
    bool operator<(const MaskEntry& other) const {
        return logProb < other.logProb;  // Max-heap
    }
};

//=============================================================================
// CUDA Reliability Calculator
//=============================================================================
class CUDAReliabilityCalculator {
private:
    HostGraph& graph;
    DeviceGraph deviceGraph;
    BatchData batch;
    cudaStream_t stream;
    
    // Priority queue (on CPU)
    priority_queue<MaskEntry> pq;
    
    // Results
    double reliability;
    double truncationError;
    int pathsProcessed;
    int iterations;
    
    // Working storage for batch processing
    vector<MaskEntry> currentBatch;
    vector<MaskEntry> newMasks;
    
public:
    CUDAReliabilityCalculator(HostGraph& g, int src, int t) 
        : graph(g), reliability(0.0), truncationError(0.0), 
          pathsProcessed(0), iterations(0) 
    {
        // Initialize CUDA
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        // Initialize device graph
        initDeviceGraph(&deviceGraph, g.N, g.M, g.MaskSize,
                        g.rowPtr.data(), g.colIdx.data(), g.edgeSeqNo.data(),
                        src, t);
        
        // Initialize batch data
        initBatchData(&batch, g.MaskSize);
    }
    
    ~CUDAReliabilityCalculator() {
        freeBatchData(&batch);
        freeDeviceGraph(&deviceGraph);
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    
    //=========================================================================
    // CPU-side path finding (for processing results)
    //=========================================================================
    bool findPathCPU(const unsigned char* mask, vector<int>& unmarkedEdges) {
        int N = graph.N;
        int src = deviceGraph.src;
        int t = deviceGraph.t;
        
        vector<bool> visited(N, false);
        vector<int> parent(N, -1);
        vector<int> parentEdge(N, -1);
        queue<int> q;
        
        q.push(src);
        visited[src] = true;
        
        while (!q.empty()) {
            int node = q.front();
            q.pop();
            
            int start = graph.rowPtr[node];
            int end = graph.rowPtr[node + 1];
            
            for (int e = start; e < end; e++) {
                int neighbor = graph.colIdx[e];
                int sn = graph.edgeSeqNo[e];
                
                if (!visited[neighbor] && 
                    (mask[sn] == MASK_UP || mask[sn] == MASK_NOMARK)) {
                    visited[neighbor] = true;
                    parent[neighbor] = node;
                    parentEdge[neighbor] = sn;
                    q.push(neighbor);
                    
                    if (neighbor == t) {
                        // Trace back and collect unmarked edges
                        unmarkedEdges.clear();
                        int n = t;
                        while (n != src) {
                            int sn = parentEdge[n];
                            if (mask[sn] == MASK_NOMARK) {
                                unmarkedEdges.push_back(sn);
                            }
                            n = parent[n];
                        }
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
    //=========================================================================
    // Check connectivity using GPU (batched)
    //=========================================================================
    void checkConnectivityBatch(vector<MaskEntry>& masks, vector<bool>& results) {
        results.resize(masks.size());
        
        // Process in batches
        for (size_t batchStart = 0; batchStart < masks.size(); batchStart += BATCH_SIZE) {
            size_t batchEnd = min(batchStart + BATCH_SIZE, masks.size());
            batch.count = batchEnd - batchStart;
            
            // Copy masks to pinned memory
            for (size_t i = 0; i < batch.count; i++) {
                memcpy(batch.h_masks + i * batch.maskSize,
                       masks[batchStart + i].mask.data(),
                       batch.maskSize);
            }
            
            // Launch GPU kernel
            launchConnectivityCheck(&deviceGraph, &batch, stream);
            
            // Wait for completion
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            // Copy results
            for (size_t i = 0; i < batch.count; i++) {
                results[batchStart + i] = (batch.h_results[i] == 1);
            }
        }
    }
    
    //=========================================================================
    // Main computation loop
    //=========================================================================
    void compute() {
        cout << "\n=== Starting CUDA Reliability Computation ===" << endl;
        cout << "Nodes: " << graph.N << ", Edges: " << graph.M 
             << ", MaskSize: " << graph.MaskSize << endl;
        cout << "Batch size: " << BATCH_SIZE << endl;
        
        clock_t startTime = clock();
        
        // Initialize with starting mask
        MaskEntry initial;
        initial.mask.resize(graph.MaskSize, MASK_NOMARK);
        initial.logProb = 0.0;
        pq.push(initial);
        
        // Main loop
        while (!pq.empty()) {
            // Extract batch from priority queue
            currentBatch.clear();
            while (!pq.empty() && currentBatch.size() < BATCH_SIZE) {
                currentBatch.push_back(pq.top());
                pq.pop();
            }
            
            iterations += currentBatch.size();
            
            // Process each mask in batch
            newMasks.clear();
            
            for (auto& entry : currentBatch) {
                vector<int> unmarkedEdges;
                
                // Find path (CPU - could also batch this on GPU)
                if (!findPathCPU(entry.mask.data(), unmarkedEdges)) {
                    // No path - contributes 0
                    continue;
                }
                
                if (unmarkedEdges.empty()) {
                    // All edges on path are marked UP
                    pathsProcessed++;
                    double prob = exp(entry.logProb);
                    if (isfinite(prob)) {
                        reliability += prob;
                    }
                    continue;
                }
                
                // Factor on unmarked edges
                double logMult = entry.logProb;
                vector<unsigned char> mask = entry.mask;
                
                for (size_t i = 0; i < unmarkedEdges.size(); i++) {
                    int sn = unmarkedEdges[i];
                    
                    // Branch: edge fails
                    MaskEntry failEntry;
                    failEntry.mask = mask;
                    failEntry.mask[sn] = MASK_DOWN;
                    failEntry.logProb = logMult + graph.logProbFail[sn];
                    newMasks.push_back(failEntry);
                    
                    // Continue: edge works
                    mask[sn] = MASK_UP;
                    logMult += graph.logProb[sn];
                }
                
                // All path edges now UP
                pathsProcessed++;
                double prob = exp(logMult);
                if (isfinite(prob)) {
                    reliability += prob;
                }
            }
            
            // Batch connectivity check for new masks
            if (!newMasks.empty()) {
                vector<bool> connected;
                checkConnectivityBatch(newMasks, connected);
                
                // Add connected masks to queue
                for (size_t i = 0; i < newMasks.size(); i++) {
                    if (connected[i]) {
                        if (pq.size() < MAX_Q_SIZE  ) {
                            pq.push(newMasks[i]);
                        } else {
                            double prob = exp(newMasks[i].logProb);
                            if (isfinite(prob)) {
                                truncationError += prob;
                            }
                        }
                    }
                }
            }
            
            // Progress report
            if (iterations % 10000 == 0) {
                clock_t now = clock();
                double elapsed = (double)(now - startTime) / CLOCKS_PER_SEC;
                printf("Iter %d | Q=%zu | rel=%.10e | Eps=%.10e | %.0f iter/s\n",
                       iterations, pq.size(), reliability, truncationError,
                       iterations / elapsed);
            }
        }
        
        clock_t endTime = clock();
        double totalTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;
        
        cout << "\n=== Computation Complete ===" << endl;
        cout << "Total iterations: " << iterations << endl;
        cout << "Paths processed:  " << pathsProcessed << endl;
        cout << "Time:             " << totalTime << " seconds" << endl;
        if (totalTime > 0) {
            cout << "Rate:             " << (iterations / totalTime) << " iter/sec" << endl;
        }
    }
    
    double getReliability() const { return reliability; }
    double getTruncationError() const { return truncationError; }
    int getPathsProcessed() const { return pathsProcessed; }
};

//=============================================================================
// Main Function
//=============================================================================
int main(int argc, char* argv[]) {
    string filename;
    int src, t;
    
    if (argc >= 4) {
        filename = argv[1];
        src = atoi(argv[2]);
        t = atoi(argv[3]);
    } else {
        cout << "Usage: " << argv[0] << " <input_file> <source> <terminal>" << endl;
        cout << "Enter Input File Name: ";
        cin >> filename;
        cout << "Enter source and terminal nodes: ";
        cin >> src >> t;
    }
    
    cout << "\n=== CUDA Network Reliability Calculator ===" << endl;
    cout << "Input file: " << filename << endl;
    cout << "Source: " << src << ", Terminal: " << t << endl;
    
    // Print CUDA device info
    int deviceId;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));
    cout << "GPU: " << props.name << " (SM " << props.major << "." << props.minor << ")" << endl;
    cout << "Memory: " << (props.totalGlobalMem / 1024 / 1024) << " MB" << endl;
    
    // Read graph
    HostGraph graph;
    if (!readGraph(filename, graph)) {
        return 1;
    }
    
    // Validate inputs
    if (src < 0 || src >= graph.N || t < 0 || t >= graph.N) {
        cerr << "Error: Invalid source or terminal node" << endl;
        return 1;
    }
    
    if (src == t) {
        cout << "Source equals terminal - reliability = 1.0" << endl;
        return 0;
    }
    
    // Run computation
    clock_t totalStart = clock();
    
    CUDAReliabilityCalculator calculator(graph, src, t);
    calculator.compute();
    
    clock_t totalEnd = clock();
    
    // Results
    cout << "\n=== RESULTS ===" << endl;
    printf("Reliability:       %.20lf\n", calculator.getReliability());
    printf("Truncation (Eps):  %.20lf\n", calculator.getTruncationError());
    printf("Total:             %.20lf\n", 
           calculator.getReliability() + calculator.getTruncationError());
    cout << "Paths enumerated:  " << calculator.getPathsProcessed() << endl;
    
    double totalTime = (double)(totalEnd - totalStart) * 1000 / CLOCKS_PER_SEC;
    cout << "Total time:        " << totalTime << " ms" << endl;
    
    // Memory usage
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    cout << "Max RSS:           " << usage.ru_maxrss << " KB" << endl;
    
    return 0;
}
