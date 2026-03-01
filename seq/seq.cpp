//=============================================================================
// seq_improved.cpp - Two-Terminal Network Reliability using Factoring Method
//
// Improvements over original:
// 1.  Memory pool for masks (single contiguous allocation)
// 2.  memcpy/memset for fast copying
// 3.  Logarithmic probability arithmetic (prevents underflow)
// 4.  Consistent data types (double for probabilities, int32 for indices)
// 5.  std::swap for cleaner code
// 6.  Input validation
// 7.  Optimized BFS with early termination
// 8.  Progress reporting
// 9.  Proper memory cleanup
// 10. Better code organization and documentation
//=============================================================================

#include "TYPES.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <sys/resource.h>

using namespace std;

//=============================================================================
// Global File Handle
//=============================================================================
ifstream infile;

#include "GraphRP.h"

//=============================================================================
// Constants
//=============================================================================
#define MAX_Q_SIZE 13000000    // Maximum queue size
#define PROB_EPSILON 1e-15     // Minimum probability to avoid log(0)
#define REPORT_INTERVAL 100000 // Progress report every N iterations

//=============================================================================
// Priority Queue Structure (Max-Heap by Log-Probability)
//=============================================================================
struct MaskPriorityQueue {
    unsigned char** masks;     // Array of pointers to masks
    unsigned char* pool;       // Contiguous memory pool for all masks
    double* logKeys;           // Log-probability keys
    int size;                  // Current number of elements
    int capacity;              // Maximum capacity
    int maskSize;              // Size of each mask in bytes
    
    //-------------------------------------------------------------------------
    // Initialize the priority queue
    //-------------------------------------------------------------------------
    bool init(int cap, int mSize) {
        capacity = cap; // = MAX_Q_SIZE constant
        maskSize = mSize; // Number of Edges = mask size
        size = 0;
        
        // Allocate log-keys array
        logKeys = new (nothrow) double[cap];
        if (!logKeys) {
            cerr << "Failed to allocate logKeys array" << endl;
            return false;
        }
        
        // Initialize all keys to 0 (log(1) = 0)
        for (int i = 0; i < cap; i++) {
            logKeys[i] = 0.0;
        }
        
        // Allocate pointer array
        masks = new (nothrow) unsigned char*[cap];
        if (!masks) {
            cerr << "Failed to allocate masks array" << endl;
            delete[] logKeys;
            return false;
        }
        
        // Single contiguous allocation for all mask data
        pool = new (nothrow) unsigned char[(size_t)cap * mSize];
        if (!pool) {
            cerr << "Failed to allocate mask pool (" 
                 << ((size_t)cap * mSize / 1024 / 1024) << " MB)" << endl;
            delete[] logKeys;
            delete[] masks;
            return false;
        }
        
        // Set up pointers into the pool
        for (int i = 0; i < cap; i++) {
            masks[i] = pool + (size_t)i * mSize;
        }
        
        cout << "Allocated mask pool: " << ((size_t)cap * mSize / 1024 / 1024) 
             << " MB for " << cap << " masks of " << mSize << " bytes each" << endl;
        
        return true;
    }
    
    //-------------------------------------------------------------------------
    // Cleanup
    //-------------------------------------------------------------------------
    void destroy() {
        delete[] pool;
        delete[] masks;
        delete[] logKeys;
        pool = nullptr;
        masks = nullptr;
        logKeys = nullptr;
    }
    
    //-------------------------------------------------------------------------
    // Sift up (for insertion)
    //-------------------------------------------------------------------------
    void siftUp(int i) {
        while (i > 0) {
            int parent = (i - 1) / 2;
            if (logKeys[parent] >= logKeys[i]) break;
            
            swap(masks[parent], masks[i]);
            swap(logKeys[parent], logKeys[i]);
            i = parent;
        }
    }
    
    //-------------------------------------------------------------------------
    // Sift down (for extraction)
    //-------------------------------------------------------------------------
    void siftDown(int i) {
        while (2 * i + 1 < size) {
            int left = 2 * i + 1;
            int right = 2 * i + 2;
            int largest = i;
            
            if (logKeys[left] > logKeys[largest]) largest = left;
            if (right < size && logKeys[right] > logKeys[largest]) largest = right;
            
            if (largest == i) break;
            
            swap(masks[i], masks[largest]);
            swap(logKeys[i], logKeys[largest]);
            i = largest;
        }
    }
    
    //-------------------------------------------------------------------------
    // Push a new mask onto the queue
    //-------------------------------------------------------------------------
    bool push(const unsigned char* mask, double logKey) {
        if (size >= capacity) return false;
        
        memcpy(masks[size], mask, maskSize);
        logKeys[size] = logKey;
        size++;
        siftUp(size - 1);
        return true;
    }
    
    //-------------------------------------------------------------------------
    // Pop the highest-priority mask
    //-------------------------------------------------------------------------
    void pop(unsigned char* outMask, double& outLogKey) {
        memcpy(outMask, masks[0], maskSize);
        outLogKey = logKeys[0];
        
        size--;
        if (size > 0) {
            swap(masks[0], masks[size]);
            swap(logKeys[0], logKeys[size]);
            siftDown(0);
        }
    }
    
    //-------------------------------------------------------------------------
    // Check if empty/full
    //-------------------------------------------------------------------------
    bool empty() const { return size == 0; }
    bool full() const { return size >= capacity; }
    
    //-------------------------------------------------------------------------
    // Sum remaining probability mass (for truncation reporting)
    //-------------------------------------------------------------------------
    double sumRemainingProbability() const {
        double sum = 0.0;
        for (int i = 0; i < size; i++) {
            double p = exp(logKeys[i]);
            if (isfinite(p)) sum += p;
        }
        return sum;
    }
};

//=============================================================================
// Algorithm Context (encapsulates all state)
//=============================================================================
struct ReliabilityContext {
    // Graph data (compact representation)
    N_Type* Nodes;
    E_Type* Edges;
    int N;                      // Number of nodes
    int M;                      // Number of edges
    int MaskSize;               // Number of unique edges (M/2 for undirected)
    unsigned char GT;           // Graph type
    
    // Probability data
    Probability* Prob;          // Edge probabilities
    double* LogProb;            // Log of edge probabilities
    double* LogProbFail;        // Log of (1 - edge probability)
    
    // Source and terminal
    int src;
    int t;
    
    // Priority queue
    MaskPriorityQueue pq;
    
    // Working arrays
    unsigned char* Mask;        // Current working mask
    unsigned char* NewMask;     // Temporary mask for branching
    short* PathStack;           // Stack of unmarked edges on path
    int PathStackTop;
    
    // BFS arrays
    bool* Visited;
    short* Parent;
    short* Queue;
    
    // Results
    double rel;                 // Accumulated reliability
    double Eps;                 // Truncated probability mass
    int NP;                     // Number of paths processed
    int iterations;             // Total iterations
    
    // Timing
    clock_t startTime;
};

//=============================================================================
// Find SeqNo of edge (i, j)
//=============================================================================
inline EdgeId Seq_No(int i, int j, N_Type Nodes[], E_Type Edges[]) {
    int offset = Nodes[i].Offset;
    int degree = Nodes[i].degree;
    
    for (int d = 0; d < degree; d++) {
        if (Edges[offset + d].dst == j) {
            return Edges[offset + d].SeqNo;
        }
    }
    return -1;  // Should never reach here
}

//=============================================================================
// BFS: Check if s-t path exists under given mask
// Returns true if path exists
// In this function, we don't care about the path, we only test if path exists. 
//=============================================================================
inline bool checkConnectivity(ReliabilityContext* ctx, const unsigned char* mask) {
    int N = ctx->N;
    int src = ctx->src;
    int t = ctx->t;
    N_Type* Nodes = ctx->Nodes;
    E_Type* Edges = ctx->Edges;
    bool* Visited = ctx->Visited;
    short* Queue = ctx->Queue;
    
    memset(Visited, 0, N * sizeof(bool));
    
    int front = 0, rear = 0;
    Queue[rear++] = (short)src;
    Visited[src] = true;
    
    while (front < rear) {
        int V = Queue[front++];
        
        // Early exit if target reached
        if (V == t) return true;
        
        int d = Nodes[V].degree;
        int offset = Nodes[V].Offset;
        
        for (int dx = 0; dx < d; dx++) {
            int j = Edges[offset + dx].dst;
            int sn = Edges[offset + dx].SeqNo;
            
            // Edge is traversable if UP (1) or unmarked (0xFF)
            if (!Visited[j] && (mask[sn] == MASK_UP || mask[sn] == MASK_NOMARK)) {
                Visited[j] = true;
                Queue[rear++] = j;
                
                // Early exit
                if (j == t) return true;
            }
        }
    }
    
    return false;
}

//===================================================================================
// BFS: Find s-t path and populate Parent array
// Returns true if path exists
// --> Same as checkConnectivity, but finds and return the actual path (parent array)
//===================================================================================
inline bool findPath(ReliabilityContext* ctx) {
    int N = ctx->N;
    int src = ctx->src;
    int t = ctx->t;
    N_Type* Nodes = ctx->Nodes;
    E_Type* Edges = ctx->Edges;
    unsigned char* Mask = ctx->Mask;
    bool* Visited = ctx->Visited;
    short* Parent = ctx->Parent;
    short* Queue = ctx->Queue;
    
    memset(Visited, 0, N * sizeof(bool));
    memset(Parent, -1, N * sizeof(short));
    
    int front = 0, rear = 0;
    Queue[rear++] = (short)src;
    Visited[src] = true;
    
    while (front < rear) {
        int V = Queue[front++];
        
        int d = Nodes[V].degree;
        int offset = Nodes[V].Offset;
        
        for (int dx = 0; dx < d; dx++) {
            int j = Edges[offset + dx].dst;
            int sn = Edges[offset + dx].SeqNo;
            
            // Edge is traversable if UP or unmarked
            if (!Visited[j] && (Mask[sn] == MASK_UP || Mask[sn] == MASK_NOMARK)) {
                Visited[j] = true;
                Parent[j] = V;
                Queue[rear++] = j;
                
                // Early exit if target found
                if (j == t) return true;
            }
        }
    }
    
    return false;
}

//=============================================================================
// Trace path from t back to src, collect unmarked edges
//=============================================================================
inline void tracePathAndCollectUnmarkedEdges(ReliabilityContext* ctx) {
    int src = ctx->src;
    int t = ctx->t;
    short* Parent = ctx->Parent;
    unsigned char* Mask = ctx->Mask;
    short* PathStack = ctx->PathStack;
    N_Type* Nodes = ctx->Nodes;
    E_Type* Edges = ctx->Edges;
    
    ctx->PathStackTop = -1;
    
    int j = t;
    int i = Parent[t];
    
    while (j != src) {
        EdgeId sn = Seq_No(i, j, Nodes, Edges);
        if (Mask[sn] == MASK_NOMARK) {
            ctx->PathStackTop++;
            PathStack[ctx->PathStackTop] = sn;
        }
        j = i;
        i = Parent[j];
    }
}

//=============================================================================
// Initialize context
//=============================================================================
bool initContext(ReliabilityContext* ctx, Graph_Type* G, 
                 N_Type* Nodes, E_Type* Edges,
                 int src, int t, int maxQSize) {
    
    ctx->Nodes = Nodes;
    ctx->Edges = Edges;
    ctx->N = G->N;
    ctx->M = G->M;
    ctx->GT = G->GT;
    ctx->src = src;
    ctx->t = t;
    
    // Determine mask size
    ctx->MaskSize = (G->GT == 'U') ? G->M / 2 : G->M;
    
    // Initialize results
    ctx->rel = 0.0;
    ctx->Eps = 0.0;
    ctx->NP = 0;
    ctx->iterations = 0;
    
    // Allocate probability arrays
    ctx->Prob = new (nothrow) Probability[ctx->MaskSize];
    ctx->LogProb = new (nothrow) double[ctx->MaskSize];
    ctx->LogProbFail = new (nothrow) double[ctx->MaskSize];
    
    if (!ctx->Prob || !ctx->LogProb || !ctx->LogProbFail) {
        cerr << "Failed to allocate probability arrays" << endl;
        return false;
    }
    
    // Fill probability arrays from edges
    for (int i = 0; i < G->M; i++) {
        int sn = Edges[i].SeqNo;
        ctx->Prob[sn] = Edges[i].prob;
    }
    
    // Compute log probabilities with clamping
    for (int i = 0; i < ctx->MaskSize; i++) {
        double p = ctx->Prob[i];
        
        // Validate probability
        if (p <= 0.0 || p >= 1.0) {
            cerr << "Warning: Edge " << i << " has invalid probability " << p 
                 << ", clamping to valid range" << endl;
        }
        
        // Clamp to avoid log(0)
        if (p < PROB_EPSILON) p = PROB_EPSILON;
        if (p > 1.0 - PROB_EPSILON) p = 1.0 - PROB_EPSILON;
        
        ctx->LogProb[i] = log(p);
        ctx->LogProbFail[i] = log(1.0 - p);
    }
    
    // Initialize priority queue
    if (!ctx->pq.init(maxQSize, ctx->MaskSize)) {
        return false;
    }
    
    // Allocate working arrays
    ctx->Mask = new (nothrow) unsigned char[ctx->MaskSize];
    ctx->NewMask = new (nothrow) unsigned char[ctx->MaskSize];
    ctx->PathStack = new (nothrow) short[ctx->MaskSize];
    ctx->Visited = new (nothrow) bool[ctx->N];
    ctx->Parent = new (nothrow) short[ctx->N];
    ctx->Queue = new (nothrow) short[ctx->N];
    
    if (!ctx->Mask || !ctx->NewMask || !ctx->PathStack ||
        !ctx->Visited || !ctx->Parent || !ctx->Queue) {
        cerr << "Failed to allocate working arrays" << endl;
        return false;
    }
    
    return true;
}

//=============================================================================
// Cleanup context
//=============================================================================
void destroyContext(ReliabilityContext* ctx) {
    delete[] ctx->Prob;
    delete[] ctx->LogProb;
    delete[] ctx->LogProbFail;
    delete[] ctx->Mask;
    delete[] ctx->NewMask;
    delete[] ctx->PathStack;
    delete[] ctx->Visited;
    delete[] ctx->Parent;
    delete[] ctx->Queue;
    ctx->pq.destroy();
}

//=============================================================================
// Main Reliability Algorithm
//=============================================================================
void computeReliability(ReliabilityContext* ctx) {
    ctx->startTime = clock();
    
    cout << "\n=== Starting Reliability Computation ===" << endl;
    cout << "Nodes: " << ctx->N << ", Edges: " << ctx->M 
         << ", MaskSize: " << ctx->MaskSize << endl;
    cout << "Source: " << ctx->src << ", Terminal: " << ctx->t << endl;
    
    //-------------------------------------------------------------------------
    // Initialize with starting mask (all edges unmarked)
    //-------------------------------------------------------------------------
    memset(ctx->Mask, MASK_NOMARK, ctx->MaskSize);
    
    // Push initial mask with log(1) = 0
    ctx->pq.push(ctx->Mask, 0.0);
    
    // clock_t lastReport = ctx->startTime;  // For future use with timed reports
    
    //-------------------------------------------------------------------------
    // Main processing loop
    //-------------------------------------------------------------------------
    while (!ctx->pq.empty()) {
        // Extract highest-priority mask
        double logMult;
        ctx->pq.pop(ctx->Mask, logMult);
        
        ctx->iterations++;
        
        //---------------------------------------------------------------------
        // Progress reporting
        //---------------------------------------------------------------------
        if (ctx->iterations % REPORT_INTERVAL == 0) {
            clock_t now = clock();
            double elapsed = (double)(now - ctx->startTime) / CLOCKS_PER_SEC;
            double rate = ctx->iterations / elapsed;
            
            printf("Iter %d | Q=%d | rel=%.10e | Eps=%.10e | %.0f iter/s\n",
                   ctx->iterations, ctx->pq.size, ctx->rel, ctx->Eps, rate);
        }
        
        //---------------------------------------------------------------------
        // Find s-t path using BFS
        //---------------------------------------------------------------------
        if (!findPath(ctx)) {
            // No path exists - this mask contributes 0 to reliability
            continue;
        }
        
        //---------------------------------------------------------------------
        // Path exists - trace it and collect unmarked edges
        //---------------------------------------------------------------------
        
        
        tracePathAndCollectUnmarkedEdges(ctx);
    
        //---------------------------------------------------------------------
        // Factor on each unmarked edge
        //---------------------------------------------------------------------
        while (ctx->PathStackTop >= 0) {
            int sn = ctx->PathStack[ctx->PathStackTop];
            ctx->PathStackTop--;
            
            //-----------------------------------------------------------------
            // Branch 1: Edge FAILS
            //-----------------------------------------------------------------
            memcpy(ctx->NewMask, ctx->Mask, ctx->MaskSize);
            ctx->NewMask[sn] = MASK_DOWN;
            double logNewMult = logMult + ctx->LogProbFail[sn];
            
            // Only add to queue if graph still operational
            if(checkConnectivity(ctx, ctx->NewMask)) {
                if (!ctx->pq.full()) {
                    ctx->pq.push(ctx->NewMask, logNewMult);
                } else {
                    // Queue full - add to truncation error
                    ctx->NP++;
                    double prob = exp(logNewMult);
                    if (isfinite(prob)) {
                        ctx->Eps += prob;
                    }
                }
            }
            
            //-----------------------------------------------------------------
            // Branch 2: Edge WORKS (continue with current mask)
            //-----------------------------------------------------------------
            ctx->Mask[sn] = MASK_UP;
            logMult += ctx->LogProb[sn];
        }
        
        //---------------------------------------------------------------------
        // All path edges now marked UP - accumulate probability
        //---------------------------------------------------------------------
        ctx->NP++;
        double prob = exp(logMult);
        if (isfinite(prob)) {
            ctx->rel += prob;
        }
    }
    
    //-------------------------------------------------------------------------
    // Final summary
    //-------------------------------------------------------------------------
    clock_t endTime = clock();
    double totalTime = (double)(endTime - ctx->startTime) / CLOCKS_PER_SEC;
    
    cout << "\n=== Computation Complete ===" << endl;
    cout << "Total iterations: " << ctx->iterations << endl;
    cout << "Paths processed:  " << ctx->NP << endl;
    cout << "Time:             " << totalTime << " seconds" << endl;
    cout << "Rate:             " << (ctx->iterations / totalTime) << " iter/sec" << endl;
}

//=============================================================================
// MAIN FUNCTION
//=============================================================================
int main(int argc, char* argv[]) {
    Graph_Type* G = new Graph_Type;
    string fname;
    int src, t;
    
    //-------------------------------------------------------------------------
    // Parse arguments or use defaults
    //-------------------------------------------------------------------------
    if (argc >= 4) {
        fname = argv[1];
        src = atoi(argv[2]);
        t = atoi(argv[3]);
    } else {
        cout << "Enter Input File Name: ";
        cin >> fname;
        cout << "Enter source and terminal nodes: ";
        cin >> src >> t;
        
        // Defaults for testing
        // fname = "temp4x4.txt";
        // src = 0;
        // t = 1;
    }
    
    cout << "\n=== Network Reliability Calculator ===" << endl;
    cout << "Input file: " << fname << endl;
    cout << "Source: " << src << ", Terminal: " << t << endl;
    
    //-------------------------------------------------------------------------
    // Open and read graph
    //-------------------------------------------------------------------------
    infile.open(fname.c_str());
    if (!infile) {
        cerr << "Error: Cannot open input file: " << fname << endl;
        return 1;
    }
    
    clock_t totalStart = clock();
    
    if (!ReadGraph(G)) {
        cerr << "Error: Failed to read graph" << endl;
        return 1;
    }
    infile.close();
    
    cout << "Graph loaded successfully" << endl;
    
    //-------------------------------------------------------------------------
    // Validate inputs
    //-------------------------------------------------------------------------
    if (src < 0 || src >= G->N) {
        cerr << "Error: Source node " << src << " out of range [0, " << G->N - 1 << "]" << endl;
        return 1;
    }
    
    if (t < 0 || t >= G->N) {
        cerr << "Error: Terminal node " << t << " out of range [0, " << G->N - 1 << "]" << endl;
        return 1;
    }
    
    if (src == t) {
        cout << "Source equals terminal - reliability = 1.0" << endl;
        return 0;
    }
    
    //-------------------------------------------------------------------------
    // Build compact representation (CSR-like)
    //-------------------------------------------------------------------------
    N_Type* Nodes = new N_Type[G->N];
    E_Type* Edges = new E_Type[G->M];
    
    Nodes[0].Offset = 0;
    int edgeIdx = 0;
    
    for (int n = 0; n < G->N; n++) {
        int d = G->Nodes[n].degree;
        Nodes[n].degree = d;
        
        if (n > 0) {
            Nodes[n].Offset = Nodes[n - 1].Offset + Nodes[n - 1].degree;
        }
        
        Edge_Type* adjList = G->Nodes[n].Adj_List;
        for (int dx = 0; dx < d; dx++) {
            Edges[edgeIdx].SeqNo = adjList[dx].SeqNo;
            Edges[edgeIdx].prob = adjList[dx].prob;
            Edges[edgeIdx].dst = adjList[dx].dst;
            edgeIdx++;
        }
    }
    
    //-------------------------------------------------------------------------
    // Initialize context and run algorithm
    //-------------------------------------------------------------------------
    ReliabilityContext ctx;
    
    if (!initContext(&ctx, G, Nodes, Edges, src, t, MAX_Q_SIZE)) {
        cerr << "Error: Failed to initialize context" << endl;
        return 1;
    }
    
    computeReliability(&ctx);
    
    //-------------------------------------------------------------------------
    // Output results
    //-------------------------------------------------------------------------
    cout << "\n=== RESULTS ===" << endl;
    printf("Reliability:       %.20lf\n", ctx.rel);
    printf("Truncation (Eps):  %.20lf\n", ctx.Eps);
    printf("Total (rel + Eps): %.20lf\n", ctx.rel + ctx.Eps);
    cout << "Paths enumerated:  " << ctx.NP << endl;
    
    //-------------------------------------------------------------------------
    // Memory usage
    //-------------------------------------------------------------------------
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    cout << "Max RSS:           " << usage.ru_maxrss << " KB" << endl;
    
    //-------------------------------------------------------------------------
    // Total time
    //-------------------------------------------------------------------------
    clock_t totalEnd = clock();
    double totalTime = (double)(totalEnd - totalStart) * 1000 / CLOCKS_PER_SEC;
    cout << "Total time:        " << totalTime << " ms" << endl;
    
    //-------------------------------------------------------------------------
    // Cleanup
    //-------------------------------------------------------------------------
    destroyContext(&ctx);
    delete[] Nodes;
    delete[] Edges;
    FreeGraph(G);
    delete G;
    
    cout << "\n=== Done ===" << endl;
    
    return 0;
}

/*

Reliability:       0.99596245255088433623
Truncation (Eps):  0.00000000000000000000
Total (rel + Eps): 0.99596245255088433623
Paths enumerated:  7433032
Max RSS:           256409600 KB
Total time:        9679.99 ms



Reliability:       0.99596245255088433623
Truncation (Eps):  0.00000000000000000000
Total (rel + Eps): 0.99596245255088433623
Paths enumerated:  7433032
Max RSS:           256294912 KB
Total time:        8849.21 ms

=== Done ===

*/