//=============================================================================
// CUDA_Kernels.cu - CUDA Kernels for Network Reliability
//=============================================================================

#include "CUDA_Reliability.h"
#include <cstdio>

//=============================================================================
// Kernel 1: Parallel Connectivity Check
// 
// Each thread block handles one mask from the batch.
// Threads within a block cooperate to perform BFS.
//=============================================================================
__global__ void connectivityCheckKernel(
    // Graph structure (CSR)
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const int* __restrict__ edgeSeqNo,
    int N,
    int src,
    int t,
    // Batch data
    const unsigned char* __restrict__ masks,
    int maskSize,
    int batchCount,
    // Output
    int* __restrict__ results
) {
    // Each block handles one mask
    int maskIdx = blockIdx.x;
    if (maskIdx >= batchCount) return;
    
    // Pointer to this mask
    const unsigned char* myMask = masks + maskIdx * maskSize;
    
    // Shared memory for BFS
    __shared__ int frontier[MAX_NODES_SHARED];
    __shared__ int nextFrontier[MAX_NODES_SHARED];
    __shared__ int frontierSize;
    __shared__ int nextFrontierSize;
    __shared__ int visited[MAX_NODES_SHARED];
    __shared__ bool found;
    
    int tid = threadIdx.x;
    int numThreads = blockDim.x;
    
    // Initialize
    if (tid == 0) {
        frontierSize = 1;
        nextFrontierSize = 0;
        frontier[0] = src;
        found = false;
    }
    
    // Initialize visited array (parallel)
    for (int i = tid; i < N; i += numThreads) {
        visited[i] = (i == src);
    }
    __syncthreads();
    
    // BFS iterations
    while (frontierSize > 0 && !found) {
        // Each thread processes some frontier nodes
        for (int f = tid; f < frontierSize; f += numThreads) {
            if (found) break;
            
            int node = frontier[f];
            int start = rowPtr[node];
            int end = rowPtr[node + 1];
            
            // Explore neighbors
            for (int e = start; e < end; e++) {
                int neighbor = colIdx[e];
                int sn = edgeSeqNo[e];
                
                // Check if edge is traversable
                unsigned char edgeState = myMask[sn];
                if (edgeState == MASK_UP || edgeState == MASK_NOMARK) {
                    // Try to visit neighbor (atomic to avoid races)
                    //bool wasVisited = atomicExch((int*)&visited[neighbor], 1);
                    //if (!wasVisited) {
                    if (atomicCAS(&visited[neighbor], 0, 1) == 0) {
                        // Check if we found target
                        if (neighbor == t) {
                            found = true;
                        }
                        
                        // Add to next frontier
                        int pos = atomicAdd(&nextFrontierSize, 1);
                        if (pos < MAX_NODES_SHARED) {
                            nextFrontier[pos] = neighbor;
                        }
                    }
                }
            }
        }
        __syncthreads();
        
        // Swap frontiers
        if (tid == 0) {
            frontierSize = min(nextFrontierSize, MAX_NODES_SHARED);
            nextFrontierSize = 0;
        }
        __syncthreads();
        
        // Copy next frontier to current frontier
        for (int i = tid; i < frontierSize; i += numThreads) {
            frontier[i] = nextFrontier[i];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        results[maskIdx] = found ? 1 : 0;
    }
}

//=============================================================================
// Kernel 2: Parallel Path Finding with Unmarked Edge Detection
//
// Similar to connectivity check, but also finds the path and 
// identifies the first unmarked edge on the path.
//=============================================================================
__global__ void pathFindingKernel(
    // Graph structure (CSR)
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const int* __restrict__ edgeSeqNo,
    int N,
    int src,
    int t,
    // Batch data
    const unsigned char* __restrict__ masks,
    int maskSize,
    int batchCount,
    // Output
    int* __restrict__ results,      // 1 if path found
    int* __restrict__ branchEdge    // First unmarked edge on path, -1 if none
) {
    int maskIdx = blockIdx.x;
    if (maskIdx >= batchCount) return;
    
    const unsigned char* myMask = masks + maskIdx * maskSize;
    
    // Shared memory
    __shared__ int frontier[MAX_NODES_SHARED];
    __shared__ int nextFrontier[MAX_NODES_SHARED];
    __shared__ int parent[MAX_NODES_SHARED];
    __shared__ int parentEdge[MAX_NODES_SHARED];  // SeqNo of edge from parent
    __shared__ int frontierSize;
    __shared__ int nextFrontierSize;
    __shared__ int visited[MAX_NODES_SHARED];
    __shared__ bool found;
    
    int tid = threadIdx.x;
    int numThreads = blockDim.x;
    
    // Initialize
    if (tid == 0) {
        frontierSize = 1;
        nextFrontierSize = 0;
        frontier[0] = src;
        found = false;
    }
    
    for (int i = tid; i < N; i += numThreads) {
        visited[i] = (i == src);
        parent[i] = -1;
        parentEdge[i] = -1;
    }
    __syncthreads();
    
    // BFS
    while (frontierSize > 0 && !found) {
        for (int f = tid; f < frontierSize; f += numThreads) {
            if (found) break;
            
            int node = frontier[f];
            int start = rowPtr[node];
            int end = rowPtr[node + 1];
            
            for (int e = start; e < end; e++) {
                int neighbor = colIdx[e];
                int sn = edgeSeqNo[e];
                
                unsigned char edgeState = myMask[sn];
                if (edgeState == MASK_UP || edgeState == MASK_NOMARK) {
                    //bool wasVisited = atomicExch((int*)&visited[neighbor], 1);
                    
                    //if (!wasVisited) {
                    if (atomicCAS(&visited[neighbor], 0, 1) == 0) {
                        parent[neighbor] = node;
                        parentEdge[neighbor] = sn;
                        
                        if (neighbor == t) {
                            found = true;
                        }
                        
                        int pos = atomicAdd(&nextFrontierSize, 1);
                        if (pos < MAX_NODES_SHARED) {
                            nextFrontier[pos] = neighbor;
                        }
                    }
                }
            }
        }
        __syncthreads();
        
        if (tid == 0) {
            frontierSize = min(nextFrontierSize, MAX_NODES_SHARED);
            nextFrontierSize = 0;
        }
        __syncthreads();
        
        for (int i = tid; i < frontierSize; i += numThreads) {
            frontier[i] = nextFrontier[i];
        }
        __syncthreads();
    }
    
    // Thread 0: trace path and find first unmarked edge
    if (tid == 0) {
        results[maskIdx] = found ? 1 : 0;
        branchEdge[maskIdx] = -1;
        
        if (found) {
            // Trace back from t to src
            int node = t;
            while (node != src) {
                int sn = parentEdge[node];
                if (myMask[sn] == MASK_NOMARK) {
                    branchEdge[maskIdx] = sn;
                    break;  // Found first unmarked edge
                }
                node = parent[node];
            }
        }
    }
}

//=============================================================================
// Kernel 3: Warp-Level Connectivity Check (for small graphs)
//
// More efficient for small graphs - uses warp-level primitives
//=============================================================================
__global__ void connectivityCheckWarpKernel(
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const int* __restrict__ edgeSeqNo,
    int N,
    int src,
    int t,
    const unsigned char* __restrict__ masks,
    int maskSize,
    int batchCount,
    int* __restrict__ results
) {
    // Each warp handles one mask
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    
    if (warpId >= batchCount) return;
    
    const unsigned char* myMask = masks + warpId * maskSize;
    
    // Use warp-level voting for visited tracking
    unsigned int visitedMask = (laneId == src) ? (1u << laneId) : 0;
    visitedMask = __ballot_sync(0xFFFFFFFF, laneId == src);
    
    unsigned int frontier = (1u << src);
    bool found = (src == t);
    
    // BFS using warp primitives (for N <= 32)
    for (int iter = 0; iter < N && !found && frontier != 0; iter++) {
        unsigned int nextFrontier = 0;
        
        // Each lane checks if it's in frontier
        if ((frontier >> laneId) & 1) {
            int node = laneId;
            int start = rowPtr[node];
            int end = rowPtr[node + 1];
            
            for (int e = start; e < end; e++) {
                int neighbor = colIdx[e];
                int sn = edgeSeqNo[e];
                
                unsigned char edgeState = myMask[sn];
                if ((edgeState == MASK_UP || edgeState == MASK_NOMARK) && neighbor < 32) {
                    if (!((visitedMask >> neighbor) & 1)) {
                        nextFrontier |= (1u << neighbor);
                        if (neighbor == t) found = true;
                    }
                }
            }
        }
        
        // Combine across warp
        unsigned mask = __activemask();
        nextFrontier = __any_sync(mask, nextFrontier != 0);
        //nextFrontier = __reduce_or_sync(0xFFFFFFFF, nextFrontier);
        
        found = __any_sync(0xFFFFFFFF, found);
        
        visitedMask |= nextFrontier;
        frontier = nextFrontier;
    }
    
    // Lane 0 writes result
    if (laneId == 0) {
        results[warpId] = found ? 1 : 0;
    }
}

//=============================================================================
// Host Functions
//=============================================================================

void initDeviceGraph(DeviceGraph* dg, 
                     int N, int M, int MaskSize,
                     int* rowPtr, int* colIdx, int* edgeSeqNo,
                     int src, int t) {
    dg->N = N;
    dg->M = M;
    dg->MaskSize = MaskSize;
    dg->src = src;
    dg->t = t;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&dg->d_rowPtr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dg->d_colIdx, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dg->d_edgeSeqNo, M * sizeof(int)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(dg->d_rowPtr, rowPtr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg->d_colIdx, colIdx, M * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg->d_edgeSeqNo, edgeSeqNo, M * sizeof(int), cudaMemcpyHostToDevice));
    
    printf("Device graph initialized: N=%d, M=%d, MaskSize=%d\n", N, M, MaskSize);
}

void freeDeviceGraph(DeviceGraph* dg) {
    CUDA_CHECK(cudaFree(dg->d_rowPtr));
    CUDA_CHECK(cudaFree(dg->d_colIdx));
    CUDA_CHECK(cudaFree(dg->d_edgeSeqNo));
}

void initBatchData(BatchData* batch, int maskSize) {
    batch->maskSize = maskSize;
    batch->count = 0;
    
    size_t masksBytes = BATCH_SIZE * maskSize;
    
    // Allocate pinned host memory (faster transfers)
    CUDA_CHECK(cudaMallocHost(&batch->h_masks, masksBytes));
    CUDA_CHECK(cudaMallocHost(&batch->h_logProbs, BATCH_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&batch->h_results, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&batch->h_branchEdge, BATCH_SIZE * sizeof(int)));
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&batch->d_masks, masksBytes));
    CUDA_CHECK(cudaMalloc(&batch->d_logProbs, BATCH_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&batch->d_results, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&batch->d_branchEdge, BATCH_SIZE * sizeof(int)));
    
    printf("Batch data initialized: BATCH_SIZE=%d, maskSize=%d\n", BATCH_SIZE, maskSize);
}

void freeBatchData(BatchData* batch) {
    CUDA_CHECK(cudaFreeHost(batch->h_masks));
    CUDA_CHECK(cudaFreeHost(batch->h_logProbs));
    CUDA_CHECK(cudaFreeHost(batch->h_results));
    CUDA_CHECK(cudaFreeHost(batch->h_branchEdge));
    
    CUDA_CHECK(cudaFree(batch->d_masks));
    CUDA_CHECK(cudaFree(batch->d_logProbs));
    CUDA_CHECK(cudaFree(batch->d_results));
    CUDA_CHECK(cudaFree(batch->d_branchEdge));
}

void launchConnectivityCheck(DeviceGraph* dg, BatchData* batch, cudaStream_t stream) {
    if (batch->count == 0) return;
    
    // Copy masks to device
    size_t masksBytes = batch->count * batch->maskSize;
    CUDA_CHECK(cudaMemcpyAsync(batch->d_masks, batch->h_masks, masksBytes, 
                               cudaMemcpyHostToDevice, stream));
    
    // Launch kernel: one block per mask
    int numBlocks = batch->count;
    int threadsPerBlock = BFS_THREADS_PER_BLOCK;
    
    // Calculate shared memory size
    size_t sharedMem = 0;  // Already using static shared memory
    
    connectivityCheckKernel<<<numBlocks, threadsPerBlock, sharedMem, stream>>>(
        dg->d_rowPtr, dg->d_colIdx, dg->d_edgeSeqNo,
        dg->N, dg->src, dg->t,
        batch->d_masks, batch->maskSize, batch->count,
        batch->d_results
    );
    
    // Copy results back
    CUDA_CHECK(cudaMemcpyAsync(batch->h_results, batch->d_results, 
                               batch->count * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
}

void launchPathFinding(DeviceGraph* dg, BatchData* batch, cudaStream_t stream) {
    if (batch->count == 0) return;
    
    size_t masksBytes = batch->count * batch->maskSize;
    CUDA_CHECK(cudaMemcpyAsync(batch->d_masks, batch->h_masks, masksBytes,
                               cudaMemcpyHostToDevice, stream));
    
    int numBlocks = batch->count;
    int threadsPerBlock = BFS_THREADS_PER_BLOCK;
    
    pathFindingKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        dg->d_rowPtr, dg->d_colIdx, dg->d_edgeSeqNo,
        dg->N, dg->src, dg->t,
        batch->d_masks, batch->maskSize, batch->count,
        batch->d_results, batch->d_branchEdge
    );
    
    CUDA_CHECK(cudaMemcpyAsync(batch->h_results, batch->d_results,
                               batch->count * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(batch->h_branchEdge, batch->d_branchEdge,
                               batch->count * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
}
