#ifndef CUDA_RELIABILITY_H
#define CUDA_RELIABILITY_H

//=============================================================================
// CUDA_Reliability.h - CUDA-specific definitions for Network Reliability
//=============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//=============================================================================
// CUDA Configuration Constants
//=============================================================================

#define MAX_Q_SIZE 13000000

// Batch size for parallel connectivity checks
#define BATCH_SIZE 1024

// Threads per block for BFS kernel
#define BFS_THREADS_PER_BLOCK 256

// Maximum nodes supported (for shared memory allocation)
#define MAX_NODES_SHARED 1024

// Maximum edges per node (for coalesced access)
#define MAX_DEGREE 64

// Warp size
#define WARP_SIZE 32

//=============================================================================
// Mask values (same as CPU)
//=============================================================================
#define MASK_UP     0x01
#define MASK_DOWN   0x00
#define MASK_NOMARK 0xFF

//=============================================================================
// CUDA Error Checking Macro
//=============================================================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

//=============================================================================
// Device Graph Structure (CSR format - optimal for GPU)
//=============================================================================
struct DeviceGraph {
    int N;              // Number of nodes
    int M;              // Number of edges
    int MaskSize;       // Number of unique edges
    
    // CSR representation
    int* d_rowPtr;      // Node offsets [N+1]
    int* d_colIdx;      // Edge destinations [M]
    int* d_edgeSeqNo;   // Edge sequence numbers [M]
    
    // Source and terminal
    int src;
    int t;
};

//=============================================================================
// Batch data structure for parallel processing
//=============================================================================
struct BatchData {
    // Host arrays (pinned memory for fast transfer)
    unsigned char* h_masks;         // [BATCH_SIZE * MaskSize]
    double* h_logProbs;             // [BATCH_SIZE]
    int* h_results;                 // [BATCH_SIZE] - 1 if connected, 0 if not
    int* h_branchEdge;              // [BATCH_SIZE] - edge to branch on, -1 if none
    
    // Device arrays
    unsigned char* d_masks;         // [BATCH_SIZE * MaskSize]
    double* d_logProbs;             // [BATCH_SIZE]
    int* d_results;                 // [BATCH_SIZE]
    int* d_branchEdge;              // [BATCH_SIZE]
    
    // Batch info
    int count;                      // Number of masks in current batch
    int maskSize;                   // Size of each mask
};

//=============================================================================
// Function Declarations
//=============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Initialize device graph from host data
void initDeviceGraph(DeviceGraph* dg, 
                     int N, int M, int MaskSize,
                     int* rowPtr, int* colIdx, int* edgeSeqNo,
                     int src, int t);

// Free device graph memory
void freeDeviceGraph(DeviceGraph* dg);

// Initialize batch data structures
void initBatchData(BatchData* batch, int maskSize);

// Free batch data structures
void freeBatchData(BatchData* batch);

// Launch parallel connectivity check
void launchConnectivityCheck(DeviceGraph* dg, BatchData* batch, cudaStream_t stream);

// Launch parallel BFS to find paths and unmarked edges
void launchPathFinding(DeviceGraph* dg, BatchData* batch, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // CUDA_RELIABILITY_H
