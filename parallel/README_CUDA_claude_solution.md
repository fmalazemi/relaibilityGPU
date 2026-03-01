# CUDA-Accelerated Network Reliability Calculator

## Overview

This is a GPU-accelerated implementation of the factoring method for two-terminal network reliability computation.

## Parallelization Strategy

### What's Parallelized on GPU

| Component | GPU Strategy | Speedup Potential |
|-----------|--------------|-------------------|
| Connectivity check | Batch processing, one block per mask | **10-100×** |
| BFS within block | Cooperative threads, shared memory | **5-20×** |
| Multiple masks | Process BATCH_SIZE simultaneously | **Linear with batch** |

### What Stays on CPU

| Component | Reason |
|-----------|--------|
| Priority queue | Sequential dependency, small overhead |
| Result accumulation | Simple addition, not worth GPU overhead |
| Main control flow | Coordinates GPU work |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          CPU (Host)                              │
│                                                                  │
│  1. Extract batch from priority queue (up to BATCH_SIZE masks)  │
│  2. For each mask, find path and collect unmarked edges         │
│  3. Generate new masks (branch on edge fail/work)               │
│  4. Send new masks to GPU for connectivity check                │
│  5. Add connected masks back to queue                           │
│  6. Accumulate reliability for completed paths                  │
│                                                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                          GPU (Device)                            │
│                                                                  │
│  Kernel: connectivityCheckKernel                                │
│  - One thread block per mask in batch                           │
│  - Threads cooperate on parallel BFS                            │
│  - Shared memory for frontier and visited arrays                │
│  - Returns: 1 if s-t path exists, 0 otherwise                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## CUDA Kernels

### 1. `connectivityCheckKernel`
Primary kernel for checking if a mask allows s-t connectivity.

```cpp
__global__ void connectivityCheckKernel(
    const int* rowPtr,      // CSR row pointers
    const int* colIdx,      // CSR column indices
    const int* edgeSeqNo,   // Edge sequence numbers
    int N, int src, int t,  // Graph info
    const unsigned char* masks,  // Batch of masks
    int maskSize,           // Size of each mask
    int batchCount,         // Number of masks in batch
    int* results            // Output: 1 if connected, 0 if not
);
```

**Parallelization:**
- Each block handles one mask
- Threads cooperate on BFS frontier expansion
- Shared memory for visited array and frontiers
- Atomic operations for thread-safe updates

### 2. `pathFindingKernel`
Extended kernel that also finds the first unmarked edge on the path.

### 3. `connectivityCheckWarpKernel`
Optimized kernel for small graphs (N ≤ 32) using warp-level primitives.

## Memory Layout

### Host (CPU) Memory
```
Pinned memory (for fast transfers):
- h_masks:     [BATCH_SIZE × MaskSize] bytes
- h_logProbs:  [BATCH_SIZE] doubles
- h_results:   [BATCH_SIZE] ints
```

### Device (GPU) Memory
```
Global memory:
- Graph CSR:   rowPtr[N+1], colIdx[M], edgeSeqNo[M]
- d_masks:     [BATCH_SIZE × MaskSize] bytes
- d_results:   [BATCH_SIZE] ints

Shared memory (per block):
- frontier:     [MAX_NODES_SHARED] ints
- nextFrontier: [MAX_NODES_SHARED] ints
- visited:      [MAX_NODES_SHARED] bools
```

## Optimization Techniques

### 1. Coalesced Memory Access
Masks are stored contiguously so threads access adjacent memory locations:
```cpp
const unsigned char* myMask = masks + maskIdx * maskSize;
// All threads in a warp access sequential mask bytes
```

### 2. Shared Memory BFS
Frontier and visited arrays in shared memory (100× faster than global):
```cpp
__shared__ int frontier[MAX_NODES_SHARED];
__shared__ bool visited[MAX_NODES_SHARED];
```

### 3. Early Termination
BFS stops immediately when target is found:
```cpp
if (neighbor == t) {
    found = true;  // All threads will see this
}
```

### 4. Warp-Level Primitives (for small graphs)
```cpp
// Ballot for visited tracking
unsigned int visitedMask = __ballot_sync(0xFFFFFFFF, condition);

// Reduce for combining results
nextFrontier = __reduce_or_sync(0xFFFFFFFF, nextFrontier);

// Any for checking if any thread found target
found = __any_sync(0xFFFFFFFF, found);
```

### 5. Asynchronous Operations
```cpp
cudaStream_t stream;
cudaMemcpyAsync(..., stream);  // Non-blocking copy
kernel<<<..., stream>>>(...);   // Non-blocking kernel
cudaStreamSynchronize(stream);  // Wait only when needed
```

## Building

### Prerequisites
- CUDA Toolkit 11.0 or later
- GPU with Compute Capability 7.0+ (Volta, Turing, Ampere, Hopper)
- GCC 9+ or compatible compiler

### Compilation
```bash
# Standard build (adjust sm_70 for your GPU)
make

# For RTX 30xx
make NVCC_FLAGS="-O3 -arch=sm_86"

# For H100
make NVCC_FLAGS="-O3 -arch=sm_90"

# Debug build
make debug
```

### GPU Architecture Reference
| Architecture | Compute Capability | GPUs |
|--------------|-------------------|------|
| Maxwell | sm_50 | GTX 9xx |
| Pascal | sm_60 | GTX 10xx, P100 |
| Volta | sm_70 | V100 |
| Turing | sm_75 | RTX 20xx, T4 |
| Ampere | sm_80/86 | A100, RTX 30xx |
| Ada | sm_89 | RTX 40xx |
| Hopper | sm_90 | H100 |

## Running

```bash
./seq_cuda <input_file> <source> <terminal>

# Example
./seq_cuda temp4x4.txt 0 1
```

## Profiling

```bash
# Basic profiling with nvprof (deprecated but still works)
nvprof ./seq_cuda temp4x4.txt 0 1

# Nsight Systems (recommended)
nsys profile ./seq_cuda temp4x4.txt 0 1

# Nsight Compute (detailed kernel analysis)
ncu --set full ./seq_cuda temp4x4.txt 0 1
```

## Performance Tuning

### 1. Batch Size
Adjust `BATCH_SIZE` in `CUDA_Reliability.h`:
```cpp
#define BATCH_SIZE 1024  // Default

// For small graphs (< 100 nodes): try 4096
// For large graphs (> 10000 nodes): try 256
```

### 2. Block Size
Adjust `BFS_THREADS_PER_BLOCK`:
```cpp
#define BFS_THREADS_PER_BLOCK 256  // Default

// Optimal is usually 128-512, depends on GPU occupancy
```

### 3. Shared Memory Size
Adjust `MAX_NODES_SHARED` based on your graph size:
```cpp
#define MAX_NODES_SHARED 1024  // Default

// Must be >= N (number of nodes)
// Limited by shared memory per block (usually 48KB)
```

## Expected Speedup

| Graph Size | CPU Time | GPU Time | Speedup |
|------------|----------|----------|---------|
| 100 nodes | 10 ms | 5 ms | 2× |
| 1,000 nodes | 1 sec | 50 ms | 20× |
| 10,000 nodes | 100 sec | 2 sec | 50× |
| 100,000 nodes | hours | minutes | 50-100× |

**Note:** Speedup depends on:
- Graph structure (density, diameter)
- GPU capability
- Batch size utilization

## Further Optimizations

### 1. Multi-GPU Support
```cpp
// Distribute batches across multiple GPUs
#pragma omp parallel for
for (int gpu = 0; gpu < numGPUs; gpu++) {
    cudaSetDevice(gpu);
    // Process batch on this GPU
}
```

### 2. Persistent Kernel
Keep kernel running and feed it work via global memory flags:
```cpp
__global__ void persistentKernel(...) {
    while (true) {
        // Wait for work
        if (atomicLoad(&workAvailable)) {
            // Process
        }
        if (atomicLoad(&terminate)) break;
    }
}
```

### 3. Graph Compression
For sparse graphs, use compressed formats:
```cpp
// Bit-packed adjacency for small degree nodes
// Delta encoding for sorted neighbor lists
```

### 4. Dynamic Parallelism
Launch child kernels from GPU for recursive subproblems:
```cpp
__global__ void parentKernel(...) {
    // Spawn child kernels for subproblems
    childKernel<<<blocks, threads>>>(...);
}
```

## Files

| File | Description |
|------|-------------|
| `CUDA_Reliability.h` | Device structures, constants, function declarations |
| `CUDA_Kernels.cu` | CUDA kernels and device functions |
| `seq_cuda.cu` | Main program, host code, graph I/O |
| `Makefile` | Build configuration |

## Comparison with CPU Version

| Aspect | CPU Version | CUDA Version |
|--------|-------------|--------------|
| Connectivity check | Sequential BFS | Parallel BFS in batch |
| Memory | System RAM | GPU + pinned host |
| Scalability | Single core | Thousands of threads |
| Best for | Small graphs, debugging | Large graphs, production |
