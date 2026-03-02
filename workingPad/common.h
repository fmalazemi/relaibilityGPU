#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <cstring>

/* ------------------------------------------------------------------ */
/*  Compile-time mask width – override with  -DMAX_MASK_WORDS=N       */
/*  Each word holds 16 edges (2 bits/edge).  Default: 16 words = 256  */
/*  edges.  Set to 256 for up to 4096 edges.                          */
/* ------------------------------------------------------------------ */
#ifndef MAX_MASK_WORDS
#define MAX_MASK_WORDS 16
#endif
#define MAX_EDGES_SUPPORTED (MAX_MASK_WORDS * 16)

/* -------- Edge states (2 bits per edge) -------- */
#define EDGE_FAILED   0   /* 0b00 – deleted / non-operational       */
#define EDGE_WORKING  1   /* 0b01 – contracted / operational        */
#define EDGE_UNKNOWN  3   /* 0b11 – undetermined (pivot candidate)  */

/* -------- Queue priorities -------- */
#define Q_HIGH 0
#define Q_MED  1
#define Q_LOW  2
#define NUM_QUEUES 3

/* -------- Default parameters -------- */
#define DEFAULT_QUEUE_CAPACITY  (1 << 20)   /* 1 M items per queue   */
#define DEFAULT_TRUNCATION_EPS  1e-15f
#define DEFAULT_THRESH_HIGH     (-4.60517f) /* ln(0.01)              */
#define DEFAULT_THRESH_LOW      (-9.21034f) /* ln(0.0001)            */

/* -------- CUDA error check -------- */
#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d – %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

/* ================================================================== */
/*  EdgeMask  –  2 bits per edge packed into uint32_t words           */
/* ================================================================== */
struct EdgeMask {
    uint32_t bits[MAX_MASK_WORDS];
};

/* ================================================================== */
/*  WorkItem  –  one node in the factoring tree                       */
/* ================================================================== */
struct WorkItem {
    EdgeMask mask;
    float    log_prob;   /* cumulative log-probability */
};

/* ================================================================== */
/*  Run-time configuration                                            */
/* ================================================================== */
struct Config {
    char     graph_file[512];
    int      src;
    int      dst;
    int      queue_capacity;
    float    truncation_log_eps;   /* ln(eps) */
    float    thresh_high;
    float    thresh_low;
};

#endif /* COMMON_H */
