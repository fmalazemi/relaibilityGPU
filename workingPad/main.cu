/* ================================================================== *
 *  main.cu  –  CUDA Two-Terminal Network Reliability via Factoring   *
 *                                                                    *
 *  Build:  nvcc -O3 -arch=sm_70 main.cu -o network_reliability      *
 *  Run:    ./network_reliability <graph> <src> <dst> [options]       *
 * ================================================================== */

#include "common.h"
#include "graph.h"
#include "device_code.cuh"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <climits>



#define USE_MAX_MEMORY false
#define FORK_BEST_EDGE false  


/* ================================================================== */
/*  Single-queue (read / write) structures                            */
/*  Read queue   – consumed by the kernel (immutable during launch).  */
/*  Write queue  – append-only by the kernel (atomic counter).        */
/* ================================================================== */

/* ---- device-side read queue ---- */
struct ReadQueue {
    const WorkItem* buffer;   /* device ptr */
    int             size;     /* items available */
    int*            next_idx; /* device: atomic dequeue counter */
};

/* ---- device-side write queue ---- */
struct WriteQueue {
    WorkItem* buffer;   /* device ptr */
    int       capacity;
    int*      count;    /* device: atomic append counter */
};

/* ================================================================== */
/*  Statistics (device-side)                                          */
/*                                                                    */
/*  R    – global atomic accumulator for the reliability value.       */
/*         Each terminal (all-WORKING) path adds its probability      */
/*         exp(log_prob) directly here.                               */
/*                                                                    */
/*  esp_truncated – global atomic accumulator for the truncation /    */
/*         overflow bound (a.k.a. ESP).  Two contributions:           */
/*           1) WorkItems whose log_prob falls below the truncation  */
/*              epsilon (handled in factoring_kernel).                */
/*           2) Children that could not be enqueued because the       */
/*              write queue was full (handled in path_edge_sweep).   */
/* ================================================================== */
struct DeviceStats {
    double*               R;                  /* atomic (CAS) – reliability */
    double*               esp_truncated;      /* atomic (CAS) – ESP bound   */
    unsigned long long*   paths_enumerated;   /* atomic */
    unsigned long long*   nodes_processed;    /* atomic */
};

/* ================================================================== */
/*  atomicAdd for double via CAS loop                                 */
/* ================================================================== */
__device__ __forceinline__
void atomicAddDouble(double* addr, double val)
{
    unsigned long long* ull = (unsigned long long*)addr;
    unsigned long long  old = *ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(ull, assumed,
                __double_as_longlong(
                    __longlong_as_double(assumed) + val));
    } while (assumed != old);
}




/* ================================================================== */
/*  Path-edge sweep  (warp-parallel)                                  */
/*                                                                    */
/*  Given the parent[] array from the optimistic BFS, enumerates     */
/*  every UNKNOWN edge on the src→dst path and spawns one child       */
/*  WorkItem per such edge:                                           */
/*                                                                    */
/*    child for hop i:                                                */
/*      - all UNKNOWN hops 0 .. i-1  → WORKING  (+ log_p each)      */
/*      - hop i                      → FAILED   (+ log_q)            */
/*      - all hops > i               unchanged from cur_mask          */
/*                                                                    */
/*  PHASE 1 (lane 0)  : walk parent[] dst→src, reverse to src→dst,  */
/*                       write into s_path_nodes / s_path_len.        */
/*  PHASE 2 (all lanes): CSR lookup to resolve edge id per hop.      */
/*  PHASE 3 (all lanes): build child mask + enqueue, one lane/edge.  */
/*                                                                    */
/*  Returns the number of UNKNOWN edges found on the path (0 means   */
/*  the path was all-WORKING → caller should handle as terminal or   */
/*  fall through to confirmed BFS).                                   */
/*                                                                    */
/*  Caller-allocated shared scratch (must be in the kernel's smem    */
/*  layout after the BFS bitmasks and mask copy):                    */
/*    s_path_nodes [MAX_PATH_LEN]  int                               */
/*    s_path_eids  [MAX_PATH_LEN]  int                               */
/*    s_path_len   [1]             int                               */
/* ================================================================== */

#define MAX_PATH_LEN MAX_NODE_WORDS   /* = MAX_NODE_WORDS * 32 */

__device__
int path_edge_sweep_warp(
    /* graph */
    const int*   g_row_ptr,
    const int*   g_col_idx,
    const int*   g_edge_id_arr,
    int          g_src,
    int          g_dst,
    /* current item (read-only; each lane builds its own child copy)  */
    const EdgeMask* cur_mask,
    float           cur_log_prob,
    /* BFS parent array filled by optimistic BFS (size g_N)           */
    const volatile int* s_parent,
    /* edge probabilities */
    const float* g_log_p,
    const float* g_log_q,
    /* single write queue */
    WriteQueue*  wq,
    /* shared scratch */
    int*         s_path_nodes,   /* [MAX_PATH_LEN] */
    int*         s_path_eids,    /* [MAX_PATH_LEN] */
    int*         s_path_len,      /* [1]            */
    DeviceStats  stats)
{
    int lane = threadIdx.x & 31;
    
    /* ============================================================== *
    *  PHASE 1 – lane 0 only                                         *
    *  Walk parent[] from g_dst back to g_src (pointer-chasing,      *
    *  inherently serial), then reverse in-place so:                 *
    *    s_path_nodes[0]           = g_src                           *
    *    s_path_nodes[path_len-1]  = g_dst                           *
    * ============================================================== */
    if (lane == 0) {
        int len = 0;
        
        /* collect dst → src */
        int v = g_dst;
        while (v != -1 && len < MAX_PATH_LEN) {
            s_path_nodes[len++] = v;
            if (v == g_src) break;
            v = s_parent[v];
        }
        
        /* reverse in-place → src → dst */
        for (int lo = 0, hi = len - 1; lo < hi; lo++, hi--) {
            int tmp          = s_path_nodes[lo];
            s_path_nodes[lo] = s_path_nodes[hi];
            s_path_nodes[hi] = tmp;
        }
        *s_path_len = len;
    }
    __syncwarp();
    
    int path_len = *s_path_len;
    int n_hops   = path_len - 1;   /* one edge per consecutive node pair */
    if (n_hops <= 0) return 0;
    
    /* ============================================================== *
    *  PHASE 2 – all 32 lanes, strided by 32                         *
    *  Each lane resolves the global edge id for its assigned hop(s) *
    *  by scanning the CSR row of node u looking for neighbour v.    *
    *                                                                *
    *  s_path_eids[hop] = -1 signals a missing edge (bug guard).    *
    * ============================================================== */
    for (int hop = lane; hop < n_hops; hop += 32) {
        int u         = s_path_nodes[hop];
        int v         = s_path_nodes[hop + 1];
        int eid       = -1;
        int row_start = __ldg(&g_row_ptr[u]);
        int row_end   = __ldg(&g_row_ptr[u + 1]);
        
        for (int j = row_start; j < row_end; j++) {
            if (__ldg(&g_col_idx[j]) == v) {
                eid = __ldg(&g_edge_id_arr[j]);
                break;
            }
        }
        s_path_eids[hop] = eid;
    }
    __syncwarp();
    
    /* ============================================================== *
    *  PHASE 3 – all 32 lanes, one lane per UNKNOWN edge             *
    *                                                                *
    *  Lane assigned to hop i (via stride-32 distribution):          *
    *    • Skip if the edge is not UNKNOWN in cur_mask.              *
    *    • Start from cur_mask (the parent item's full state).       *
    *    • For all hops j < i that are UNKNOWN: set → WORKING,       *
    *      accumulate log_p[eid_j] into child.log_prob.              *
    *    • Set hop i → FAILED, accumulate log_q[eid_i].              *
    *    • Enqueue the resulting WorkItem into the single queue.    *
    *      If the queue is full, fold the child's mass into ESP.    *
    *                                                                *
    *  All reads are from cur_mask (const, no race).                 *
    *  Each lane writes only to its own private child (registers /   *
    *  local memory) before a single atomic enqueue.                 *
    * ============================================================== */
    int n_unknown = 0;   /* count UNKNOWN hops – return value */
    
    for (int hop = lane; hop < n_hops; hop += 32) {
        int eid_i = s_path_eids[hop];
        
        /* skip WORKING edges (already contracted) and gap sentinels  */
        if (eid_i < 0 || mask_get(cur_mask, eid_i) != EDGE_UNKNOWN)
            continue;
        
        n_unknown++;   /* local count; we just need > 0 check         */
        
        /* ---- build child mask ---- */
        WorkItem child;
        child.mask     = *cur_mask;     /* copy parent state          */
        child.log_prob = cur_log_prob;
        
        /* commit all UNKNOWN hops before hop i → WORKING             */
        for (int h = 0; h < hop; h++) {
            int eid_h = s_path_eids[h];
            if (eid_h >= 0 && mask_get(cur_mask, eid_h) == EDGE_UNKNOWN) {
                mask_set(&child.mask, eid_h, EDGE_WORKING);
                child.log_prob += __ldg(&g_log_p[eid_h]);
            }
        }
        
        /* set hop i → FAILED                                         */
        mask_set(&child.mask, eid_i, EDGE_FAILED);
        child.log_prob += __ldg(&g_log_q[eid_i]);
        
        /* ---- enqueue into the single write queue ----              *
         *  If the queue is full (pos >= capacity) the child cannot  *
         *  be explored – fold its probability mass into the ESP     *
         *  bound so the final R + ESP stays a valid upper bound.    */
        int pos = atomicAdd(wq->count, 1);
        if (pos < wq->capacity) {
            wq->buffer[pos] = child;
        } else {
            atomicAddDouble(stats.esp_truncated,
                            exp((double)child.log_prob));
	    atomicAdd(stats.paths_enumerated, 1ULL);
        }
        
        if(hop == n_hops-1){
            //mask_set(&child.mask, eid_i, EDGE_WORKING);
            child.log_prob += __ldg(&g_log_p[eid_i]);
            child.log_prob -= __ldg(&g_log_q[eid_i]);
            
            /* ---- terminal (all-WORKING) path ----                  *
             *  Add this path's probability directly to R.            */
            atomicAddDouble(stats.R, exp((double)child.log_prob));
            atomicAdd(stats.paths_enumerated, 1ULL);
            
            
            
        }
        
        
        
    }
    
    
    
    
    __syncwarp();
    
    /* Warp-reduce n_unknown so all lanes agree on the return value.  */
    for (int off = 16; off > 0; off >>= 1)
        n_unknown += __shfl_down_sync(0xFFFFFFFF, n_unknown, off);
    n_unknown = __shfl_sync(0xFFFFFFFF, n_unknown, 0);
    
    return n_unknown;
}

/* ================================================================== */
/*  Path-pivot factoring  (warp-parallel, two-way branching)         */
/*                                                                    */
/*  1. Reconstruct the src→dst path from the BFS parent[] array.     */
/*  2. Find the UNKNOWN edge on the path with the highest log_p.    */
/*  3. If no UNKNOWN edge exists  → path is all-WORKING → terminal   */
/*     success: add exp(cur_log_prob) to R and return.               */
/*  4. Otherwise fork two children:                                  */
/*       WORKING:  mask[e] = WORKING,  log_prob += log_p[e]          */
/*       FAILED :  mask[e] = FAILED,   log_prob += log_q[e]          */
/*                                                                    */
/*  PHASE 1 (lane 0)  : walk parent[] dst → src                      */
/*  PHASE 2 (all)     : CSR lookup + per-lane local argmax          */
/*  PHASE 3 (all)     : warp reduction → broadcast pivot            */
/*  PHASE 4 (lanes 0,1): build + enqueue the two children           */
/*                                                                    */
/*  Returns 1 if a pivot was found and forked, 0 if terminal.        */
/* ================================================================== */
__device__
int path_edge_sweep_warp_fork_best_edge(
    /* graph */
    const int*   g_row_ptr,
    const int*   g_col_idx,
    const int*   g_edge_id_arr,
    int          g_src,
    int          g_dst,
    /* current item (read-only; each lane has its own register copy) */
    const EdgeMask* cur_mask,
    float           cur_log_prob,
    /* BFS parent array filled by the optimistic BFS                 */
    const volatile int* s_parent,
    /* edge probabilities */
    const float* g_log_p,
    const float* g_log_q,
    /* single write queue */
    WriteQueue*  wq,
    /* shared scratch */
    int*         s_path_nodes,   /* [MAX_PATH_LEN] */
    int*         s_path_eids,    /* [MAX_PATH_LEN] – unused now      */
    int*         s_path_len,     /* [1]            */
    DeviceStats  stats)
{
    int lane = threadIdx.x & 31;
    
    /* ============================================================== *
    *  PHASE 1 – lane 0 walks parent[] dst → src                    *
    * ============================================================== */
    if (lane == 0) {
        int len = 0;
        int v   = g_dst;
        while (v != -1 && len < MAX_PATH_LEN) {
            s_path_nodes[len++] = v;
            if (v == g_src) break;
            v = s_parent[v];
        }
        *s_path_len = len;
    }
    __syncwarp();
    
    int path_len = *s_path_len;
    int n_hops   = path_len - 1;
    if (n_hops <= 0) return 0;
    
    /* ============================================================== *
    *  PHASE 2 – per-lane local argmax over UNKNOWN path edges      *
    *                                                                *
    *  Each lane handles a stride-32 subset of hops.  For each hop  *
    *  it resolves the edge id via CSR scan and, if UNKNOWN,        *
    *  tracks the highest log_p seen in registers.                  *
    * ============================================================== */
    int   best_edge = -1;
    float best_lp   = -1e30f;
    
    for (int hop = lane; hop < n_hops; hop += 32) {
        int u = s_path_nodes[hop];
        int v = s_path_nodes[hop + 1];
        
        /* CSR scan for the (u, v) edge id */
        int eid = -1;
        int rs  = __ldg(&g_row_ptr[u]);
        int re  = __ldg(&g_row_ptr[u + 1]);
        for (int j = rs; j < re; j++) {
            if (__ldg(&g_col_idx[j]) == v) {
                eid = __ldg(&g_edge_id_arr[j]);
                break;
            }
        }
        
        if (eid >= 0 && mask_get(cur_mask, eid) == EDGE_UNKNOWN) {
            float lp = __ldg(&g_log_p[eid]);
            if (lp > best_lp) { best_lp = lp; best_edge = eid; }
        }
    }
    
    /* ============================================================== *
    *  PHASE 3 – warp reduction: argmax across the 32 lanes         *
    * ============================================================== */
    for (int off = 16; off > 0; off >>= 1) {
        float o_lp = __shfl_down_sync(0xFFFFFFFF, best_lp,   off);
        int   o_e  = __shfl_down_sync(0xFFFFFFFF, best_edge, off);
        if (o_lp > best_lp) { best_lp = o_lp; best_edge = o_e; }
    }
    best_edge = __shfl_sync(0xFFFFFFFF, best_edge, 0);
    best_lp   = __shfl_sync(0xFFFFFFFF, best_lp,   0);
    
    /* ============================================================== *
    *  PHASE 4 – act on the pivot                                   *
    * ============================================================== */
    if (best_edge < 0) {
        /* No UNKNOWN edge on the path → all-WORKING terminal.        *
        * Add this branch's probability mass directly to R.          */
        if (lane == 0) {
            atomicAddDouble(stats.R, exp((double)cur_log_prob));
        }
        return 0;
    }

    /* Two-way fork: lane 0 → WORKING child, lane 1 → FAILED child.   */
    if (lane < 2) {
        WorkItem child;
        child.mask     = *cur_mask;
        child.log_prob = cur_log_prob;
        
        if (lane == 0) {
            /* WORKING branch — multiply by p_e */
            mask_set(&child.mask, best_edge, EDGE_WORKING);
            child.log_prob += best_lp;                       /* = log_p[e] */
        } else {
            /* FAILED branch — multiply by 1 - p_e */
            mask_set(&child.mask, best_edge, EDGE_FAILED);
            child.log_prob += __ldg(&g_log_q[best_edge]);
        }
        
        /* enqueue, or fold into ESP if the queue is full */
        int pos = atomicAdd(wq->count, 1);
        if (pos < wq->capacity) {
            wq->buffer[pos] = child;
        } else {
            atomicAddDouble(stats.esp_truncated,
                exp((double)child.log_prob));
	    atomicAdd(stats.paths_enumerated, 1ULL);
        }
    }
    
    return 1;
}













/* ================================================================== */
/*  FACTORING  KERNEL                                                 */
/*                                                                    */
/*  Each block = 1 warp (32 threads).                                 */
/*  1. Dequeue one WorkItem from the single read queue.               */
/*  2. Truncation check.                                              */
/*  3. Optimistic BFS – prune dead ends; fills parent[] array.        */
/*  4. Confirmed  BFS – detect terminal successes.                    */
/*  5. Path-edge sweep – warp-parallel child generation.              */
/*                                                                    */
/*  Shared-memory layout (base → top):                                */
/*    uint32_t  s_frontier     [node_words]                           */
/*    uint32_t  s_visited      [node_words]                           */
/*    uint32_t  s_next_frontier[node_words]                           */
/*    uint32_t  s_mask_bits    [MAX_MASK_WORDS]                       */
/*    float     s_log_prob     [1]                                    */
/*    int       s_parent       [g_N]          (optimistic BFS output) */
/*    int       s_path_nodes   [MAX_PATH_LEN] (path reconstruction)   */
/*    int       s_path_eids    [MAX_PATH_LEN] (edge ids on path)      */
/*    int       s_path_len     [1]                                    */
/* ================================================================== */
__global__ void factoring_kernel(
    /* single read queue */
    ReadQueue   rq,
    /* single write queue */
    WriteQueue  wq,
    /* graph */
    const int*   g_row_ptr,
    const int*   g_col_idx,
    const int*   g_edge_id,
    const float* g_log_p,
    const float* g_log_q,
    int          g_N, int g_E, int g_src, int g_dst,
    int          g_num_mask_words,
    /* thresholds */
    float truncation_log_eps,
    /* stats */
    DeviceStats  stats)
{
    int lane = threadIdx.x & 31;   /* 0..31 */

    int node_words = (g_N + 31) >> 5;

    /* ---- shared-memory layout ---- */
    extern __shared__ uint32_t smem[];

    volatile uint32_t* s_frontier      = smem;
    volatile uint32_t* s_visited       = smem + node_words;
    volatile uint32_t* s_next_frontier = smem + 2 * node_words;
    uint32_t*          s_mask_bits     = (uint32_t*)(smem + 3 * node_words);
    float*             s_log_prob_ptr  = (float*)(s_mask_bits + MAX_MASK_WORDS);

    /* parent[], path_nodes[], path_eids[], path_len sit after the    */
    /* existing region.  Cast via char* to avoid alignment issues.    */
    int* s_parent     = (int*)(s_log_prob_ptr + 1);          /* [g_N]          */
    int* s_path_nodes = s_parent     + g_N;                  /* [MAX_PATH_LEN] */
    int* s_path_eids  = s_path_nodes + MAX_PATH_LEN;         /* [MAX_PATH_LEN] */
    int* s_path_len   = s_path_eids  + MAX_PATH_LEN;         /* [1]            */

    /* ---- main work loop ---- */
    for (;;) {
        /* ---- 1. DEQUEUE ---- */
        WorkItem item;
        int got = 0;

        if (lane == 0) {
            int idx = atomicAdd(rq.next_idx, 1);
            if (idx < rq.size) { item = rq.buffer[idx]; got = 1; }
            if (got) {
                /* copy to shared mem for warp-wide access */
                for (int w = 0; w < g_num_mask_words; w++)
                    s_mask_bits[w] = item.mask.bits[w];
                for (int w = g_num_mask_words; w < MAX_MASK_WORDS; w++)
                    s_mask_bits[w] = 0;
                *s_log_prob_ptr = item.log_prob;
            }
        }
        got = __shfl_sync(0xFFFFFFFF, got, 0);
        if (!got) return;   /* nothing left in the read queue */

        __syncwarp();

        /* read mask + log_prob into registers (all lanes) */
        EdgeMask cur_mask;
        for (int w = 0; w < MAX_MASK_WORDS; w++)
            cur_mask.bits[w] = s_mask_bits[w];
        float cur_log_prob = *s_log_prob_ptr;

        /* count as processed */
        if (lane == 0)
            atomicAdd(stats.nodes_processed, 1ULL);

        /* ---- 2. TRUNCATION CHECK ---- */
        if (cur_log_prob < truncation_log_eps) {
            if (lane == 0){
                atomicAddDouble(stats.esp_truncated,
                                exp((double)cur_log_prob));
	    }
            continue;
        }

        /* ---- 3. OPTIMISTIC BFS  (WORKING + UNKNOWN edges) ----     */
        /*  Pass s_parent so the BFS records predecessors.  The array */
        /*  is initialised to -1 inside bfs_reachable before use.     */


	if(FORK_BEST_EDGE){
		int opt_reach = bfs_reachable(
            		g_row_ptr, g_col_idx, g_edge_id, g_N,
            		&cur_mask, g_src, g_dst,
            		/*allow_unknown=*/ 0,
            		s_frontier, s_visited, s_next_frontier,
            		node_words,
            		/*parent=*/ (volatile int*)s_parent);
		//A working path exists. no need to do forks
        	if (opt_reach) {
            		if (lane == 0){
                		atomicAddDouble(stats.R, exp((double)cur_log_prob));
				atomicAdd(stats.paths_enumerated, 1ULL);
        		}
			continue; 
		}

	}



        int opt_reach = bfs_reachable(
            g_row_ptr, g_col_idx, g_edge_id, g_N,
            &cur_mask, g_src, g_dst,
            /*allow_unknown=*/ 1,
            s_frontier, s_visited, s_next_frontier,
            node_words,
            /*parent=*/ (volatile int*)s_parent);

        if (!opt_reach) {
            /* dst unreachable even optimistically → dead end */
            continue;
        }

        
        /* ---- 5. PATH-EDGE SWEEP ----                                *
         *                                                             *
         * The optimistic BFS found a src→dst path and recorded it in *
         * s_parent[].  path_edge_sweep_warp:                         *
         *   - lane 0   reconstructs the path into s_path_nodes[]     *
         *   - all lanes resolve edge ids (CSR lookup, phase 2)       *
         *   - all lanes build + enqueue one child per UNKNOWN edge   *
         *     on the path (phase 3) – terminal cases go to R,        *
         *     queue-full cases go to ESP.                            */
        /*int n_spawned = path_edge_sweep_warp(
            g_row_ptr, g_col_idx, g_edge_id,
            g_src, g_dst,
            &cur_mask, cur_log_prob,
            (const volatile int*)s_parent,
            g_log_p, g_log_q,
            &wq,
            s_path_nodes, s_path_eids, s_path_len,
            stats);
        */
        int n_spawned ; 
	if(FORK_BEST_EDGE)
		n_spawned = path_edge_sweep_warp_fork_best_edge(
            g_row_ptr, g_col_idx, g_edge_id,
            g_src, g_dst,
            &cur_mask, cur_log_prob,
            (const volatile int*)s_parent,
            g_log_p, g_log_q,
            &wq,
            s_path_nodes, s_path_eids, s_path_len,
            stats);
	else
		n_spawned = path_edge_sweep_warp(
            g_row_ptr, g_col_idx, g_edge_id,
            g_src, g_dst,
            &cur_mask, cur_log_prob,
            (const volatile int*)s_parent,
            g_log_p, g_log_q,
            &wq,
            s_path_nodes, s_path_eids, s_path_len,
            stats);
	
        

    } /* end work loop */
}

/* ================================================================== */
/*  HOST-SIDE  QUEUE  MANAGEMENT                                      */
/* ================================================================== */

struct HostQueuePair {
    /* device buffers */
    WorkItem*  buf_A;      /* ping */
    WorkItem*  buf_B;      /* pong */
    int*       d_counter;  /* device atomic counter */
    int        capacity;

    /* which buffer is currently "read" vs "write" */
    WorkItem*  read_buf;
    int        read_size;
    WorkItem*  write_buf;
};

static void queue_pair_init(HostQueuePair* qp, int capacity)
{
    qp->capacity = capacity;
    CUDA_CHECK(cudaMalloc(&qp->buf_A, (size_t)capacity * sizeof(WorkItem)));
    CUDA_CHECK(cudaMalloc(&qp->buf_B, (size_t)capacity * sizeof(WorkItem)));
    CUDA_CHECK(cudaMalloc(&qp->d_counter, sizeof(int)));

    qp->read_buf  = qp->buf_A;
    qp->read_size = 0;
    qp->write_buf = qp->buf_B;

    int zero = 0;
    CUDA_CHECK(cudaMemcpy(qp->d_counter, &zero, sizeof(int),
                           cudaMemcpyHostToDevice));
}

static void queue_pair_free(HostQueuePair* qp)
{
    cudaFree(qp->buf_A);
    cudaFree(qp->buf_B);
    cudaFree(qp->d_counter);
}

/* After a kernel launch: read back how many items were written,
   swap read/write buffers.  Returns the number of new items.        */
static int queue_pair_swap(HostQueuePair* qp)
{
    int written = 0;
    CUDA_CHECK(cudaMemcpy(&written, qp->d_counter, sizeof(int),
                           cudaMemcpyDeviceToHost));
    if (written > qp->capacity) written = qp->capacity; /* overflow cap */

    /* swap */
    WorkItem* old_write = qp->write_buf;
    qp->write_buf  = qp->read_buf;
    qp->read_buf   = old_write;
    qp->read_size  = written;

    /* reset write counter */
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(qp->d_counter, &zero, sizeof(int),
                           cudaMemcpyHostToDevice));
    return written;
}

/* Prepare a ReadQueue struct for kernel launch */
static ReadQueue make_read_queue(HostQueuePair* qp, int* d_next_idx)
{
    ReadQueue rq;
    rq.buffer   = qp->read_buf;
    rq.size     = qp->read_size;
    rq.next_idx = d_next_idx;
    return rq;
}

/* Prepare a WriteQueue struct for kernel launch */
static WriteQueue make_write_queue(HostQueuePair* qp)
{
    WriteQueue wq;
    wq.buffer   = qp->write_buf;
    wq.capacity = qp->capacity;
    wq.count    = qp->d_counter;
    return wq;
}

/* ================================================================== */
/*  Initial mask: set p=0 edges to FAILED, p=1 to WORKING,           */
/*  rest to UNKNOWN.                                                  */
/* ================================================================== */
static EdgeMask build_initial_mask(const GraphHost& g)
{
    EdgeMask m;
    memset(&m, 0, sizeof(m));

    /* start all-unknown */
    int total_bits = g.E * 2;
    int full_words = total_bits / 32;
    int rem        = total_bits % 32;
    for (int i = 0; i < full_words; i++) m.bits[i] = 0xFFFFFFFF;
    if (rem > 0) m.bits[full_words] = (1u << rem) - 1;

    /* pre-resolve deterministic edges */
    int pre_contract = 0, pre_delete = 0;
    for (int e = 0; e < g.E; e++) {
        if (g.prob[e] <= 0.0f) {
            int bp = e * 2, w = bp / 32, o = bp % 32;
            m.bits[w] &= ~(0x3u << o);   /* set 0b00 = FAILED  */
            pre_delete++;
        } else if (g.prob[e] >= 1.0f) {
            int bp = e * 2, w = bp / 32, o = bp % 32;
            m.bits[w] &= ~(0x3u << o);
            m.bits[w] |=  (0x1u << o);   /* set 0b01 = WORKING */
            pre_contract++;
        }
    }
    if (pre_contract || pre_delete)
        printf("Preprocessing: %d edges contracted (p=1), "
               "%d deleted (p=0)\n", pre_contract, pre_delete);

    return m;
}

/* ================================================================== */
/*  Parse command line                                                */
/* ================================================================== */
static Config parse_args(int argc, char** argv)
{
    Config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.queue_capacity     = DEFAULT_QUEUE_CAPACITY;
    cfg.truncation_log_eps = logf(DEFAULT_TRUNCATION_EPS);
    cfg.thresh_high        = DEFAULT_THRESH_HIGH;
    cfg.thresh_low         = DEFAULT_THRESH_LOW;

    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <graph_file> <src> <dst> [options]\n"
            "Options:\n"
            "  --truncation_eps <float>   Truncation epsilon (default 1e-15)\n"
            "  --queue_capacity <int>     Per-queue capacity  (default %d)\n"
            "  --thresh_high <float>      High-priority log-prob threshold\n"
            "  --thresh_low  <float>      Low-priority  log-prob threshold\n",
            argv[0], DEFAULT_QUEUE_CAPACITY);
        exit(1);
    }

    strncpy(cfg.graph_file, argv[1], sizeof(cfg.graph_file) - 1);
    cfg.src = atoi(argv[2]);
    cfg.dst = atoi(argv[3]);

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--truncation_eps") == 0 && i + 1 < argc) {
            cfg.truncation_log_eps = logf(atof(argv[++i]));
        } else if (strcmp(argv[i], "--queue_capacity") == 0 && i + 1 < argc) {
            cfg.queue_capacity = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--thresh_high") == 0 && i + 1 < argc) {
            cfg.thresh_high = atof(argv[++i]);
        } else if (strcmp(argv[i], "--thresh_low") == 0 && i + 1 < argc) {
            cfg.thresh_low = atof(argv[++i]);
        }
    }
    return cfg;
}

/* ================================================================== */
/*  MAIN                                                              */
/* ================================================================== */
int main(int argc, char** argv)
{
    Config cfg = parse_args(argc, argv);

    /* ---- load graph ---- */
    GraphHost gh = graph_load(cfg.graph_file, cfg.src, cfg.dst);

    /* ---- transfer to device ---- */
    GraphDevice gd = graph_to_device(gh);

    /* ---- initial mask ---- */
    EdgeMask init_mask = build_initial_mask(gh);
    WorkItem init_item;
    init_item.mask     = init_mask;
    init_item.log_prob = 0.0f;   /* probability = 1 initially */

    /* ================================================================ */
    /*  AUTO-SIZE THE QUEUE: as large as fits in remaining GPU memory.  */
    /*  We need to fit per "unit" of capacity:                          */
    /*    - 2 × sizeof(WorkItem)  (read + write ping-pong buffers)      */
    /*  Reserve 256 MB for stats counters, smem, driver overhead, etc.  */
    /* ================================================================ */
    {
        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

        size_t reserve_bytes = (size_t)256 * 1024 * 1024;   /* 256 MB */
        size_t available = (free_bytes > reserve_bytes)
                              ? (free_bytes - reserve_bytes)
                              : (free_bytes / 2);

        size_t per_unit = 2 * sizeof(WorkItem);
        size_t max_cap  = available / per_unit;

        /* clamp to INT_MAX (queue counters are int) */
        if (max_cap > (size_t)INT_MAX) max_cap = (size_t)INT_MAX;
        if (max_cap < 1) max_cap = 1;

        //Uncomment below if you want to dynamically allocate all space
        //Warning: program may run very slow. 
        if(USE_MAX_MEMORY)
		cfg.queue_capacity = (int)max_cap;

        printf("GPU memory: free=%.2f GB, total=%.2f GB\n",
               free_bytes  / (1024.0 * 1024.0 * 1024.0),
               total_bytes / (1024.0 * 1024.0 * 1024.0));
        printf("Auto-sized queue capacity: %d items "
               "(%.2f MB per buffer × 2)\n",
               cfg.queue_capacity,
               (cfg.queue_capacity * (double)sizeof(WorkItem)) / (1024.0 * 1024.0));
    }

    /* ---- allocate the single queue ---- */
    HostQueuePair queues[NUM_QUEUES];
    for (int q = 0; q < NUM_QUEUES; q++)
        queue_pair_init(&queues[q], cfg.queue_capacity);

    /* seed Q_HIGH (the only queue) with the initial work item */
    CUDA_CHECK(cudaMemcpy(queues[Q_HIGH].read_buf, &init_item,
                           sizeof(WorkItem), cudaMemcpyHostToDevice));
    queues[Q_HIGH].read_size = 1;

    /* ---- allocate per-queue dequeue counter ---- */
    int* d_next_idx[NUM_QUEUES];
    for (int q = 0; q < NUM_QUEUES; q++)
        CUDA_CHECK(cudaMalloc(&d_next_idx[q], sizeof(int)));

    /* ---- allocate stats ---- */
    DeviceStats dstats;
    CUDA_CHECK(cudaMalloc(&dstats.R,                sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dstats.esp_truncated,    sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dstats.paths_enumerated, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&dstats.nodes_processed,  sizeof(unsigned long long)));

    /* zero stats */
    {
        double z = 0.0;
        CUDA_CHECK(cudaMemcpy(dstats.R, &z, sizeof(double),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dstats.esp_truncated, &z, sizeof(double),
                               cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(dstats.paths_enumerated, 0,
                           sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(dstats.nodes_processed, 0,
                           sizeof(unsigned long long)));
    /* ---- determine launch config ---- */
    int device_id = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    int num_sms           = prop.multiProcessorCount;
    //int n_blocks          = num_sms * 2;   /* 2 warps per SM */
    int threads_per_block = 32;            /* exactly 1 warp */

    int node_words = (gh.N + 31) / 32;

    /* ---- shared-memory size ----
     *
     * Region                          Elements          Bytes
     * -------                         --------          -----
     * s_frontier                       node_words        × 4
     * s_visited                        node_words        × 4
     * s_next_frontier                  node_words        × 4
     * s_mask_bits                      MAX_MASK_WORDS    × 4
     * s_log_prob                       1                 × 4  (float)
     * s_parent         [gh.N]          gh.N              × 4  (int)
     * s_path_nodes     [MAX_PATH_LEN]  MAX_PATH_LEN      × 4  (int)
     * s_path_eids      [MAX_PATH_LEN]  MAX_PATH_LEN      × 4  (int)
     * s_path_len       [1]             1                 × 4  (int)
     */
    int smem_bytes =
        (3 * node_words + MAX_MASK_WORDS) * (int)sizeof(uint32_t)
        + (int)sizeof(float)                        /* s_log_prob      */
        + gh.N          * (int)sizeof(int)          /* s_parent        */
        + MAX_PATH_LEN  * (int)sizeof(int)          /* s_path_nodes    */
        + MAX_PATH_LEN  * (int)sizeof(int)          /* s_path_eids     */
        + (int)sizeof(int);                         /* s_path_len      */

    /* Guard: reduce blocks if smem exceeds device limit. */
    /*if (smem_bytes > (int)prop.sharedMemPerBlock) {
        fprintf(stderr,
            "WARNING: smem_bytes=%d exceeds sharedMemPerBlock=%d.\n"
            "  Consider reducing MAX_PATH_LEN or graph size.\n",
            smem_bytes, (int)prop.sharedMemPerBlock);
        // clamp to 1 block – still correct, just slower 
        n_blocks = 1;
    }*/
    
    /* ---- query smem/register-limited occupancy ---- */
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        factoring_kernel,
        threads_per_block,
        smem_bytes));
    
    if (blocks_per_sm < 1) {
        fprintf(stderr,
            "ERROR: kernel cannot fit on any SM with smem_bytes=%d "
            "(per-block max=%d, per-SM max=%d).\n"
            "  Reduce MAX_PATH_LEN or graph size.\n",
            smem_bytes,
            (int)prop.sharedMemPerBlock,
            (int)prop.sharedMemPerMultiprocessor);
        exit(EXIT_FAILURE);
    }
    
    int n_blocks = blocks_per_sm * num_sms;
    
    printf("Occupancy: %d blocks/SM × %d SMs = %d blocks total\n",
        blocks_per_sm, num_sms, n_blocks);

    printf("Launch config: %d blocks × %d threads, smem=%d B, SMs=%d\n",
           n_blocks, threads_per_block, smem_bytes, num_sms);

    /* ---- timing ---- */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    /* ================================================================ */
    /*  Iterative kernel launch loop                                    */
    /* ================================================================ */
    int    iteration    = 0;
    long long total_read   = 0;
    size_t max_rss_items = 0;

    for (;;) {
        int total_items = 0;
        for (int q = 0; q < NUM_QUEUES; q++)
            total_items += queues[q].read_size;

        if (total_items == 0) break;

        total_read += total_items;

        if ((size_t)total_items > max_rss_items)
            max_rss_items = (size_t)total_items;

        /* reset dequeue counters */
        for (int q = 0; q < NUM_QUEUES; q++) {
            int zero = 0;
            CUDA_CHECK(cudaMemcpy(d_next_idx[q], &zero, sizeof(int),
                                   cudaMemcpyHostToDevice));
        }

        /* build queue structs */
        ReadQueue  rq[NUM_QUEUES];
        WriteQueue wq[NUM_QUEUES];
        for (int q = 0; q < NUM_QUEUES; q++) {
            rq[q] = make_read_queue(&queues[q], d_next_idx[q]);
            wq[q] = make_write_queue(&queues[q]);
        }

        /* launch kernel */
        factoring_kernel<<<n_blocks, threads_per_block, smem_bytes>>>(
            rq[0],
            wq[0],
            gd.row_ptr, gd.col_idx, gd.edge_id,
            gd.log_p,   gd.log_q,
            gd.N, gd.E, gd.src, gd.dst,
            gd.num_mask_words,
            cfg.truncation_log_eps,
            dstats);

        CUDA_CHECK(cudaDeviceSynchronize());

        /* swap read/write for the queue */
        for (int q = 0; q < NUM_QUEUES; q++)
            queue_pair_swap(&queues[q]);

        iteration++;
    }

    /* ---- stop timer ---- */
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    /* ---- read back stats ---- */
    double R = 0.0, h_esp = 0.0;
    CUDA_CHECK(cudaMemcpy(&R, dstats.R, sizeof(double),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_esp, dstats.esp_truncated,
                           sizeof(double), cudaMemcpyDeviceToHost));

    unsigned long long h_paths = 0, h_nodes = 0;
    CUDA_CHECK(cudaMemcpy(&h_paths, dstats.paths_enumerated,
                           sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_nodes, dstats.nodes_processed,
                           sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost));

    /* ---- compute final reliability ---- */
    double R_with_esp = R + h_esp;

    size_t rss_bytes = max_rss_items * sizeof(WorkItem);

    /* ---- output ---- */
    printf("\n");
    printf("============================================\n");
    printf("  Two-Terminal Network Reliability Results\n");
    printf("============================================\n");
    printf("Graph:            N = %d, E = %d\n", gh.N, gh.E);
    printf("Terminals:        src = %d, dst = %d\n", gh.src, gh.dst);
    printf("--------------------------------------------\n");
    printf("Reliability:      R(G) = %.12f\n", R);
    printf("R + ESP:                 %.12f\n", R_with_esp);
    printf("ESP (truncated):         %.6e\n", h_esp);
    printf("Truncation bound:        %.6e\n",
           (R > 0) ? h_esp / R : h_esp);
    printf("--------------------------------------------\n");
    printf("Paths enumerated: %llu\n", h_paths);
    printf("Nodes processed:  %llu\n", h_nodes);
    printf("Kernel iterations:%d\n", iteration);
    printf("Max queue RSS:    %.3f MB  (%zu items)\n",
           rss_bytes / (1024.0 * 1024.0), max_rss_items);
    printf("Total time:       %.3f ms\n", elapsed_ms);
    printf("============================================\n");

    /* ---- cleanup ---- */
    for (int q = 0; q < NUM_QUEUES; q++) {
        queue_pair_free(&queues[q]);
        cudaFree(d_next_idx[q]);
    }
    cudaFree(dstats.R);
    cudaFree(dstats.esp_truncated);
    cudaFree(dstats.paths_enumerated);
    cudaFree(dstats.nodes_processed);
    graph_free_device(gd);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return 0;
}

