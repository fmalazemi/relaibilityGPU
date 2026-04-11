#ifndef DEVICE_CODE_CUH
#define DEVICE_CODE_CUH

#include "common.h"

/* ================================================================== */
/*  Mask operations  (device inline)                                  */
/* ================================================================== */

__device__ __forceinline__
int mask_get(const EdgeMask* m, int edge_id)
{
    int bit_pos = edge_id * 2;
    int word    = bit_pos >> 5;          /* / 32 */
    int offset  = bit_pos & 31;          /* % 32 */
    return (m->bits[word] >> offset) & 0x3;
}

__device__ __forceinline__
void mask_set(EdgeMask* m, int edge_id, int state)
{
    int bit_pos = edge_id * 2;
    int word    = bit_pos >> 5;
    int offset  = bit_pos & 31;
    m->bits[word] &= ~(0x3u << offset);
    m->bits[word] |=  ((uint32_t)state << offset);
}

/* Count edges still in UNKNOWN state.
   UNKNOWN = 0b11 → both bits set in each 2-bit slot.                */
__device__ __forceinline__
int mask_count_unknown(const EdgeMask* m, int num_words)
{
    int count = 0;
    for (int i = 0; i < num_words; i++) {
        uint32_t w = m->bits[i];
        /* pairs where both bits are 1 */
        uint32_t pairs = w & (w >> 1) & 0x55555555u;
        count += __popc(pairs);
    }
    return count;
}

/* ================================================================== */
/*  Node-bitmask helpers  (for BFS frontier / visited)                */
/*  Max nodes supported: 32 * 32 = 1024                              */
/* ================================================================== */
#define NODE_WORDS(N) (((N) + 31) >> 5)
#define MAX_NODE_WORDS 32   /* supports up to 1024 nodes */

__device__ __forceinline__
int  bitmask_test(const volatile uint32_t* bm, int node)
{
    return (bm[node >> 5] >> (node & 31)) & 1;
}

__device__ __forceinline__
void bitmask_set_atomic(volatile uint32_t* bm, int node)
{
    atomicOr((uint32_t*)&bm[node >> 5], 1u << (node & 31));
}

/* ================================================================== */
/*  Warp-cooperative BFS                                              */
/*                                                                    */
/*  allow_unknown = 1  →  optimistic BFS  (WORKING | UNKNOWN edges)  */
/*  allow_unknown = 0  →  confirmed  BFS  (WORKING edges only)       */
/*                                                                    */
/*  Returns 1 if dst is reachable, 0 otherwise.                       */
/*                                                                    */
/*  parent[] – optional output, size g_N ints.                        */
/*    Pass a non-NULL pointer only for the optimistic BFS call.       */
/*    On entry, caller must have initialised parent[i] = -1 for all i.*/
/*    On return, parent[v] = u means u was v's BFS predecessor.       */
/*    parent[src] remains -1 (sentinel).                              */
/*    Pass NULL for the confirmed BFS – the array is not touched.     */
/* ================================================================== */
__device__
int bfs_reachable(
    /* graph (device pointers) */
    const int*   row_ptr,
    const int*   col_idx,
    const int*   edge_id_arr,
    int          N,
    /* mask */
    const EdgeMask* mask,
    /* terminals */
    int src, int dst,
    /* mode */
    int allow_unknown,
    /* shared-memory scratch (caller must provide, size >= 3*node_words) */
    volatile uint32_t* frontier,      /* node_words uint32_t */
    volatile uint32_t* visited,       /* node_words uint32_t */
    volatile uint32_t* next_frontier, /* node_words uint32_t */
    int node_words,
    /* optional parent array for path reconstruction (size N ints)    */
    /* pass NULL when not needed (confirmed BFS)                      */
    volatile int* parent)
{
    int lane = threadIdx.x & 31;

    /* ---- initialise bitmasks ---- */
    for (int i = lane; i < node_words; i += 32) {
        frontier[i]      = 0;
        visited[i]       = 0;
        next_frontier[i] = 0;
    }
    __syncwarp();

    /* ---- initialise parent array ---- */
    if (parent != NULL) {
        for (int i = lane; i < N; i += 32)
            parent[i] = -1;
        __syncwarp();
    }

    if (lane == 0) {
        frontier[src >> 5] = 1u << (src & 31);
        visited[src >> 5]  = 1u << (src & 31);
        /* parent[src] stays -1 (sentinel) */
    }
    __syncwarp();

    /* check trivial case */
    if (src == dst) return 1;

    /* ---- BFS levels ---- */
    for (;;) {
        /* clear next_frontier */
        for (int i = lane; i < node_words; i += 32)
            next_frontier[i] = 0;
        __syncwarp();

        /* --- expand frontier ---
           Each lane handles nodes whose (node / 32) ≡ lane (mod 32),
           i.e. lane owns word(s) of the bitmask.  For N ≤ 1024
           each lane owns at most one word.                            */
        for (int wi = lane; wi < node_words; wi += 32) {
            uint32_t fw = frontier[wi];
            while (fw) {
                int bit  = __ffs(fw) - 1;   /* lowest set bit */
                fw      &= fw - 1;          /* clear it       */
                int node = wi * 32 + bit;
                if (node >= N) continue;

                /* iterate over CSR neighbours */
                int start = __ldg(&row_ptr[node]);
                int end   = __ldg(&row_ptr[node + 1]);
                for (int j = start; j < end; j++) {
                    int nbr = __ldg(&col_idx[j]);
                    int eid = __ldg(&edge_id_arr[j]);
                    int st  = mask_get(mask, eid);

                    int ok = (st == EDGE_WORKING) ||
                             (allow_unknown && st == EDGE_UNKNOWN);
                    if (!ok) continue;

                    if (!bitmask_test(visited, nbr)) {
                        bitmask_set_atomic(next_frontier, nbr);
                        bitmask_set_atomic(visited,       nbr);

                        /* Record BFS parent on the optimistic pass.
                           Multiple lanes may race on the same nbr but
                           any valid predecessor is correct for our
                           purposes (we only need one src→dst path).  */
                        if (parent != NULL)
                            parent[nbr] = node;
                    }
                }
            }
        }
        __syncwarp();

        /* did we reach dst? */
        if (bitmask_test(visited, dst)) return 1;

        /* is next_frontier empty? */
        int any = 0;
        for (int i = lane; i < node_words; i += 32)
            if (next_frontier[i]) any = 1;
        if (!__any_sync(0xFFFFFFFF, any)) return 0; /* no progress */

        /* swap frontier ← next_frontier */
        for (int i = lane; i < node_words; i += 32)
            frontier[i] = next_frontier[i];
        __syncwarp();
    }
}

/* ================================================================== */
/*  Pivot selection  (warp-parallel)                                  */
/*                                                                    */
/*  Finds the UNKNOWN edge with the highest  p_i  (= highest log_p). */
/*  Returns the global edge id, or -1 if none found.                  */
/*  Used as fallback when the path has no UNKNOWN edges.              */
/* ================================================================== */
__device__
int select_pivot(const EdgeMask* mask,
                 const float*    log_p,
                 int             E)
{
    int   lane = threadIdx.x & 31;
    int   best_edge = -1;
    float best_lp   = -1e30f;

    /* distribute edges across lanes */
    for (int e = lane; e < E; e += 32) {
        if (mask_get(mask, e) == EDGE_UNKNOWN) {
            float lp = __ldg(&log_p[e]);
            if (lp > best_lp) { best_lp = lp; best_edge = e; }
        }
    }

    /* warp reduction – find global max */
    for (int off = 16; off > 0; off >>= 1) {
        float  o_lp = __shfl_down_sync(0xFFFFFFFF, best_lp,   off);
        int    o_e  = __shfl_down_sync(0xFFFFFFFF, best_edge, off);
        if (o_lp > best_lp) { best_lp = o_lp; best_edge = o_e; }
    }

    /* broadcast result from lane 0 */
    best_edge = __shfl_sync(0xFFFFFFFF, best_edge, 0);
    return best_edge;
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

#define MAX_PATH_LEN 30   /* = MAX_NODE_WORDS * 32 */

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
    /* write queues */
    WriteQueue*  wq0,
    WriteQueue*  wq1,
    WriteQueue*  wq2,
    float        thresh_high,
    float        thresh_low,
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
     *    • Enqueue the resulting WorkItem into the correct queue.    *
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

        /* ---- select queue by log-probability ---- */
        WriteQueue* wq;
        if      (child.log_prob > thresh_high) wq = wq0;
        else if (child.log_prob > thresh_low)  wq = wq1;
        else                                   wq = wq2;

        int pos = atomicAdd(wq->count, 1);
        if (pos < wq->capacity)
            wq->buffer[pos] = child;
        
        if(hop == n_hops-1){
            //mask_set(&child.mask, eid_i, EDGE_WORKING);
            child.log_prob += __ldg(&g_log_p[eid_i]);
            child.log_prob -= __ldg(&g_log_q[eid_i]);
            
            int slot = atomicAdd(stats.terminal_count, 1);
            if (slot < stats.terminal_capacity)
                stats.terminal_log_probs[slot] = child.log_prob;
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

#endif /* DEVICE_CODE_CUH */
