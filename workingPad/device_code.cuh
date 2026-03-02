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
/*  Uses shared-memory bitmasks (caller allocates).                   */
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
    /* shared-memory scratch (caller must provide, size ≥ 3*node_words) */
    volatile uint32_t* frontier,      /* node_words uint32_t */
    volatile uint32_t* visited,       /* node_words uint32_t */
    volatile uint32_t* next_frontier, /* node_words uint32_t */
    int node_words)
{
    int lane = threadIdx.x & 31;

    /* ---- initialise ---- */
    for (int i = lane; i < node_words; i += 32) {
        frontier[i]      = 0;
        visited[i]       = 0;
        next_frontier[i] = 0;
    }
    __syncwarp();

    if (lane == 0) {
        frontier[src >> 5]  = 1u << (src & 31);
        visited[src >> 5]   = 1u << (src & 31);
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

#endif /* DEVICE_CODE_CUH */
