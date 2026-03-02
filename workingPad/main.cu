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

/* ================================================================== */
/*  Double-buffered queue structures                                  */
/*  Read queues  – consumed by the kernel (immutable during launch).  */
/*  Write queues – append-only by the kernel (atomic counter).        */
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

/* ---- statistics (device-side) ---- */
struct DeviceStats {
    float*                terminal_log_probs; /* buffer for results */
    int*                  terminal_count;     /* atomic */
    int                   terminal_capacity;
    double*               esp_truncated;      /* atomic (CAS) */
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
/*  FACTORING  KERNEL                                                 */
/*                                                                    */
/*  Each block = 1 warp (32 threads).                                 */
/*  1. Dequeue one WorkItem from the highest-priority read queue.     */
/*  2. Optimistic BFS → prune dead ends.                              */
/*  3. Confirmed  BFS → detect terminal successes.                    */
/*  4. Otherwise select pivot, generate 2 children, enqueue.          */
/* ================================================================== */

/* shared-memory layout: mask + BFS bitmasks */
__global__ void factoring_kernel(
    /* read queues (3) */
    ReadQueue   rq0, ReadQueue  rq1, ReadQueue  rq2,
    /* write queues (3) */
    WriteQueue  wq0, WriteQueue wq1, WriteQueue wq2,
    /* graph */
    const int*   g_row_ptr,
    const int*   g_col_idx,
    const int*   g_edge_id,
    const float* g_log_p,
    const float* g_log_q,
    int          g_N, int g_E, int g_src, int g_dst,
    int          g_num_mask_words,
    /* thresholds */
    float thresh_high,
    float thresh_low,
    float truncation_log_eps,
    /* stats */
    DeviceStats  stats)
{
    int lane = threadIdx.x;   /* 0..31 */

    int node_words = (g_N + 31) >> 5;

    /* shared memory:
       [0 .. node_words-1]              frontier
       [node_words .. 2*node_words-1]   visited
       [2*node_words .. 3*node_words-1] next_frontier
       [3*node_words .. 3*node_words + MAX_MASK_WORDS - 1]  work mask copy
       [last float]                                         work log_prob  */
    extern __shared__ uint32_t smem[];
    volatile uint32_t* s_frontier      = smem;
    volatile uint32_t* s_visited       = smem + node_words;
    volatile uint32_t* s_next_frontier = smem + 2 * node_words;
    uint32_t*          s_mask_bits     = (uint32_t*)(smem + 3 * node_words);
    float*             s_log_prob_ptr  = (float*)(s_mask_bits + MAX_MASK_WORDS);

    /* ---- main work loop ---- */
    for (;;) {
        /* ---- 1. DEQUEUE ---- */
        WorkItem item;
        int got = 0;

        if (lane == 0) {
            /* try Q_HIGH first */
            int idx = atomicAdd(rq0.next_idx, 1);
            if (idx < rq0.size) { item = rq0.buffer[idx]; got = 1; }
            if (!got) {
                idx = atomicAdd(rq1.next_idx, 1);
                if (idx < rq1.size) { item = rq1.buffer[idx]; got = 1; }
            }
            if (!got) {
                idx = atomicAdd(rq2.next_idx, 1);
                if (idx < rq2.size) { item = rq2.buffer[idx]; got = 1; }
            }
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
        if (!got) return;   /* nothing left in any read queue */

        __syncwarp();

        /* read mask from shared memory into a register-local copy */
        EdgeMask  cur_mask;
        for (int w = 0; w < MAX_MASK_WORDS; w++)
            cur_mask.bits[w] = s_mask_bits[w];
        float cur_log_prob = *s_log_prob_ptr;

        /* count as processed */
        if (lane == 0)
            atomicAdd(stats.nodes_processed, 1ULL);

        /* ---- 2. TRUNCATION CHECK ---- */
        if (cur_log_prob < truncation_log_eps) {
            /* prune – accumulate into ESP */
            if (lane == 0)
                atomicAddDouble(stats.esp_truncated, exp((double)cur_log_prob));
            continue;
        }

        /* ---- 3. OPTIMISTIC BFS  (WORKING + UNKNOWN edges) ---- */
        int opt_reach = bfs_reachable(
            g_row_ptr, g_col_idx, g_edge_id, g_N,
            &cur_mask, g_src, g_dst,
            /*allow_unknown=*/ 1,
            (volatile uint32_t*)s_frontier,
            (volatile uint32_t*)s_visited,
            (volatile uint32_t*)s_next_frontier,
            node_words);

        if (!opt_reach) {
            /* dst unreachable even optimistically → dead end */
            continue;
        }

        /* ---- 4. CONFIRMED BFS  (WORKING edges only) ---- */
        int conf_reach = bfs_reachable(
            g_row_ptr, g_col_idx, g_edge_id, g_N,
            &cur_mask, g_src, g_dst,
            /*allow_unknown=*/ 0,
            (volatile uint32_t*)s_frontier,
            (volatile uint32_t*)s_visited,
            (volatile uint32_t*)s_next_frontier,
            node_words);

        if (conf_reach) {
            /* ---- TERMINAL SUCCESS ---- */
            if (lane == 0) {
                int slot = atomicAdd(stats.terminal_count, 1);
                if (slot < stats.terminal_capacity)
                    stats.terminal_log_probs[slot] = cur_log_prob;
                atomicAdd(stats.paths_enumerated, 1ULL);
            }
            continue;
        }

        /* ---- 5. PIVOT & BRANCH ---- */
        int pivot = select_pivot(&cur_mask, g_log_p, g_E);
        if (pivot < 0) continue;   /* shouldn't happen */

        /* create two children */
        if (lane == 0) {
            WorkItem child_w = { cur_mask, cur_log_prob + __ldg(&g_log_p[pivot]) };
            mask_set(&child_w.mask, pivot, EDGE_WORKING);

            WorkItem child_f = { cur_mask, cur_log_prob + __ldg(&g_log_q[pivot]) };
            mask_set(&child_f.mask, pivot, EDGE_FAILED);

            /* enqueue child_w */
            {
                float lp = child_w.log_prob;
                WriteQueue* wq;
                if      (lp > thresh_high) wq = &wq0;
                else if (lp > thresh_low)  wq = &wq1;
                else                       wq = &wq2;
                int pos = atomicAdd(wq->count, 1);
                if (pos < wq->capacity)
                    wq->buffer[pos] = child_w;
            }
            /* enqueue child_f */
            {
                float lp = child_f.log_prob;
                WriteQueue* wq;
                if      (lp > thresh_high) wq = &wq0;
                else if (lp > thresh_low)  wq = &wq1;
                else                       wq = &wq2;
                int pos = atomicAdd(wq->count, 1);
                if (pos < wq->capacity)
                    wq->buffer[pos] = child_f;
            }
        }
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
    CUDA_CHECK(cudaMalloc(&qp->buf_A, capacity * sizeof(WorkItem)));
    CUDA_CHECK(cudaMalloc(&qp->buf_B, capacity * sizeof(WorkItem)));
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
            /* always failed */
            int bp = e * 2, w = bp / 32, o = bp % 32;
            m.bits[w] &= ~(0x3u << o);   /* set 0b00 */
            pre_delete++;
        } else if (g.prob[e] >= 1.0f) {
            /* always working */
            int bp = e * 2, w = bp / 32, o = bp % 32;
            m.bits[w] &= ~(0x3u << o);
            m.bits[w] |=  (0x1u << o);   /* set 0b01 */
            pre_contract++;
        }
    }
    if (pre_contract || pre_delete)
        printf("Preprocessing: %d edges contracted (p=1), "
               "%d deleted (p=0)\n", pre_contract, pre_delete);

    return m;
}

/* ================================================================== */
/*  Host-side log-sum-exp  (double precision)                         */
/* ================================================================== */
static double log_sum_exp(const float* vals, int n)
{
    if (n == 0) return -INFINITY;

    double max_val = (double)vals[0];
    for (int i = 1; i < n; i++)
        if ((double)vals[i] > max_val) max_val = (double)vals[i];

    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += exp((double)vals[i] - max_val);

    return max_val + log(sum);
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

    /* ---- allocate queues ---- */
    HostQueuePair queues[NUM_QUEUES];
    for (int q = 0; q < NUM_QUEUES; q++)
        queue_pair_init(&queues[q], cfg.queue_capacity);

    /* seed Q_HIGH with initial work item */
    CUDA_CHECK(cudaMemcpy(queues[Q_HIGH].read_buf, &init_item,
                           sizeof(WorkItem), cudaMemcpyHostToDevice));
    queues[Q_HIGH].read_size = 1;

    /* ---- allocate per-queue dequeue counters ---- */
    int* d_next_idx[NUM_QUEUES];
    for (int q = 0; q < NUM_QUEUES; q++) {
        CUDA_CHECK(cudaMalloc(&d_next_idx[q], sizeof(int)));
    }

    /* ---- allocate stats ---- */
    int terminal_capacity = cfg.queue_capacity;   /* generous */
    DeviceStats dstats;
    CUDA_CHECK(cudaMalloc(&dstats.terminal_log_probs,
                           terminal_capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dstats.terminal_count, sizeof(int)));
    dstats.terminal_capacity = terminal_capacity;
    CUDA_CHECK(cudaMalloc(&dstats.esp_truncated, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dstats.paths_enumerated,
                           sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&dstats.nodes_processed,
                           sizeof(unsigned long long)));

    /* zero stats */
    CUDA_CHECK(cudaMemset(dstats.terminal_count, 0, sizeof(int)));
    {
        double z = 0.0;
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
    int num_sms  = prop.multiProcessorCount;
    int n_blocks = num_sms * 2;            /* 2 warps per SM */
    int threads_per_block = 32;            /* exactly 1 warp */

    int node_words = (gh.N + 31) / 32;
    /* shared memory: 3*node_words + MAX_MASK_WORDS + 1 float */
    int smem_bytes = (3 * node_words + MAX_MASK_WORDS) * sizeof(uint32_t)
                   + sizeof(float);

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
    int iteration = 0;
    long long total_read = 0;
    size_t max_rss_items = 0;

    for (;;) {
        int total_items = 0;
        for (int q = 0; q < NUM_QUEUES; q++)
            total_items += queues[q].read_size;

        if (total_items == 0) break;

        total_read += total_items;

        /* track peak queue occupancy */
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
            rq[0], rq[1], rq[2],
            wq[0], wq[1], wq[2],
            gd.row_ptr, gd.col_idx, gd.edge_id,
            gd.log_p,   gd.log_q,
            gd.N, gd.E, gd.src, gd.dst,
            gd.num_mask_words,
            cfg.thresh_high, cfg.thresh_low, cfg.truncation_log_eps,
            dstats);

        CUDA_CHECK(cudaDeviceSynchronize());

        /* swap read/write for each queue */
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
    int h_terminal_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_terminal_count, dstats.terminal_count,
                           sizeof(int), cudaMemcpyDeviceToHost));
    if (h_terminal_count > terminal_capacity)
        h_terminal_count = terminal_capacity;

    std::vector<float> h_terminal_lps(h_terminal_count);
    if (h_terminal_count > 0) {
        CUDA_CHECK(cudaMemcpy(h_terminal_lps.data(),
                               dstats.terminal_log_probs,
                               h_terminal_count * sizeof(float),
                               cudaMemcpyDeviceToHost));
    }

    double h_esp = 0.0;
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
    double log_R = log_sum_exp(h_terminal_lps.data(), h_terminal_count);
    double R     = (h_terminal_count > 0) ? exp(log_R) : 0.0;
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
    cudaFree(dstats.terminal_log_probs);
    cudaFree(dstats.terminal_count);
    cudaFree(dstats.esp_truncated);
    cudaFree(dstats.paths_enumerated);
    cudaFree(dstats.nodes_processed);
    graph_free_device(gd);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return 0;
}
