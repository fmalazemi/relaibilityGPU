#ifndef GRAPH_H
#define GRAPH_H

#include "common.h"
#include <vector>
#include <map>
#include <algorithm>
#include <utility>

/* ================================================================== */
/*  Host-side graph (CSR)                                             */
/* ================================================================== */
struct GraphHost {
    int  N, E;
    int  src, dst;
    int  num_mask_words;          /* ceil(E*2 / 32) */

    /* CSR arrays */
    std::vector<int>   row_ptr;   /* N+1 */
    std::vector<int>   col_idx;   /* 2*E */
    std::vector<int>   edge_id;   /* 2*E  – global edge index per CSR entry */

    /* Per-edge (indexed by global edge id) */
    std::vector<float> prob;      /* p_i                */
    std::vector<float> log_p;     /* ln(p_i)            */
    std::vector<float> log_q;     /* ln(1 - p_i)        */
};

/* ================================================================== */
/*  Device-side graph (raw pointers in global / constant memory)      */
/* ================================================================== */
struct GraphDevice {
    int  N, E, src, dst;
    int  num_mask_words;

    int*   row_ptr;   /* device */
    int*   col_idx;   /* device */
    int*   edge_id;   /* device */
    float* log_p;     /* device */
    float* log_q;     /* device */
};

/* ------------------------------------------------------------------ */
/*  graph_load  –  parse the adjacency-list file, build CSR           */
/* ------------------------------------------------------------------ */
inline GraphHost graph_load(const char* filename, int src, int dst)
{
    FILE* f = fopen(filename, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }

    int N, E_file;
    if (fscanf(f, "%d %d", &N, &E_file) != 2) {
        fprintf(stderr, "Bad header in %s\n", filename); exit(1);
    }

    /* ---- read adjacency lists ---- */
    struct AdjEntry { int nbr; float p; };
    std::vector<std::vector<AdjEntry>> adj(N);

    for (int i = 0; i < N; i++) {
        int deg;
        if (fscanf(f, "%d", &deg) != 1) {
            fprintf(stderr, "Bad degree for node %d\n", i); exit(1);
        }
        adj[i].resize(deg);
        for (int j = 0; j < deg; j++) {
            if (fscanf(f, "%d %f", &adj[i][j].nbr, &adj[i][j].p) != 2) {
                fprintf(stderr, "Bad edge for node %d entry %d\n", i, j);
                exit(1);
            }
        }
    }
    fclose(f);

    /* ---- assign unique edge IDs (u < v) ---- */
    std::map<std::pair<int,int>, int> edge_map;
    std::vector<float> probs;

    for (int u = 0; u < N; u++) {
        for (auto& e : adj[u]) {
            int a = std::min(u, e.nbr), b = std::max(u, e.nbr);
            auto key = std::make_pair(a, b);
            if (edge_map.find(key) == edge_map.end()) {
                int id = (int)edge_map.size();
                edge_map[key] = id;
                probs.push_back(e.p);
            }
        }
    }

    int E = (int)edge_map.size();
    if (E != E_file)
        fprintf(stderr, "Warning: header E=%d, found %d unique edges\n",
                E_file, E);
    if (E > MAX_EDGES_SUPPORTED) {
        fprintf(stderr, "E=%d exceeds MAX_EDGES_SUPPORTED=%d.  "
                "Recompile with -DMAX_MASK_WORDS=%d\n",
                E, MAX_EDGES_SUPPORTED, (E * 2 + 31) / 32);
        exit(1);
    }

    /* ---- build CSR ---- */
    GraphHost g;
    g.N   = N;
    g.E   = E;
    g.src = src;
    g.dst = dst;
    g.num_mask_words = (E * 2 + 31) / 32;

    g.row_ptr.resize(N + 1, 0);
    for (int u = 0; u < N; u++)
        g.row_ptr[u + 1] = (int)adj[u].size();
    for (int i = 0; i < N; i++)
        g.row_ptr[i + 1] += g.row_ptr[i];

    int nnz = g.row_ptr[N];            /* = 2*E */
    g.col_idx.resize(nnz);
    g.edge_id.resize(nnz);

    std::vector<int> offset(N, 0);
    for (int u = 0; u < N; u++) {
        for (auto& e : adj[u]) {
            int a = std::min(u, e.nbr), b = std::max(u, e.nbr);
            int idx = g.row_ptr[u] + offset[u]++;
            g.col_idx[idx] = e.nbr;
            g.edge_id[idx] = edge_map[{a, b}];
        }
    }

    /* ---- probabilities ---- */
    g.prob.resize(E);
    g.log_p.resize(E);
    g.log_q.resize(E);
    for (int i = 0; i < E; i++) {
        g.prob[i]  = probs[i];
        g.log_p[i] = (probs[i] > 0.0f)       ? logf(probs[i])       : -1e30f;
        g.log_q[i] = (1.0f - probs[i] > 0.0f) ? logf(1.0f - probs[i]) : -1e30f;
    }

    printf("Graph loaded: N=%d  E=%d  mask_words=%d  src=%d  dst=%d\n",
           N, E, g.num_mask_words, src, dst);
    return g;
}

/* ------------------------------------------------------------------ */
/*  graph_to_device  –  copy CSR arrays to GPU global memory          */
/* ------------------------------------------------------------------ */
inline GraphDevice graph_to_device(const GraphHost& h)
{
    GraphDevice d;
    d.N   = h.N;
    d.E   = h.E;
    d.src = h.src;
    d.dst = h.dst;
    d.num_mask_words = h.num_mask_words;

    int nnz = h.row_ptr[h.N];

    CUDA_CHECK(cudaMalloc(&d.row_ptr, (h.N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d.col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d.edge_id, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d.log_p,   h.E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.log_q,   h.E * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d.row_ptr, h.row_ptr.data(),
                           (h.N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.col_idx, h.col_idx.data(),
                           nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.edge_id, h.edge_id.data(),
                           nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.log_p, h.log_p.data(),
                           h.E * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.log_q, h.log_q.data(),
                           h.E * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

/* ------------------------------------------------------------------ */
inline void graph_free_device(GraphDevice& d)
{
    cudaFree(d.row_ptr);
    cudaFree(d.col_idx);
    cudaFree(d.edge_id);
    cudaFree(d.log_p);
    cudaFree(d.log_q);
}

#endif /* GRAPH_H */
