/* ================================================================== *
 *  cpu_reference.cpp  –  Sequential CPU factoring for validation     *
 *                                                                    *
 *  Build:  g++ -O2 -o cpu_ref cpu_reference.cpp                     *
 *  Run:    ./cpu_ref <graph_file> <src> <dst>                       *
 * ================================================================== */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <map>
#include <algorithm>
#include <chrono>

/* -------- edge states -------- */
#define EDGE_FAILED   0
#define EDGE_WORKING  1
#define EDGE_UNKNOWN  3

#define MAX_EDGES 256

struct Graph {
    int N, E;
    std::vector<int>   row_ptr;
    std::vector<int>   col_idx;
    std::vector<int>   edge_id;
    std::vector<float> prob;
};

static Graph load_graph(const char* fn)
{
    FILE* f = fopen(fn, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fn); exit(1); }

    int N, E;
    fscanf(f, "%d %d", &N, &E);

    struct AE { int nbr; float p; };
    std::vector<std::vector<AE>> adj(N);
    for (int i = 0; i < N; i++) {
        int d; fscanf(f, "%d", &d);
        adj[i].resize(d);
        for (int j = 0; j < d; j++)
            fscanf(f, "%d %f", &adj[i][j].nbr, &adj[i][j].p);
    }
    fclose(f);

    std::map<std::pair<int,int>, int> em;
    std::vector<float> probs;
    for (int u = 0; u < N; u++)
        for (auto& e : adj[u]) {
            auto k = std::make_pair(std::min(u, e.nbr), std::max(u, e.nbr));
            if (em.find(k) == em.end()) {
                em[k] = (int)em.size();
                probs.push_back(e.p);
            }
        }

    Graph g;
    g.N = N; g.E = (int)em.size(); g.prob = probs;
    g.row_ptr.resize(N + 1, 0);
    for (int u = 0; u < N; u++) g.row_ptr[u+1] = (int)adj[u].size();
    for (int i = 0; i < N; i++) g.row_ptr[i+1] += g.row_ptr[i];
    g.col_idx.resize(g.row_ptr[N]);
    g.edge_id.resize(g.row_ptr[N]);
    std::vector<int> off(N, 0);
    for (int u = 0; u < N; u++)
        for (auto& e : adj[u]) {
            int idx = g.row_ptr[u] + off[u]++;
            g.col_idx[idx] = e.nbr;
            g.edge_id[idx] = em[{std::min(u, e.nbr), std::max(u, e.nbr)}];
        }
    printf("Graph: N=%d E=%d\n", g.N, g.E);
    return g;
}

/* -------- mask ops -------- */
struct Mask {
    uint8_t state[MAX_EDGES];  /* simple byte array for clarity */
};

/* BFS reachability check */
static bool bfs_reach(const Graph& g, const Mask& m, int src, int dst,
                      bool allow_unknown)
{
    std::vector<bool> vis(g.N, false);
    std::vector<int> q;
    vis[src] = true;
    q.push_back(src);
    for (int qi = 0; qi < (int)q.size(); qi++) {
        int u = q[qi];
        if (u == dst) return true;
        for (int j = g.row_ptr[u]; j < g.row_ptr[u+1]; j++) {
            int v = g.col_idx[j], eid = g.edge_id[j];
            int st = m.state[eid];
            bool ok = (st == EDGE_WORKING) ||
                      (allow_unknown && st == EDGE_UNKNOWN);
            if (ok && !vis[v]) { vis[v] = true; q.push_back(v); }
        }
    }
    return false;
}

/* select pivot: first unknown edge with highest prob */
static int pick_pivot(const Graph& g, const Mask& m)
{
    int best = -1;
    float bestp = -1.0f;
    for (int e = 0; e < g.E; e++)
        if (m.state[e] == EDGE_UNKNOWN && g.prob[e] > bestp) {
            bestp = g.prob[e]; best = e;
        }
    return best;
}

/* -------- recursive factoring -------- */
static unsigned long long g_nodes = 0, g_paths = 0;

static double factor(const Graph& g, const Mask& m, int src, int dst)
{
    g_nodes++;

    /* optimistic BFS */
    if (!bfs_reach(g, m, src, dst, true))
        return 0.0;   /* dead end */

    /* confirmed BFS */
    if (bfs_reach(g, m, src, dst, false)) {
        g_paths++;
        return 1.0;   /* terminal success */
    }

    /* pivot */
    int pivot = pick_pivot(g, m);
    if (pivot < 0) return 0.0;

    double p = g.prob[pivot], q = 1.0 - p;

    Mask mw = m;  mw.state[pivot] = EDGE_WORKING;
    Mask mf = m;  mf.state[pivot] = EDGE_FAILED;

    return p * factor(g, mw, src, dst) + q * factor(g, mf, src, dst);
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <graph> <src> <dst>\n", argv[0]);
        return 1;
    }
    Graph g = load_graph(argv[1]);
    int src = atoi(argv[2]), dst = atoi(argv[3]);

    /* initial mask: all unknown, except p=0 → failed, p=1 → working */
    Mask m;
    memset(&m, 0, sizeof(m));
    for (int e = 0; e < g.E; e++) {
        if (g.prob[e] <= 0.0f)      m.state[e] = EDGE_FAILED;
        else if (g.prob[e] >= 1.0f) m.state[e] = EDGE_WORKING;
        else                        m.state[e] = EDGE_UNKNOWN;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double R = factor(g, m, src, dst);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("============================================\n");
    printf("  CPU Reference – Network Reliability\n");
    printf("============================================\n");
    printf("src=%d  dst=%d\n", src, dst);
    printf("Reliability:      R = %.12f\n", R);
    printf("Paths enumerated: %llu\n", g_paths);
    printf("Nodes processed:  %llu\n", g_nodes);
    printf("Time:             %.3f ms\n", ms);
    printf("============================================\n");
    return 0;
}
