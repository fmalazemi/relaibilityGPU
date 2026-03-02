# CUDA Two-Terminal Network Reliability via Factoring

GPU-accelerated computation of two-terminal (s,t) network reliability on
undirected graphs with probabilistic edges, using the recursive factoring
(decomposition) algorithm.

## Algorithm

The factoring algorithm recursively applies:

```
R(G, M) = p_i · R(G, M | e_i = working) + q_i · R(G, M | e_i = failed)
```

**Termination conditions** (evaluated via BFS from source):

| BFS Check | Result | Action |
|---|---|---|
| Optimistic BFS (WORKING + UNKNOWN edges) | dst unreachable | Prune — contributes 0 |
| Confirmed BFS (WORKING edges only) | dst reachable | Terminal success — accumulate probability |
| Otherwise | — | Select pivot edge, branch into 2 children |

## Architecture

```
┌─────────────────────────────────────────────┐
│  Host Driver (iterative kernel launches)    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Q_HIGH   │  │ Q_MED    │  │ Q_LOW    │  │  ← Read queues
│  │ (read)   │  │ (read)   │  │ (read)   │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       └──────────────┼──────────────┘       │
│                      ▼                      │
│  ┌──────────────────────────────────────┐   │
│  │  CUDA Kernel (N blocks × 32 threads) │   │
│  │  Each warp:                          │   │
│  │    1. Dequeue WorkItem               │   │
│  │    2. Optimistic BFS (prune?)        │   │
│  │    3. Confirmed BFS (terminal?)      │   │
│  │    4. Pivot → enqueue 2 children     │   │
│  └──────────────────────────────────────┘   │
│       ┌──────────────┼──────────────┐       │
│  ┌────┴─────┐  ┌────┴─────┐  ┌────┴─────┐  │
│  │ Q_HIGH   │  │ Q_MED    │  │ Q_LOW    │  │  ← Write queues
│  │ (write)  │  │ (write)  │  │ (write)  │  │
│  └──────────┘  └──────────┘  └──────────┘  │
│                                             │
│  After kernel: swap read ↔ write, repeat    │
└─────────────────────────────────────────────┘
```

### Key Design Decisions

| Feature | Choice | Rationale |
|---|---|---|
| Graph storage | CSR in global memory with `__ldg()` | Cache-friendly, supports large graphs |
| Edge mask | 2 bits/edge in `uint32_t[]` | Compact; bitwise ops for state queries |
| Probability | Log-space (`ln`) | Prevents underflow in deep recursion |
| Work unit | 1 warp (32 threads) per WorkItem | Warp-synchronous BFS, no `__syncthreads` needed |
| BFS | Bitmask frontier in shared memory | Uses `__ballot_sync`, `__any_sync`, `atomicOr` |
| Queues | Double-buffered (read/write separation) | Eliminates producer-consumer races |
| Accumulation | Host-side log-sum-exp in `double` | Numerically stable final result |

## Build

### Prerequisites
- CUDA Toolkit ≥ 10.0
- GPU with compute capability ≥ 7.0 (Volta+)
- g++ (for CPU reference)

### Compile

```bash
# GPU version (adjust sm_XX to your GPU)
make ARCH=sm_70

# For larger graphs (up to 512 edges):
make ARCH=sm_80 MASK_WORDS=32

# CPU reference (for validation)
g++ -O2 -o cpu_ref cpu_reference.cpp
```

### GPU Architecture Flags

| GPU | Flag |
|---|---|
| V100 | `sm_70` |
| T4 | `sm_75` |
| A100 | `sm_80` |
| RTX 3090 | `sm_86` |
| RTX 4090 | `sm_89` |
| H100 | `sm_90` |

## Usage

```bash
./network_reliability <graph_file> <src> <dst> [options]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--truncation_eps <float>` | `1e-15` | Prune WorkItems with probability below this |
| `--queue_capacity <int>` | `1048576` | Max items per priority queue |
| `--thresh_high <float>` | `-4.605` | Log-prob threshold for Q_HIGH vs Q_MED |
| `--thresh_low <float>` | `-9.210` | Log-prob threshold for Q_MED vs Q_LOW |

### Examples

```bash
# Simple series graph (expected R ≈ 0.504)
./network_reliability test_simple.txt 0 3

# Bridge network (expected R ≈ 0.835)
./network_reliability test_bridge.txt 0 3

# Kite network
./network_reliability test_kite.txt 0 4

# With aggressive truncation
./network_reliability test_kite.txt 0 4 --truncation_eps 1e-8
```

## Graph File Format

```
N E
d_0  v_0a p_0a  v_0b p_0b  ...
d_1  v_1a p_1a  ...
...
d_{N-1} ...
```

- **Line 1**: `N` nodes, `E` undirected edges
- **Lines 2..N+1**: For node `i`, degree `d_i` followed by `d_i` pairs of `(neighbor, probability)`
- Each undirected edge appears in both endpoints' adjacency lists
- Simple graphs only (no parallel edges between the same node pair)

### Example (series graph)

```
4 3
1 1 0.9
2 0 0.9 2 0.8
2 1 0.8 3 0.7
1 2 0.7
```

## Output

```
============================================
  Two-Terminal Network Reliability Results
============================================
Graph:            N = 4, E = 3
Terminals:        src = 0, dst = 3
--------------------------------------------
Reliability:      R(G) = 0.504000000000
R + ESP:                 0.504000000000
ESP (truncated):         0.000000e+00
Truncation bound:        0.000000e+00
--------------------------------------------
Paths enumerated: 1
Nodes processed:  5
Kernel iterations:3
Max queue RSS:    0.001 MB  (2 items)
Total time:       1.234 ms
============================================
```

## Validation

Compare GPU results against the CPU reference:

```bash
g++ -O2 -o cpu_ref cpu_reference.cpp
./cpu_ref test_simple.txt 0 3    # R = 0.504000000000
./cpu_ref test_bridge.txt 0 3    # R = 0.835000000000
```

### Analytically Known Values

| Graph | src→dst | R(G) | Formula |
|---|---|---|---|
| `test_simple.txt` | 0→3 | 0.504 | 0.9 × 0.8 × 0.7 |
| `test_bridge.txt` | 0→3 | 0.835 | Factor on edge 1-2 |

## File Structure

```
├── common.h            Types, defines, CUDA macros
├── graph.h             Graph CSR: parse, build, transfer to GPU
├── device_code.cuh     Device functions: mask ops, BFS, pivot selection
├── main.cu             Kernel + host driver + main()
├── Makefile            Build system
├── cpu_reference.cpp   Sequential CPU reference for validation
├── test_simple.txt     Series graph (3 edges)
├── test_bridge.txt     Bridge network (5 edges)
├── test_kite.txt       Kite network (7 edges)
└── README.md           This file
```

## Tuning

- **Small graphs (E < 50)**: Default settings work well. Most work finishes in few iterations.
- **Medium graphs (50 < E < 200)**: Increase `queue_capacity` if you see overflow warnings.
- **Large graphs (E > 200)**: Use aggressive truncation (`--truncation_eps 1e-8`) and increase `MASK_WORDS` at compile time.
- **Priority thresholds**: Adjust `--thresh_high` / `--thresh_low` to control how aggressively low-probability branches are deprioritized. Lower values push more work into Q_HIGH for faster convergence.

## Limitations

- Simple graphs only (no parallel edges between the same pair of nodes)
- Maximum nodes: 1024 (limited by shared-memory BFS bitmasks)
- Maximum edges: `MAX_MASK_WORDS × 16` (default 256; recompile to increase)
- The double-buffered queue design means newly generated WorkItems are processed in the *next* kernel launch, not the current one. This is breadth-first exploration with wave-synchronous execution.
