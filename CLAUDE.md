# QuadriFlow-cuda Development Guide

## Build
```bash
cd build
PATH=/usr/local/cuda-13/bin:/usr/bin:/bin:$PATH make -j$(nproc)
# If CMake needed:
PATH=/usr/local/cuda-13/bin:/usr/bin:/bin:$PATH cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11
```

## Test
```bash
# Quick test (tiny mesh)
./build/quadriflow -i examples/test_tiny.obj -o /dev/null -f 50 -G

# Full test (Stanford Dragon, ~1min with edkarp)
./build/quadriflow -i examples/dragon.obj -o /tmp/out.obj -f 100000 -G

# Test from checkpoint (flow solver only, ~25s)
./build/quadriflow -run-from post-constraints -run-to post-flow -flow edkarp -save-dir /tmp/qf-debug-cuda -o /dev/null
```

## CLI Flags
- `-G` / `--cuda`: Enable GPU for all strategies (subdivide=cuda, dse=cuda, flow=edkarp)
- `-flow boykov|cuda|lemon|edkarp|dinic`: Max-flow solver selection
- `-subdiv cpu|cuda`: Mesh subdivision strategy
- `-dse cpu|cuda`: DownsampleEdgeGraph strategy
- `-ff cpu|gpu-prefilter|gpu-only`: FixFlip strategy
- `-save-all -save-dir <dir>`: Save checkpoints at all 12 pipeline stages
- `-run-from <stage> -run-to <stage>`: Resume from/stop at checkpoint
- `-list-stages`: Show all stage names

## Architecture
12-stage pipeline with checkpoint support. Key source files:
- `src/main.cpp` — CLI parsing, pipeline orchestration
- `src/flow.hpp` — All flow solver wrappers (Boykov, CudaMaxFlow, EK, Dinic)
- `src/cuda_edkarp.cu` — GPU Edmonds-Karp and GPU Dinic implementations
- `src/cuda_maxflow.cu` — GPU push-relabel (ECL-MaxFlow based)
- `src/subdivide_gpu.cu` — GPU parallel subdivision with conflict resolution
- `src/hierarchy.hpp` — Strategy flags (flow_strategy, subdiv_strategy, etc.)
- `src/optimizer.cpp:1462` — Flow solver dispatch
- `src/checkpoint.cpp/hpp` — Binary checkpoint save/load

## Current Performance (Stanford Dragon, 871K faces → 100K target)
- Initialize (GPU subdiv): ~4-6s (subdivision 0.27s GPU, hierarchy 2s CPU, copy 0.8s)
- Field solving (GPU PCG): ~4s (orient 0.7s, scale 0.1s, position 3.5s)
- BuildIntegerConstraints: ~4.7s (100% CPU — union-find + BFS, unordered_map optimized)
- ComputeMaxFlow (edkarp): ~25s (GPU EK, 43K bad edges)
- subdivide_edgeDiff: ~3.4s (100% CPU — priority-queue splitting with orientation recomputation)
- FixFlipHierarchy: ~8s+ (100% CPU — recursive hierarchy, fundamentally sequential)
- Extract + dynamic: ~8s (mixed — pre-dynamic CSR 2.3s with unordered_map)

## Nsight Systems Profile Results (saved in /home/shadeform/qf_nsight/)
- **Initialize**: 258ms GPU kernels, 5.6s CPU (96% CPU-bound: graph coloring + downsample loops)
- **BuildIntegerConstraints**: 0ms GPU, 4.8s CPU (100% CPU: union-find, BFS, orientation compat)
- **subdivide_edgeDiff**: 0ms GPU, 3.1s CPU (100% CPU: priority queue + compat_orientation calls)
- **ComputeMaxFlow**: ~24s GPU, ~1s CPU (96% GPU: BFS kernels, memory-latency bound)

## Flow Solver Strategies
| Strategy | Flag | Time | Bad edges | Notes |
|----------|------|------|-----------|-------|
| boykov | 0 | 111s | 50K | Boost BK, baseline |
| cuda | 1 | ~8s | 79K | GPU PR + refinement |
| edkarp | 3 | 28s | 43K | GPU EK |
| dinic | 4 | 19s | 44K | GPU Dinic + EK fallback |
| ek-persistent | 5 | 24s | 44K | GPU EK with cooperative groups |
| dinic-persistent | 6 | **13s** | 44K | **Default.** Dinic + persistent EK fallback |

## Paper Library
53 papers in `/home/shadeform/relevant_papers_cuda_remeshing/` covering:
- GPU max-flow (13 papers), IBFS/EIBFS, push-relabel, BK variants
- GPU mesh operations: RXMesh, edge collapse, subdivision, AMR
- GPU solvers: JGS2, Vivace, chaotic relaxation, graph coloring
- Quad meshing: QuadriFlow original, CrossGen, integer cross fields

## Council of Boffins
Debate format for hard problems. 4 experts: Lena (geometer), Ravi (graph theory), Doug (NVIDIA systems), Keiko (CUDA kernels). Description at `/home/shadeform/boffins.md`. 21 sessions documented in conversation history.

## Current Best Timings (full pipeline, Release mode)
| Stage | Time | Device |
|---|---|---|
| Initialize (GPU subdiv) | ~4-5s | CPU+GPU |
| Field solving | ~4-5s | GPU |
| BuildIntegerConstraints | ~4-6s | CPU |
| ComputeMaxFlow (dinic-persistent) | **~14s** | GPU |
| subdivide_edgeDiff (1st) | ~2.5-5s | CPU |
| FixFlipHierarchy (depth=3) | ~3.8-9s | CPU |
| subdivide_edgeDiff (2nd) | SKIPPED | CPU |
| optimize_positions_sharp+fixed | ~2.5-5.8s | CPU+GPU |
| AdvancedExtractQuad | ~3.8s | CPU |
| pre-dynamic + dynamic | ~7.5-12s | CPU+GPU |
| **TOTAL** | **~41s** | |
| **vs edkarp** | **~61s** | **1.5x faster** |
| **vs Boykov baseline** | **~160s** | **3.9x faster** |

## Key Findings
- GPU subdivision: Luby-like independent set with length-based priority prevents race conditions
- Mesh residual graphs stay 99.98% connected — no compaction/subgraph tricks work
- Push-relabel is fast but produces poor flow quality (286K bad edges)
- Augmenting-path algorithms (EK, Dinic) produce good quality but are inherently sequential per augmentation
- IBFS/BK tree reuse is theoretically ideal but hard to implement correctly (orphan cascading)
- JF-Cut is grid-only, doesn't apply to CSR mesh graphs
- Direction-optimizing BFS: doesn't help on mesh graphs (uniform degree, high diameter)
- GPU DSE hangs on multi-level builds — forced to CPU DSE (works fine)
- unordered_map + reserve() saved ~4s across BuildIntegerConstraints + pre-dynamic
- Penner optimization: NaN divergence / slow convergence on real meshes (not a quick replacement)

## Approaches Tried (with results)
| Approach | Result | Status |
|---|---|---|
| GPU subdivision (Luby IS) | 10.8x speedup | **Working** |
| GPU Edmonds-Karp flow | 4.4x vs Boykov | **Working** |
| GPU Dinic + persistent EK | **8.5x vs Boykov** | **Working (default)** |
| GPU Dinic + EK hybrid | 5.8x vs Boykov | Working |
| Persistent kernel EK (coop groups) | 4.6x vs Boykov | Working |
| GPU push-relabel + refinement | Fast but poor quality | Available as strategy |
| unordered_map optimization | -4s across stages | **Working** |
| FixFlip depth cap | -2.2s | **Working (depth=3)** |
| o2e.reserve() | -2.5s pre-dynamic | **Working** |
| Raw math compat_orientation | -0.2s | **Working** (marginal) |
| Direction-optimizing BFS | Slower on mesh graphs | Failed |
| Compact subgraph EK | 99.98% still active | Failed |
| IBFS/BK tree reuse | Cycle bugs (3 attempts) | Failed |
| JF-Cut push-relabel | Grid-only algorithm | Failed |
| Selective BFS reset | D2D copy overhead | Failed |
| Penner optimization | NaN/slow on real meshes | Not viable yet |

## WIP: GPU Dynamic Optimization (src/dynamic_gpu.cu)

**Status:** Code written, not yet building. Needs `cmake ..` re-run to pick up new .cu file.

Two GPU kernels for the pre-dynamic + dynamic optimization stage (currently 9-10s):

1. **k_find_nearest** — Parallel manifold walk. Each thread handles one quad vertex,
   walking the mesh adjacency graph to find the nearest base vertex. Rotates diffs
   vectors along the walk path. 187K threads, replaces CPU loop (1.4s → ~5ms).

2. **k_fill_csr** — Parallel sparse matrix assembly. Each thread handles one edge,
   computing 16 dot products and atomically adding to CSR values + RHS.
   ~187K threads, replaces CPU loop (0.6s → ~1ms).

**Files:**
- `src/dynamic_gpu.cu` — CUDA kernels + host wrappers
- `src/optimizer.cpp` — Modified to call GPU versions when WITH_CUDA
- `src/optimizer.hpp` — Added extern "C" declarations

**To finish:**
```bash
cd build
PATH=/usr/local/cuda-13/bin:/usr/bin:/bin:$PATH cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11
make -j$(nproc)
# Test:
./quadriflow -i ../examples/dragon.obj -o /tmp/out.obj -f 100000 -G
```

**Expected savings:** FindNearest 1.4s → ~5ms, fillCSR 0.6s → ~1ms = ~2s saved per run.

**Council of Boffins Session 30 notes:** Full debate in plan file. Additional targets after GPU kernels:
- Pre-dynamic diffs (1.71s): replace unordered_map with GPU hash table or sorted lookup
- PCG solve (3.66s): replace IC0 with AmgX multigrid (fewer iterations)
- Fully GPU-resident loop: eliminate H2D/D2H transfers (~2.7s overhead)
