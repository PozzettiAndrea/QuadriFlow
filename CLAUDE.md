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
- Initialize (GPU subdiv): ~4s
- Field solving (GPU PCG): ~4s
- BuildIntegerConstraints: ~4-6s (CPU)
- ComputeMaxFlow (edkarp): ~25s (GPU EK, 43K bad edges)
- subdivide_edgeDiff: ~3-5s (CPU)
- FixFlipHierarchy: ~8s (CPU)
- Extract + dynamic: ~8s (mixed)

## Flow Solver Strategies
| Strategy | Flag | Time | Bad edges | Notes |
|----------|------|------|-----------|-------|
| boykov | 0 | 111s | 50K | Boost BK, baseline |
| cuda | 1 | ~8s | 79K | GPU PR + refinement |
| edkarp | 3 | 25.7s | 43K | **Default.** GPU EK, best quality/speed |
| dinic | 4 | varies | varies | WIP, cleanup phase has issues |

## Paper Library
53 papers in `/home/shadeform/relevant_papers_cuda_remeshing/` covering:
- GPU max-flow (13 papers), IBFS/EIBFS, push-relabel, BK variants
- GPU mesh operations: RXMesh, edge collapse, subdivision, AMR
- GPU solvers: JGS2, Vivace, chaotic relaxation, graph coloring
- Quad meshing: QuadriFlow original, CrossGen, integer cross fields

## Council of Boffins
Debate format for hard problems. 4 experts: Lena (geometer), Ravi (graph theory), Doug (NVIDIA systems), Keiko (CUDA kernels). Description at `/home/shadeform/boffins.md`. 21 sessions documented in conversation history.

## Key Findings
- GPU subdivision: Luby-like independent set with length-based priority prevents race conditions
- Mesh residual graphs stay 99.98% connected — no compaction/subgraph tricks work
- Push-relabel is fast but produces poor flow quality (286K bad edges)
- Augmenting-path algorithms (EK, Dinic) produce good quality but are inherently sequential per augmentation
- IBFS/BK tree reuse is theoretically ideal but hard to implement correctly (orphan cascading)
- JF-Cut is grid-only, doesn't apply to CSR mesh graphs
