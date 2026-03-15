// ============================================================
// GPU-resident max-flow solvers: Edmonds-Karp and Dinic's
//
// Both use CSR graph representation on GPU.
// Edmonds-Karp: 1 BFS per augmentation, GPU-resident.
// Dinic's: fwd BFS + bwd BFS pruning + multi-augment per phase.
// ============================================================

#include "cuda_maxflow.hpp"
#include <cstdio>
#include <climits>
#include <vector>
#include <queue>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace qflow {

// ---- Fused BFS kernel for Edmonds-Karp (parent-tracking) ----
__global__ void k_ek_bfs_expand(
    const int* nindex, const int* nlist, const int* cap, const int* flow,
    const int* rnindex, const int* rnlist, const int* retoe,
    const int* frontier, int frontier_size,
    int* parent, int* parent_edge, int* parent_dir,
    int* next_frontier, int* next_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;
    int u = frontier[tid];

    for (int e = nindex[u]; e < nindex[u + 1]; e++) {
        int v = nlist[e];
        if (cap[e] - flow[e] <= 0) continue;
        if (atomicCAS(&parent[v], -1, u) == -1) {
            parent_edge[v] = e;
            parent_dir[v] = 1;
            next_frontier[atomicAdd(next_count, 1)] = v;
        }
    }
    for (int re = rnindex[u]; re < rnindex[u + 1]; re++) {
        int v = rnlist[re];
        int e_fwd = retoe[re];
        if (flow[e_fwd] <= 0) continue;
        if (atomicCAS(&parent[v], -1, u) == -1) {
            parent_edge[v] = e_fwd;
            parent_dir[v] = -1;
            next_frontier[atomicAdd(next_count, 1)] = v;
        }
    }
}

// ---- Bottom-up BFS: each UNVISITED node checks if any neighbor is visited ----
// Much faster than top-down when frontier is large (>V/20) because:
// 1. Most nodes are already visited → exit immediately (cheap)
// 2. Unvisited nodes scan their OWN adjacency (sequential CSR access, good coalescing)
// 3. No random parent[] writes from frontier expansion
// Reference: Merrill et al. "Scalable GPU Graph Traversal" (NVIDIA 2011)
__global__ void k_ek_bfs_bottom_up(
    const int* nindex, const int* nlist, const int* cap, const int* flow,
    const int* rnindex, const int* rnlist, const int* retoe,
    int* parent, int* parent_edge, int* parent_dir,
    int num_nodes, int source, int sink,
    int* next_count
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    if (parent[v] != -1) return;  // already visited, skip

    // Check reverse edges: u→v with residual (u is visited, v is not)
    for (int re = rnindex[v]; re < rnindex[v + 1]; re++) {
        int u = rnlist[re];
        if (parent[u] == -1) continue;  // u not visited yet
        int ef = retoe[re];
        if (cap[ef] - flow[ef] <= 0) continue;
        // Found a visited neighbor with residual — adopt v
        parent[v] = u;
        parent_edge[v] = ef;
        parent_dir[v] = 1;
        atomicAdd(next_count, 1);
        return;
    }

    // Check forward edges: v→w with flow>0 (backward residual w→v)
    // If w is visited and flow[v→w]>0, then v can reach w via cancellation
    for (int e = nindex[v]; e < nindex[v + 1]; e++) {
        int w = nlist[e];
        if (parent[w] == -1) continue;  // w not visited
        if (flow[e] <= 0) continue;
        parent[v] = w;
        parent_edge[v] = e;
        parent_dir[v] = -1;
        atomicAdd(next_count, 1);
        return;
    }
}

// ---- Path trace + flow push (single thread, Edmonds-Karp) ----
__global__ void k_ek_trace_push(
    const int* parent, const int* parent_edge, const int* parent_dir,
    const int* cap, int* flow, int source, int sink, int* out_bottleneck
) {
    int bn = 0x7FFFFFFF;
    for (int v = sink; v != source; v = parent[v]) {
        int e = parent_edge[v];
        int r = (parent_dir[v] == 1) ? (cap[e] - flow[e]) : flow[e];
        if (r < bn) bn = r;
    }
    for (int v = sink; v != source; v = parent[v]) {
        int e = parent_edge[v];
        if (parent_dir[v] == 1) flow[e] += bn; else flow[e] -= bn;
    }
    *out_bottleneck = bn;
}

// ---- Level BFS kernel for Dinic's (level-tracking, no parent) ----
__global__ void k_dinic_bfs_expand(
    const int* nindex, const int* nlist, const int* cap, const int* flow,
    const int* rnindex, const int* rnlist, const int* retoe,
    const int* frontier, int frontier_size,
    int current_level,
    int* level,   // level[v]=-1 means unvisited; set to current_level+1
    int* next_frontier, int* next_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;
    int u = frontier[tid];

    int next_lev = current_level + 1;

    // Forward: cap - flow > 0
    for (int e = nindex[u]; e < nindex[u + 1]; e++) {
        int v = nlist[e];
        if (cap[e] - flow[e] <= 0) continue;
        int old = atomicCAS(&level[v], -1, next_lev);
        if (old == -1) next_frontier[atomicAdd(next_count, 1)] = v;
    }
    // Backward: flow > 0
    for (int re = rnindex[u]; re < rnindex[u + 1]; re++) {
        int v = rnlist[re];
        if (flow[retoe[re]] <= 0) continue;
        int old = atomicCAS(&level[v], -1, next_lev);
        if (old == -1) next_frontier[atomicAdd(next_count, 1)] = v;
    }
}

// ---- Backward BFS from sink through level graph (forward residual) ----
__global__ void k_dinic_bfs_backward(
    const int* nindex, const int* nlist,
    const int* rnindex, const int* rnlist, const int* retoe,
    const int* cap, const int* flow, const int* level,
    const int* frontier, int frontier_size,
    int* reach, int* next_frontier, int* next_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;
    int v = frontier[tid];

    // Forward residual: u->v exists means u can reach v (and thus sink).
    // Check all u that have forward edge to v with level[u]=level[v]-1 and residual>0.
    for (int re = rnindex[v]; re < rnindex[v + 1]; re++) {
        int u = rnlist[re];
        int e_fwd = retoe[re];
        if (level[u] < 0 || level[u] != level[v] - 1) continue;
        if (cap[e_fwd] - flow[e_fwd] <= 0) continue;
        if (atomicCAS(&reach[u], 0, 1) == 0)
            next_frontier[atomicAdd(next_count, 1)] = u;
    }
    // Backward residual: flow on v->w > 0 means w can reach v via cancel.
    // Check all w where v->w has flow>0 and level[w]=level[v]-1.
    for (int e = nindex[v]; e < nindex[v + 1]; e++) {
        int w = nlist[e];
        if (level[w] < 0 || level[w] != level[v] - 1) continue;
        if (flow[e] <= 0) continue;
        if (atomicCAS(&reach[w], 0, 1) == 0)
            next_frontier[atomicAdd(next_count, 1)] = w;
    }
}

// ---- Backward BFS on full residual graph (no level check) ----
// Used to find all nodes that can reach sink in the residual graph.
__global__ void k_bfs_backward_residual(
    const int* nindex, const int* nlist,
    const int* rnindex, const int* rnlist, const int* retoe,
    const int* cap, const int* flow,
    const int* frontier, int frontier_size,
    int* reach, int* next_frontier, int* next_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;
    int v = frontier[tid];

    // Forward residual: edge u->v with cap-flow>0 means u can reach v→sink
    for (int re = rnindex[v]; re < rnindex[v + 1]; re++) {
        int u = rnlist[re];
        int e_fwd = retoe[re];
        if (cap[e_fwd] - flow[e_fwd] <= 0) continue;
        if (atomicCAS(&reach[u], 0, 1) == 0)
            next_frontier[atomicAdd(next_count, 1)] = u;
    }
    // Backward residual: edge v->w with flow>0 means w can reach v→sink
    for (int e = nindex[v]; e < nindex[v + 1]; e++) {
        int w = nlist[e];
        if (flow[e] <= 0) continue;
        if (atomicCAS(&reach[w], 0, 1) == 0)
            next_frontier[atomicAdd(next_count, 1)] = w;
    }
}

// ---- Multi-augment on GPU (single thread, DFS with backtracking) ----
// Uses current-arc optimization: iter[v] tracks which edge to try next.
__global__ void k_dinic_multi_augment(
    const int* nindex, const int* nlist,
    const int* rnindex, const int* rnlist, const int* retoe,
    const int* cap, int* flow,
    const int* level, int* reach,
    int source, int sink,
    int* out_total_flow, int* out_num_paths
) {
    // Current-arc pointers: for each node, which edge to try next
    // Encode: values < rnindex[v+1]-rnindex[v] are reverse-CSR indices (forward edges TO v)
    //         values >= that are forward-CSR indices (backward residual edges FROM v)
    // But simpler: just store separate forward and backward iterators.
    // Since we're single-threaded with ~4M nodes, we can't afford per-node arrays.
    // Instead: use DFS stack with edge iterators stored per stack frame.

    struct Frame {
        int node;
        int re_iter;  // next reverse-CSR index to try (for forward edges u->node)
        int fe_iter;  // next forward-CSR index to try (for backward edges node->w)
        int edge_used;
        int dir_used;
    };

    // Path stack — max depth ~200
    Frame stack[256];
    int sp = 0;

    int total = 0, paths = 0;

    // Push source
    stack[0] = {source, rnindex[source], nindex[source], -1, 0};
    sp = 1;

    while (sp > 0) {
        Frame& fr = stack[sp - 1];
        int v = fr.node;

        if (v == sink) {
            // Found a path! Find bottleneck and push flow.
            int bn = 0x7FFFFFFF;
            for (int i = 1; i < sp; i++) {
                int e = stack[i].edge_used;
                int d = stack[i].dir_used;
                int r = (d == 1) ? (cap[e] - flow[e]) : flow[e];
                if (r < bn) bn = r;
            }
            if (bn > 0) {
                for (int i = 1; i < sp; i++) {
                    int e = stack[i].edge_used;
                    if (stack[i].dir_used == 1) flow[e] += bn; else flow[e] -= bn;
                }
                total += bn;
                paths++;
            }
            // Pop sink and continue from parent (try next edge)
            sp--;
            continue;
        }

        bool advanced = false;

        // Try forward edges: v→w with residual>0 and level[w]=level[v]+1
        for (int& e = fr.fe_iter; e < nindex[v + 1]; e++) {
            int w = nlist[e];
            if (!reach[w]) continue;
            if (level[w] != level[v] + 1) continue;
            if (cap[e] - flow[e] <= 0) continue;
            if (sp < 255) {
                stack[sp] = {w, rnindex[w], nindex[w], e, 1};
                sp++;
                e++;
                advanced = true;
                break;
            }
        }

        if (!advanced) {
            // Backward residual: edge w→v with flow>0, level[w]=level[v]+1
            for (int& re = fr.re_iter; re < rnindex[v + 1]; re++) {
                int w = rnlist[re];
                int e_fwd = retoe[re];
                if (!reach[w]) continue;
                if (level[w] != level[v] + 1) continue;
                if (flow[e_fwd] <= 0) continue;
                if (sp < 255) {
                    stack[sp] = {w, rnindex[w], nindex[w], e_fwd, -1};
                    sp++;
                    re++;
                    advanced = true;
                    break;
                }
            }
        }

        if (!advanced) {
            // Dead end — mark unreachable and backtrack
            reach[v] = 0;
            sp--;
        }
    }

    *out_total_flow = total;
    *out_num_paths = paths;
}


// ============================================================
// Edmonds-Karp (GPU-resident): 1 BFS per augmentation
// ============================================================
CudaMaxFlowResult cuda_edmonds_karp_solve(
    int num_nodes, int source, int sink,
    const int* h_nindex, const int* h_nlist, const int* h_cap,
    int num_edges,
    const int* h_rnindex, const int* h_rnlist, const int* h_retoe
) {
    CudaMaxFlowResult result;
    result.edge_flows.resize(num_edges);

    int *d_nindex, *d_nlist, *d_cap, *d_flow;
    int *d_rnindex, *d_rnlist, *d_retoe;
    int *d_parent, *d_parent_edge, *d_parent_dir;
    int *d_frontier, *d_next_frontier, *d_next_count, *d_bottleneck;

    cudaMalloc(&d_nindex, (num_nodes+1)*sizeof(int));
    cudaMalloc(&d_nlist, num_edges*sizeof(int));
    cudaMalloc(&d_cap, num_edges*sizeof(int));
    cudaMalloc(&d_flow, num_edges*sizeof(int));
    cudaMalloc(&d_rnindex, (num_nodes+1)*sizeof(int));
    cudaMalloc(&d_rnlist, num_edges*sizeof(int));
    cudaMalloc(&d_retoe, num_edges*sizeof(int));
    cudaMalloc(&d_parent, num_nodes*sizeof(int));
    cudaMalloc(&d_parent_edge, num_nodes*sizeof(int));
    cudaMalloc(&d_parent_dir, num_nodes*sizeof(int));
    cudaMalloc(&d_frontier, num_nodes*sizeof(int));
    cudaMalloc(&d_next_frontier, num_nodes*sizeof(int));
    cudaMalloc(&d_next_count, sizeof(int));
    cudaMalloc(&d_bottleneck, sizeof(int));

    cudaMemcpy(d_nindex, h_nindex, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nlist, h_nlist, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cap, h_cap, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rnindex, h_rnindex, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rnlist, h_rnlist, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_retoe, h_retoe, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_flow, 0, num_edges*sizeof(int));

    const int B = 256;
    int total_flow = 0, augmentations = 0;
    int src_val = source;

    while (true) {
        cudaMemset(d_parent, 0xFF, num_nodes*sizeof(int));
        cudaMemcpy(d_parent+source, &src_val, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier, &src_val, sizeof(int), cudaMemcpyHostToDevice);
        int fsize = 1;
        bool found = false;

        while (fsize > 0) {
            cudaMemset(d_next_count, 0, sizeof(int));
            k_ek_bfs_expand<<<(fsize+B-1)/B, B>>>(
                d_nindex, d_nlist, d_cap, d_flow,
                d_rnindex, d_rnlist, d_retoe,
                d_frontier, fsize,
                d_parent, d_parent_edge, d_parent_dir,
                d_next_frontier, d_next_count);
            int sp;
            cudaMemcpy(&sp, d_parent+sink, sizeof(int), cudaMemcpyDeviceToHost);
            if (sp != -1) { found = true; break; }
            cudaMemcpy(&fsize, d_next_count, sizeof(int), cudaMemcpyDeviceToHost);
            std::swap(d_frontier, d_next_frontier);
        }
        if (!found) break;

        k_ek_trace_push<<<1,1>>>(d_parent, d_parent_edge, d_parent_dir,
                                  d_cap, d_flow, source, sink, d_bottleneck);
        int bn;
        cudaMemcpy(&bn, d_bottleneck, sizeof(int), cudaMemcpyDeviceToHost);
        total_flow += bn;
        augmentations++;
    }

    printf("[TIMING]       GPU Edmonds-Karp: %d augmentations, flow=%d\n", augmentations, total_flow);
    cudaMemcpy(result.edge_flows.data(), d_flow, num_edges*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_nindex); cudaFree(d_nlist); cudaFree(d_cap); cudaFree(d_flow);
    cudaFree(d_rnindex); cudaFree(d_rnlist); cudaFree(d_retoe);
    cudaFree(d_parent); cudaFree(d_parent_edge); cudaFree(d_parent_dir);
    cudaFree(d_frontier); cudaFree(d_next_frontier); cudaFree(d_next_count);
    cudaFree(d_bottleneck);

    result.max_flow = total_flow;
    return result;
}


// ============================================================
// Persistent BFS kernel for Edmonds-Karp
//
// Entire BFS (all levels) runs in ONE kernel launch.
// Uses cooperative_groups grid.sync() for level synchronization.
// Each thread handles multiple nodes via strided iteration.
// Returns: parent[sink] != -1 if sink was reached.
//
// Also does trace-and-push in the same kernel (thread 0 only)
// to avoid a separate kernel launch + host sync per augmentation.
// ============================================================
__global__ void k_ek_persistent_bfs_augment(
    const int* nindex, const int* nlist, const int* cap, int* flow,
    const int* rnindex, const int* rnlist, const int* retoe,
    int num_nodes, int source, int sink,
    int* parent, int* parent_edge, int* parent_dir,
    int* frontier, int* next_frontier, int* frontier_size, int* next_count,
    int* out_bottleneck,  // output: bottleneck capacity (0 if no path)
    int* out_done         // output: set to 1 when no augmenting path found
) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // === Outer loop: each iteration is one complete BFS + augment ===
    while (true) {
        // -- Reset parent array --
        for (int i = tid; i < num_nodes; i += stride)
            parent[i] = -1;
        grid.sync();

        // -- Initialize frontier with source --
        if (tid == 0) {
            parent[source] = source;
            frontier[0] = source;
            *frontier_size = 1;
            *next_count = 0;
        }
        grid.sync();

        // -- BFS levels (single grid.sync per level) --
        bool found = false;
        while (true) {
            int fsize = *frontier_size;
            if (fsize == 0) break;

            // Expand frontier: each thread processes multiple frontier nodes
            for (int idx = tid; idx < fsize; idx += stride) {
                int u = frontier[idx];

                // Forward edges: u -> v with residual
                for (int e = nindex[u]; e < nindex[u + 1]; e++) {
                    int v = nlist[e];
                    if (cap[e] - flow[e] <= 0) continue;
                    if (atomicCAS(&parent[v], -1, u) == -1) {
                        parent_edge[v] = e;
                        parent_dir[v] = 1;
                        int pos = atomicAdd(next_count, 1);
                        next_frontier[pos] = v;
                    }
                }
                // Backward edges: flow on e_fwd can be cancelled
                for (int re = rnindex[u]; re < rnindex[u + 1]; re++) {
                    int v = rnlist[re];
                    int e_fwd = retoe[re];
                    if (flow[e_fwd] <= 0) continue;
                    if (atomicCAS(&parent[v], -1, u) == -1) {
                        parent_edge[v] = e_fwd;
                        parent_dir[v] = -1;
                        int pos = atomicAdd(next_count, 1);
                        next_frontier[pos] = v;
                    }
                }
            }

            // Thread 0 prepares next level BEFORE sync
            if (tid == 0) {
                *frontier_size = *next_count;
                *next_count = 0;
            }

            grid.sync();  // Single sync: expansion complete + sizes visible

            // Check if sink was reached
            if (parent[sink] != -1) { found = true; break; }

            // Pointer swap (all threads do this identically)
            {
                int* tmp = frontier;
                frontier = next_frontier;
                next_frontier = tmp;
            }
        }

        if (!found) {
            // No augmenting path — we're done
            if (tid == 0) {
                *out_done = 1;
            }
            return;
        }

        // -- Trace path and push flow (single thread) --
        if (tid == 0) {
            // Find bottleneck
            int bn = 0x7FFFFFFF;
            for (int v = sink; v != source; v = parent[v]) {
                int e = parent_edge[v];
                int r = (parent_dir[v] == 1) ? (cap[e] - flow[e]) : flow[e];
                if (r < bn) bn = r;
            }
            // Push flow
            for (int v = sink; v != source; v = parent[v]) {
                int e = parent_edge[v];
                if (parent_dir[v] == 1) flow[e] += bn; else flow[e] -= bn;
            }
            *out_bottleneck += bn;
            *out_done = 0;
        }

        grid.sync();
        // Loop back for next augmentation
    }
}


// ============================================================
// Edmonds-Karp with persistent BFS kernel (cooperative groups)
//
// Eliminates per-BFS-level kernel launch overhead by fusing
// the entire BFS into one cooperative kernel launch.
// The whole solve (all augmentations + BFS) is ONE kernel launch.
// ============================================================
CudaMaxFlowResult cuda_ek_persistent_solve(
    int num_nodes, int source, int sink,
    const int* h_nindex, const int* h_nlist, const int* h_cap,
    int num_edges,
    const int* h_rnindex, const int* h_rnlist, const int* h_retoe
) {
    CudaMaxFlowResult result;
    result.edge_flows.resize(num_edges);

    auto t_start = std::chrono::high_resolution_clock::now();

    // Allocate GPU memory
    int *d_nindex, *d_nlist, *d_cap, *d_flow;
    int *d_rnindex, *d_rnlist, *d_retoe;
    int *d_parent, *d_parent_edge, *d_parent_dir;
    int *d_frontier, *d_next_frontier, *d_frontier_size, *d_next_count;
    int *d_bottleneck, *d_done;

    cudaMalloc(&d_nindex, (num_nodes+1)*sizeof(int));
    cudaMalloc(&d_nlist, num_edges*sizeof(int));
    cudaMalloc(&d_cap, num_edges*sizeof(int));
    cudaMalloc(&d_flow, num_edges*sizeof(int));
    cudaMalloc(&d_rnindex, (num_nodes+1)*sizeof(int));
    cudaMalloc(&d_rnlist, num_edges*sizeof(int));
    cudaMalloc(&d_retoe, num_edges*sizeof(int));
    cudaMalloc(&d_parent, num_nodes*sizeof(int));
    cudaMalloc(&d_parent_edge, num_nodes*sizeof(int));
    cudaMalloc(&d_parent_dir, num_nodes*sizeof(int));
    cudaMalloc(&d_frontier, num_nodes*sizeof(int));
    cudaMalloc(&d_next_frontier, num_nodes*sizeof(int));
    cudaMalloc(&d_frontier_size, sizeof(int));
    cudaMalloc(&d_next_count, sizeof(int));
    cudaMalloc(&d_bottleneck, sizeof(int));
    cudaMalloc(&d_done, sizeof(int));

    cudaMemcpy(d_nindex, h_nindex, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nlist, h_nlist, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cap, h_cap, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rnindex, h_rnindex, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rnlist, h_rnlist, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_retoe, h_retoe, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_flow, 0, num_edges*sizeof(int));
    cudaMemset(d_bottleneck, 0, sizeof(int));
    cudaMemset(d_done, 0, sizeof(int));

    // Query max cooperative launch occupancy
    int block_size = 256;
    int num_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks, k_ek_persistent_bfs_augment, block_size, 0);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int max_blocks = num_blocks * prop.multiProcessorCount;

    printf("[EK-PERSISTENT] Launching persistent kernel: %d blocks x %d threads = %d threads (SMs=%d)\n",
           max_blocks, block_size, max_blocks * block_size, prop.multiProcessorCount);

    auto t_launch = std::chrono::high_resolution_clock::now();

    // Launch cooperative kernel — entire solve in one launch
    void* args[] = {
        &d_nindex, &d_nlist, &d_cap, &d_flow,
        &d_rnindex, &d_rnlist, &d_retoe,
        &num_nodes, &source, &sink,
        &d_parent, &d_parent_edge, &d_parent_dir,
        &d_frontier, &d_next_frontier, &d_frontier_size, &d_next_count,
        &d_bottleneck, &d_done
    };

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)k_ek_persistent_bfs_augment,
        dim3(max_blocks), dim3(block_size),
        args, 0, 0);

    if (err != cudaSuccess) {
        printf("[EK-PERSISTENT] ERROR: cudaLaunchCooperativeKernel failed: %s\n",
               cudaGetErrorString(err));
        printf("[EK-PERSISTENT] Falling back to standard Edmonds-Karp\n");
        // Fallback to regular EK
        cudaFree(d_nindex); cudaFree(d_nlist); cudaFree(d_cap); cudaFree(d_flow);
        cudaFree(d_rnindex); cudaFree(d_rnlist); cudaFree(d_retoe);
        cudaFree(d_parent); cudaFree(d_parent_edge); cudaFree(d_parent_dir);
        cudaFree(d_frontier); cudaFree(d_next_frontier);
        cudaFree(d_frontier_size); cudaFree(d_next_count);
        cudaFree(d_bottleneck); cudaFree(d_done);
        return cuda_edmonds_karp_solve(num_nodes, source, sink,
            h_nindex, h_nlist, h_cap, num_edges,
            h_rnindex, h_rnlist, h_retoe);
    }

    cudaDeviceSynchronize();

    auto t_end = std::chrono::high_resolution_clock::now();

    // Read results
    int total_flow = 0;
    cudaMemcpy(&total_flow, d_bottleneck, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(result.edge_flows.data(), d_flow, num_edges*sizeof(int), cudaMemcpyDeviceToHost);

    double ms_total = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double ms_kernel = std::chrono::duration<double, std::milli>(t_end - t_launch).count();
    printf("[TIMING]       GPU EK-Persistent: flow=%d (kernel=%.1f ms, total=%.1f ms)\n",
           total_flow, ms_kernel, ms_total);

    cudaFree(d_nindex); cudaFree(d_nlist); cudaFree(d_cap); cudaFree(d_flow);
    cudaFree(d_rnindex); cudaFree(d_rnlist); cudaFree(d_retoe);
    cudaFree(d_parent); cudaFree(d_parent_edge); cudaFree(d_parent_dir);
    cudaFree(d_frontier); cudaFree(d_next_frontier);
    cudaFree(d_frontier_size); cudaFree(d_next_count);
    cudaFree(d_bottleneck); cudaFree(d_done);

    result.max_flow = total_flow;
    return result;
}


// ============================================================
// GPU Dinic's: Dinic phases for bulk flow, Edmonds-Karp for stragglers
//
// Phase 1: Dinic's (fwd BFS + bwd BFS + multi-augment) — gets ~83% of flow fast
// Phase 2: Edmonds-Karp (full BFS per augment) — handles remaining ~17% correctly
// ============================================================
CudaMaxFlowResult cuda_dinic_solve(
    int num_nodes, int source, int sink,
    const int* h_nindex, const int* h_nlist, const int* h_cap,
    int num_edges,
    const int* h_rnindex, const int* h_rnlist, const int* h_retoe
) {
    CudaMaxFlowResult result;
    result.edge_flows.resize(num_edges);

    // Allocate all GPU arrays (shared by both phases)
    int *d_nindex, *d_nlist, *d_cap, *d_flow;
    int *d_rnindex, *d_rnlist, *d_retoe;
    int *d_level, *d_reach;
    int *d_parent, *d_parent_edge, *d_parent_dir;
    int *d_frontier, *d_next_frontier, *d_next_count;
    int *d_total_flow, *d_num_paths, *d_bottleneck;

    cudaMalloc(&d_nindex, (num_nodes+1)*sizeof(int));
    cudaMalloc(&d_nlist, num_edges*sizeof(int));
    cudaMalloc(&d_cap, num_edges*sizeof(int));
    cudaMalloc(&d_flow, num_edges*sizeof(int));
    cudaMalloc(&d_rnindex, (num_nodes+1)*sizeof(int));
    cudaMalloc(&d_rnlist, num_edges*sizeof(int));
    cudaMalloc(&d_retoe, num_edges*sizeof(int));
    cudaMalloc(&d_level, num_nodes*sizeof(int));
    cudaMalloc(&d_reach, num_nodes*sizeof(int));
    cudaMalloc(&d_parent, num_nodes*sizeof(int));
    cudaMalloc(&d_parent_edge, num_nodes*sizeof(int));
    cudaMalloc(&d_parent_dir, num_nodes*sizeof(int));
    cudaMalloc(&d_frontier, num_nodes*sizeof(int));
    cudaMalloc(&d_next_frontier, num_nodes*sizeof(int));
    cudaMalloc(&d_next_count, sizeof(int));
    cudaMalloc(&d_total_flow, sizeof(int));
    cudaMalloc(&d_num_paths, sizeof(int));
    cudaMalloc(&d_bottleneck, sizeof(int));

    cudaMemcpy(d_nindex, h_nindex, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nlist, h_nlist, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cap, h_cap, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rnindex, h_rnindex, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rnlist, h_rnlist, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_retoe, h_retoe, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_flow, 0, num_edges*sizeof(int));

    const int B = 256;
    int total_flow = 0, total_paths = 0, dinic_phases = 0;
    int src_val = source, sink_val = sink;

    // ==== PHASE 1: Dinic's (bulk flow) ====
    while (true) {
        // Forward BFS: build level graph
        cudaMemset(d_level, 0xFF, num_nodes*sizeof(int));
        int zero = 0;
        cudaMemcpy(d_level+source, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier, &src_val, sizeof(int), cudaMemcpyHostToDevice);

        int fsize = 1, current_level = 0;
        bool found_sink = false;

        while (fsize > 0) {
            cudaMemset(d_next_count, 0, sizeof(int));
            k_dinic_bfs_expand<<<(fsize+B-1)/B, B>>>(
                d_nindex, d_nlist, d_cap, d_flow,
                d_rnindex, d_rnlist, d_retoe,
                d_frontier, fsize, current_level,
                d_level, d_next_frontier, d_next_count);
            int sink_lev;
            cudaMemcpy(&sink_lev, d_level+sink, sizeof(int), cudaMemcpyDeviceToHost);
            if (sink_lev >= 0) { found_sink = true; break; }
            cudaMemcpy(&fsize, d_next_count, sizeof(int), cudaMemcpyDeviceToHost);
            std::swap(d_frontier, d_next_frontier);
            current_level++;
        }

        if (!found_sink) break;

        // Backward BFS: mark nodes that can reach sink
        cudaMemset(d_reach, 0, num_nodes*sizeof(int));
        int one = 1;
        cudaMemcpy(d_reach+sink, &one, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier, &sink_val, sizeof(int), cudaMemcpyHostToDevice);
        fsize = 1;

        while (fsize > 0) {
            cudaMemset(d_next_count, 0, sizeof(int));
            k_dinic_bfs_backward<<<(fsize+B-1)/B, B>>>(
                d_nindex, d_nlist, d_rnindex, d_rnlist, d_retoe,
                d_cap, d_flow, d_level,
                d_frontier, fsize,
                d_reach, d_next_frontier, d_next_count);
            cudaMemcpy(&fsize, d_next_count, sizeof(int), cudaMemcpyDeviceToHost);
            std::swap(d_frontier, d_next_frontier);
        }

        // Multi-augment on pruned level graph
        k_dinic_multi_augment<<<1, 1>>>(
            d_nindex, d_nlist, d_rnindex, d_rnlist, d_retoe,
            d_cap, d_flow, d_level, d_reach,
            source, sink, d_total_flow, d_num_paths);

        int phase_flow, phase_paths;
        cudaMemcpy(&phase_flow, d_total_flow, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&phase_paths, d_num_paths, sizeof(int), cudaMemcpyDeviceToHost);

        total_flow += phase_flow;
        total_paths += phase_paths;
        dinic_phases++;

        if (phase_flow == 0) break;
    }

    printf("[TIMING]       GPU Dinic phase: %d phases, %d paths, flow=%d\n",
           dinic_phases, total_paths, total_flow);

    // ==== PHASE 2: CPU EIBFS attempt (skipped — GPU EK fallback handles it) ====
    if (false)  // EIBFS disabled: tree maintenance too slow on this graph
    {
        auto t_ibfs_start = std::chrono::high_resolution_clock::now();
        int* h_flow = result.edge_flows.data();

        // EIBFS state arrays
        std::vector<int> dist(num_nodes, num_nodes);  // distance from source
        std::vector<int> par(num_nodes, -1);           // parent node
        std::vector<int> par_e(num_nodes, -1);         // parent edge (CSR index)
        std::vector<int> par_d(num_nodes, 0);          // parent direction (+1/-1)
        std::vector<int> firstSon(num_nodes, -1);      // children linked list head
        std::vector<int> nextSib(num_nodes, -1);       // next sibling in children list

        // Build initial BFS tree with children tracking
        auto full_bfs = [&]() -> bool {
            std::fill(dist.begin(), dist.end(), num_nodes);
            std::fill(par.begin(), par.end(), -1);
            std::fill(firstSon.begin(), firstSon.end(), -1);
            std::fill(nextSib.begin(), nextSib.end(), -1);
            dist[source] = 0; par[source] = source;
            std::vector<int> q;
            q.reserve(num_nodes);
            q.push_back(source);
            for (int qi = 0; qi < (int)q.size(); qi++) {
                int u = q[qi];
                for (int e = h_nindex[u]; e < h_nindex[u+1]; e++) {
                    int v = h_nlist[e];
                    if (dist[v] != num_nodes) continue;
                    if (h_cap[e] - h_flow[e] <= 0) continue;
                    dist[v] = dist[u] + 1;
                    par[v] = u; par_e[v] = e; par_d[v] = 1;
                    nextSib[v] = firstSon[u]; firstSon[u] = v;
                    q.push_back(v);
                }
                for (int re = h_rnindex[u]; re < h_rnindex[u+1]; re++) {
                    int v = h_rnlist[re];
                    int ef = h_retoe[re];
                    if (dist[v] != num_nodes) continue;
                    if (h_flow[ef] <= 0) continue;
                    dist[v] = dist[u] + 1;
                    par[v] = u; par_e[v] = ef; par_d[v] = -1;
                    nextSib[v] = firstSon[u]; firstSon[u] = v;
                    q.push_back(v);
                }
            }
            return dist[sink] < num_nodes;
        };

        // Remove node from parent's children list
        auto remove_child = [&](int child) {
            int p = par[child];
            if (p == -1 || p == child) return;
            if (firstSon[p] == child) {
                firstSon[p] = nextSib[child];
            } else {
                int c = firstSon[p];
                while (c != -1 && nextSib[c] != child) c = nextSib[c];
                if (c != -1) nextSib[c] = nextSib[child];
            }
            nextSib[child] = -1;
        };

        // Add node as child of new parent
        auto add_child = [&](int child, int parent_node) {
            nextSib[child] = firstSon[parent_node];
            firstSon[parent_node] = child;
        };

        int ibfs_augs = 0, ibfs_flow = 0, bfs_rebuilds = 0;

        const double IBFS_TIMEOUT = 2.0;  // seconds
        bool ibfs_timed_out = false;

        while (full_bfs()) {
            bfs_rebuilds++;
            bool need_rebuild = false;

            while (!need_rebuild && dist[sink] < num_nodes) {
                // Check timeout
                auto t_now = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration<double>(t_now - t_ibfs_start).count() > IBFS_TIMEOUT) {
                    ibfs_timed_out = true;
                    need_rebuild = true;
                    break;
                }
                // Trace path, find bottleneck
                int bn = INT_MAX;
                for (int v = sink; v != source; v = par[v]) {
                    int e = par_e[v];
                    int r = (par_d[v] == 1) ? (h_cap[e] - h_flow[e]) : h_flow[e];
                    if (r < bn) bn = r;
                }
                if (bn <= 0) { need_rebuild = true; break; }

                // Push flow, collect saturated edges
                std::vector<int> orphans;
                for (int v = sink; v != source; v = par[v]) {
                    int e = par_e[v];
                    if (par_d[v] == 1) h_flow[e] += bn; else h_flow[e] -= bn;
                    // Check if parent edge saturated
                    int r = (par_d[v] == 1) ? (h_cap[e] - h_flow[e]) : h_flow[e];
                    if (r <= 0) orphans.push_back(v);
                }
                total_flow += bn;
                ibfs_augs++;
                ibfs_flow += bn;

                // EIBFS orphan processing with cascading
                std::queue<int> orphan_q;
                for (int o : orphans) {
                    remove_child(o);
                    par[o] = -1;
                    orphan_q.push(o);
                }

                while (!orphan_q.empty()) {
                    int o = orphan_q.front(); orphan_q.pop();
                    if (o == source || o == sink) continue;
                    int old_dist = dist[o];

                    // Try to re-adopt at same distance (dist-1 parent)
                    bool adopted = false;

                    // Forward edges TO o: u→o with residual, dist[u]==old_dist-1
                    for (int re = h_rnindex[o]; re < h_rnindex[o+1] && !adopted; re++) {
                        int u = h_rnlist[re];
                        int ef = h_retoe[re];
                        if (dist[u] != old_dist - 1) continue;
                        if (par[u] == -1 && u != source) continue;  // u must be in tree
                        if (h_cap[ef] - h_flow[ef] <= 0) continue;
                        par[o] = u; par_e[o] = ef; par_d[o] = 1;
                        add_child(o, u);
                        adopted = true;
                    }
                    // Backward residual: o→w with flow>0, dist[w]==old_dist-1
                    for (int e = h_nindex[o]; e < h_nindex[o+1] && !adopted; e++) {
                        int w = h_nlist[e];
                        if (dist[w] != old_dist - 1) continue;
                        if (par[w] == -1 && w != source) continue;
                        if (h_flow[e] <= 0) continue;
                        par[o] = w; par_e[o] = e; par_d[o] = -1;
                        add_child(o, w);
                        adopted = true;
                    }

                    if (adopted) continue;

                    // Try higher distances (relabel upward)
                    for (int try_dist = old_dist; try_dist < old_dist + 50 && !adopted; try_dist++) {
                        for (int re = h_rnindex[o]; re < h_rnindex[o+1] && !adopted; re++) {
                            int u = h_rnlist[re];
                            int ef = h_retoe[re];
                            if (dist[u] != try_dist) continue;
                            if (par[u] == -1 && u != source) continue;
                            if (h_cap[ef] - h_flow[ef] <= 0) continue;
                            dist[o] = try_dist + 1;
                            par[o] = u; par_e[o] = ef; par_d[o] = 1;
                            add_child(o, u);
                            adopted = true;
                        }
                        for (int e = h_nindex[o]; e < h_nindex[o+1] && !adopted; e++) {
                            int w = h_nlist[e];
                            if (dist[w] != try_dist) continue;
                            if (par[w] == -1 && w != source) continue;
                            if (h_flow[e] <= 0) continue;
                            dist[o] = try_dist + 1;
                            par[o] = w; par_e[o] = e; par_d[o] = -1;
                            add_child(o, w);
                            adopted = true;
                        }
                    }

                    if (!adopted) {
                        // Cascade: orphan all children of o
                        int c = firstSon[o];
                        while (c != -1) {
                            int next_c = nextSib[c];
                            par[c] = -1;
                            nextSib[c] = -1;
                            orphan_q.push(c);
                            c = next_c;
                        }
                        firstSon[o] = -1;
                        dist[o] = num_nodes;
                        par[o] = -1;
                    }
                }

                // Check if sink still reachable
                if (par[sink] == -1 || dist[sink] >= num_nodes) {
                    need_rebuild = true;
                }
            } // end inner while

            if (ibfs_timed_out) break;
        } // end outer while(full_bfs())

        auto t_ibfs_end = std::chrono::high_resolution_clock::now();
        double ibfs_s = std::chrono::duration<double>(t_ibfs_end - t_ibfs_start).count();
        printf("[TIMING]       CPU EIBFS cleanup: %d augs, %d BFS rebuilds, flow=%d, total=%d (%.3f s)%s\n",
               ibfs_augs, bfs_rebuilds, ibfs_flow, total_flow, ibfs_s,
               ibfs_timed_out ? " [TIMEOUT - GPU EK fallback]" : "");
    }

    // ==== PHASE 3: GPU EK (remaining flow after Dinic) ====
    // Flow is already on GPU from Dinic phase. No re-upload needed.
    {
        int ek_augs = 0;
        while (true) {
            cudaMemset(d_parent, 0xFF, num_nodes*sizeof(int));
            cudaMemcpy(d_parent+source, &src_val, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_frontier, &src_val, sizeof(int), cudaMemcpyHostToDevice);
            int fsize = 1;
            bool found = false;
            while (fsize > 0) {
                cudaMemset(d_next_count, 0, sizeof(int));
                k_ek_bfs_expand<<<(fsize+B-1)/B, B>>>(
                    d_nindex, d_nlist, d_cap, d_flow,
                    d_rnindex, d_rnlist, d_retoe,
                    d_frontier, fsize,
                    d_parent, d_parent_edge, d_parent_dir,
                    d_next_frontier, d_next_count);
                int sp;
                cudaMemcpy(&sp, d_parent+sink, sizeof(int), cudaMemcpyDeviceToHost);
                if (sp != -1) { found = true; break; }
                cudaMemcpy(&fsize, d_next_count, sizeof(int), cudaMemcpyDeviceToHost);
                std::swap(d_frontier, d_next_frontier);
            }
            if (!found) break;
            k_ek_trace_push<<<1,1>>>(d_parent, d_parent_edge, d_parent_dir,
                                      d_cap, d_flow, source, sink, d_bottleneck);
            int bn;
            cudaMemcpy(&bn, d_bottleneck, sizeof(int), cudaMemcpyDeviceToHost);
            total_flow += bn;
            ek_augs++;
        }
        if (ek_augs > 0)
            printf("[TIMING]       GPU EK fallback: %d augmentations, total flow=%d\n", ek_augs, total_flow);
    }

    // Download final flow
    cudaMemcpy(result.edge_flows.data(), d_flow, num_edges*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_nindex); cudaFree(d_nlist); cudaFree(d_cap); cudaFree(d_flow);
    cudaFree(d_rnindex); cudaFree(d_rnlist); cudaFree(d_retoe);
    cudaFree(d_level); cudaFree(d_reach);
    cudaFree(d_parent); cudaFree(d_parent_edge); cudaFree(d_parent_dir);
    cudaFree(d_frontier); cudaFree(d_next_frontier); cudaFree(d_next_count);
    cudaFree(d_total_flow); cudaFree(d_num_paths); cudaFree(d_bottleneck);

    result.max_flow = total_flow;
    return result;
}

// ============================================================
// Dinic + Persistent EK: Dinic for bulk, persistent kernel for remainder
// ============================================================
CudaMaxFlowResult cuda_dinic_persistent_solve(
    int num_nodes, int source, int sink,
    const int* h_nindex, const int* h_nlist, const int* h_cap,
    int num_edges,
    const int* h_rnindex, const int* h_rnlist, const int* h_retoe
) {
    CudaMaxFlowResult result;
    result.edge_flows.resize(num_edges);

    int *d_nindex, *d_nlist, *d_cap, *d_flow;
    int *d_rnindex, *d_rnlist, *d_retoe;
    int *d_level, *d_reach;
    int *d_parent, *d_parent_edge, *d_parent_dir;
    int *d_frontier, *d_next_frontier, *d_next_count;
    int *d_frontier_size;
    int *d_total_flow, *d_num_paths, *d_bottleneck, *d_done;

    cudaMalloc(&d_nindex, (num_nodes+1)*sizeof(int));
    cudaMalloc(&d_nlist, num_edges*sizeof(int));
    cudaMalloc(&d_cap, num_edges*sizeof(int));
    cudaMalloc(&d_flow, num_edges*sizeof(int));
    cudaMalloc(&d_rnindex, (num_nodes+1)*sizeof(int));
    cudaMalloc(&d_rnlist, num_edges*sizeof(int));
    cudaMalloc(&d_retoe, num_edges*sizeof(int));
    cudaMalloc(&d_level, num_nodes*sizeof(int));
    cudaMalloc(&d_reach, num_nodes*sizeof(int));
    cudaMalloc(&d_parent, num_nodes*sizeof(int));
    cudaMalloc(&d_parent_edge, num_nodes*sizeof(int));
    cudaMalloc(&d_parent_dir, num_nodes*sizeof(int));
    cudaMalloc(&d_frontier, num_nodes*sizeof(int));
    cudaMalloc(&d_next_frontier, num_nodes*sizeof(int));
    cudaMalloc(&d_next_count, sizeof(int));
    cudaMalloc(&d_frontier_size, sizeof(int));
    cudaMalloc(&d_total_flow, sizeof(int));
    cudaMalloc(&d_num_paths, sizeof(int));
    cudaMalloc(&d_bottleneck, sizeof(int));
    cudaMalloc(&d_done, sizeof(int));

    cudaMemcpy(d_nindex, h_nindex, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nlist, h_nlist, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cap, h_cap, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rnindex, h_rnindex, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rnlist, h_rnlist, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_retoe, h_retoe, num_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_flow, 0, num_edges*sizeof(int));

    const int B = 256;
    int total_flow = 0, total_paths = 0, dinic_phases = 0;
    int src_val = source, sink_val = sink;

    auto t_start = std::chrono::high_resolution_clock::now();

    // ==== PHASE 1: Dinic's (bulk flow) ====
    while (true) {
        cudaMemset(d_level, 0xFF, num_nodes*sizeof(int));
        int zero = 0;
        cudaMemcpy(d_level+source, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier, &src_val, sizeof(int), cudaMemcpyHostToDevice);

        int fsize = 1, current_level = 0;
        bool found_sink = false;

        while (fsize > 0) {
            cudaMemset(d_next_count, 0, sizeof(int));
            k_dinic_bfs_expand<<<(fsize+B-1)/B, B>>>(
                d_nindex, d_nlist, d_cap, d_flow,
                d_rnindex, d_rnlist, d_retoe,
                d_frontier, fsize, current_level,
                d_level, d_next_frontier, d_next_count);
            int sink_lev;
            cudaMemcpy(&sink_lev, d_level+sink, sizeof(int), cudaMemcpyDeviceToHost);
            if (sink_lev >= 0) { found_sink = true; break; }
            cudaMemcpy(&fsize, d_next_count, sizeof(int), cudaMemcpyDeviceToHost);
            std::swap(d_frontier, d_next_frontier);
            current_level++;
        }

        if (!found_sink) break;

        // Backward BFS: mark nodes that can reach sink
        cudaMemset(d_reach, 0, num_nodes*sizeof(int));
        int one = 1;
        cudaMemcpy(d_reach+sink, &one, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier, &sink_val, sizeof(int), cudaMemcpyHostToDevice);
        fsize = 1;

        while (fsize > 0) {
            cudaMemset(d_next_count, 0, sizeof(int));
            k_dinic_bfs_backward<<<(fsize+B-1)/B, B>>>(
                d_nindex, d_nlist, d_rnindex, d_rnlist, d_retoe,
                d_cap, d_flow, d_level,
                d_frontier, fsize,
                d_reach, d_next_frontier, d_next_count);
            cudaMemcpy(&fsize, d_next_count, sizeof(int), cudaMemcpyDeviceToHost);
            std::swap(d_frontier, d_next_frontier);
        }

        // Multi-augment on pruned level graph
        k_dinic_multi_augment<<<1, 1>>>(
            d_nindex, d_nlist, d_rnindex, d_rnlist, d_retoe,
            d_cap, d_flow, d_level, d_reach,
            source, sink, d_total_flow, d_num_paths);

        int phase_flow, phase_paths;
        cudaMemcpy(&phase_flow, d_total_flow, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&phase_paths, d_num_paths, sizeof(int), cudaMemcpyDeviceToHost);

        total_flow += phase_flow;
        total_paths += phase_paths;
        dinic_phases++;

        if (phase_flow == 0) break;
    }

    auto t_dinic = std::chrono::high_resolution_clock::now();
    double ms_dinic = std::chrono::duration<double, std::milli>(t_dinic - t_start).count();
    printf("[TIMING]       GPU Dinic phase: %d phases, %d paths, flow=%d (%.1f ms)\n",
           dinic_phases, total_paths, total_flow, ms_dinic);

    // ==== PHASE 2: Persistent EK for remaining flow ====
    {
        cudaMemset(d_bottleneck, 0, sizeof(int));
        cudaMemset(d_done, 0, sizeof(int));

        int block_size = 256;
        int num_blocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks, k_ek_persistent_bfs_augment, block_size, 0);
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        int max_blocks = num_blocks * prop.multiProcessorCount;

        void* args[] = {
            &d_nindex, &d_nlist, &d_cap, &d_flow,
            &d_rnindex, &d_rnlist, &d_retoe,
            &num_nodes, &source, &sink,
            &d_parent, &d_parent_edge, &d_parent_dir,
            &d_frontier, &d_next_frontier, &d_frontier_size, &d_next_count,
            &d_bottleneck, &d_done
        };

        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)k_ek_persistent_bfs_augment,
            dim3(max_blocks), dim3(block_size),
            args, 0, 0);

        if (err != cudaSuccess) {
            printf("[DINIC-PERSISTENT] Cooperative launch failed: %s, falling back to standard EK\n",
                   cudaGetErrorString(err));
            // Standard EK fallback
            int ek_augs = 0;
            while (true) {
                cudaMemset(d_parent, 0xFF, num_nodes*sizeof(int));
                cudaMemcpy(d_parent+source, &src_val, sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_frontier, &src_val, sizeof(int), cudaMemcpyHostToDevice);
                int fs2 = 1;
                bool found = false;
                while (fs2 > 0) {
                    cudaMemset(d_next_count, 0, sizeof(int));
                    k_ek_bfs_expand<<<(fs2+B-1)/B, B>>>(
                        d_nindex, d_nlist, d_cap, d_flow,
                        d_rnindex, d_rnlist, d_retoe,
                        d_frontier, fs2,
                        d_parent, d_parent_edge, d_parent_dir,
                        d_next_frontier, d_next_count);
                    int sp;
                    cudaMemcpy(&sp, d_parent+sink, sizeof(int), cudaMemcpyDeviceToHost);
                    if (sp != -1) { found = true; break; }
                    cudaMemcpy(&fs2, d_next_count, sizeof(int), cudaMemcpyDeviceToHost);
                    std::swap(d_frontier, d_next_frontier);
                }
                if (!found) break;
                k_ek_trace_push<<<1,1>>>(d_parent, d_parent_edge, d_parent_dir,
                                          d_cap, d_flow, source, sink, d_bottleneck);
                int bn;
                cudaMemcpy(&bn, d_bottleneck, sizeof(int), cudaMemcpyDeviceToHost);
                total_flow += bn;
                ek_augs++;
            }
            if (ek_augs > 0)
                printf("[TIMING]       EK fallback: %d augs, total flow=%d\n", ek_augs, total_flow);
        } else {
            cudaDeviceSynchronize();
            int ek_flow = 0;
            cudaMemcpy(&ek_flow, d_bottleneck, sizeof(int), cudaMemcpyDeviceToHost);
            total_flow += ek_flow;

            auto t_ek = std::chrono::high_resolution_clock::now();
            double ms_ek = std::chrono::duration<double, std::milli>(t_ek - t_dinic).count();
            printf("[TIMING]       Persistent EK fallback: flow=%d (%.1f ms), total flow=%d\n",
                   ek_flow, ms_ek, total_flow);
        }
    }

    cudaMemcpy(result.edge_flows.data(), d_flow, num_edges*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_nindex); cudaFree(d_nlist); cudaFree(d_cap); cudaFree(d_flow);
    cudaFree(d_rnindex); cudaFree(d_rnlist); cudaFree(d_retoe);
    cudaFree(d_level); cudaFree(d_reach);
    cudaFree(d_parent); cudaFree(d_parent_edge); cudaFree(d_parent_dir);
    cudaFree(d_frontier); cudaFree(d_next_frontier); cudaFree(d_next_count);
    cudaFree(d_frontier_size);
    cudaFree(d_total_flow); cudaFree(d_num_paths); cudaFree(d_bottleneck);
    cudaFree(d_done);

    result.max_flow = total_flow;
    return result;
}

} // namespace qflow
