// ============================================================
// GPU kernels for optimize_positions_dynamic
//
// 1. FindNearest: parallel manifold walk (one thread per quad vertex)
// 2. FillCSR: parallel sparse matrix assembly (one thread per edge)
// ============================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

// ---- FindNearest kernel ----
// Each thread handles one quad vertex.
// iteration==0: pick nearest from Vset candidates, project to tangent plane.
// iteration>0: greedy walk on adjacency graph, rotating diffs along the way.
__global__ void k_find_nearest(
    const double* V,        // base mesh vertex positions [3 * nV], col-major from Eigen
    const double* N,        // base mesh normals [3 * nV], col-major
    double* O_compact,      // quad vertex positions [3 * nQ], flat (x,y,z per vertex)
    int* Vind,              // current nearest base vertex per quad vertex [nQ]
    const int* adj_ptr,     // CSR row pointers for adjacency [nV+1]
    const int* adj_list,    // CSR column indices for adjacency
    const int* vset_ptr,    // CSR row pointers for Vset [nQ+1]
    const int* vset_list,   // Vset column indices
    double* diffs,          // target edge vectors [3 * nDE], flat (x,y,z per dedge)
    const int* dedge_ptr,   // CSR row pointers for dedges [nQ+1]
    const int* dedge_list,  // directed edge IDs per quad vertex
    int nQ, int iteration
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nQ) return;

    double ox = O_compact[i * 3], oy = O_compact[i * 3 + 1], oz = O_compact[i * 3 + 2];

    if (iteration == 0) {
        // First iteration: find nearest among Vset candidates
        double min_dis = 1e30;
        int min_ind = -1;
        for (int j = vset_ptr[i]; j < vset_ptr[i + 1]; j++) {
            int v = vset_list[j];
            double dx = V[v * 3] - ox, dy = V[v * 3 + 1] - oy, dz = V[v * 3 + 2] - oz;
            double dis = dx * dx + dy * dy + dz * dz;
            if (dis < min_dis) { min_dis = dis; min_ind = v; }
        }
        if (min_ind >= 0) {
            Vind[i] = min_ind;
            // Project onto tangent plane
            double nx = N[min_ind * 3], ny = N[min_ind * 3 + 1], nz = N[min_ind * 3 + 2];
            double dot = (ox - V[min_ind * 3]) * nx +
                         (oy - V[min_ind * 3 + 1]) * ny +
                         (oz - V[min_ind * 3 + 2]) * nz;
            O_compact[i * 3]     = ox - dot * nx;
            O_compact[i * 3 + 1] = oy - dot * ny;
            O_compact[i * 3 + 2] = oz - dot * nz;
        }
    } else {
        // Subsequent: greedy walk on adjacency graph
        int cur = Vind[i];
        if (cur < 0) return;

        double cur_nx = N[cur * 3], cur_ny = N[cur * 3 + 1], cur_nz = N[cur * 3 + 2];
        double cur_dis = (ox - V[cur * 3]) * (ox - V[cur * 3]) +
                         (oy - V[cur * 3 + 1]) * (oy - V[cur * 3 + 1]) +
                         (oz - V[cur * 3 + 2]) * (oz - V[cur * 3 + 2]);

        const double cos_thresh = 0.98480775301;  // cos(10 degrees)

        for (int walk = 0; walk < 100; walk++) {  // safety limit
            int best = -1;
            double best_dis = cur_dis;

            for (int j = adj_ptr[cur]; j < adj_ptr[cur + 1]; j++) {
                int v = adj_list[j];
                // Normal compatibility check
                double ndot = N[v * 3] * cur_nx + N[v * 3 + 1] * cur_ny + N[v * 3 + 2] * cur_nz;
                if (ndot < cos_thresh) continue;
                double dx = ox - V[v * 3], dy = oy - V[v * 3 + 1], dz = oz - V[v * 3 + 2];
                double dis = dx * dx + dy * dy + dz * dz;
                if (dis < best_dis) { best_dis = dis; best = v; }
            }
            if (best == -1) break;

            // Rotate diffs: build rotation from n1 to n2
            double n1x = N[cur * 3], n1y = N[cur * 3 + 1], n1z = N[cur * 3 + 2];
            double n2x = N[best * 3], n2y = N[best * 3 + 1], n2z = N[best * 3 + 2];

            // axis = n1 × n2
            double ax = n1y * n2z - n1z * n2y;
            double ay = n1z * n2x - n1x * n2z;
            double az = n1x * n2y - n1y * n2x;
            double len = sqrt(ax * ax + ay * ay + az * az);
            double angle = atan2(len, n1x * n2x + n1y * n2y + n1z * n2z);

            if (len > 1e-12) {
                // Normalize axis
                ax /= len; ay /= len; az /= len;
                double c = cos(angle), s = sin(angle), t = 1.0 - c;

                // Rodrigues rotation matrix
                double m00 = t*ax*ax + c,     m01 = t*ax*ay - s*az, m02 = t*ax*az + s*ay;
                double m10 = t*ax*ay + s*az,  m11 = t*ay*ay + c,    m12 = t*ay*az - s*ax;
                double m20 = t*ax*az - s*ay,  m21 = t*ay*az + s*ax, m22 = t*az*az + c;

                // Rotate all diffs for this quad vertex
                for (int de_idx = dedge_ptr[i]; de_idx < dedge_ptr[i + 1]; de_idx++) {
                    int e = dedge_list[de_idx];
                    double dx = diffs[e * 3], dy = diffs[e * 3 + 1], dz = diffs[e * 3 + 2];
                    diffs[e * 3]     = m00 * dx + m01 * dy + m02 * dz;
                    diffs[e * 3 + 1] = m10 * dx + m11 * dy + m12 * dz;
                    diffs[e * 3 + 2] = m20 * dx + m21 * dy + m22 * dz;
                }
            }

            cur = best;
            cur_dis = best_dis;
            cur_nx = N[cur * 3]; cur_ny = N[cur * 3 + 1]; cur_nz = N[cur * 3 + 2];
        }
        Vind[i] = cur;
    }
}


// ---- FillCSR kernel ----
// Each thread handles one edge, contributes 16 values to CSR matrix + 4 to RHS.
__global__ void k_fill_csr(
    // Edge data
    const int* edge_i,          // [nEdges] source quad vertex per edge
    const int* edge_j,          // [nEdges] target quad vertex per edge
    const int* edge_de,         // [nEdges] directed edge ID per edge
    const int* edge_csr_pos,    // [nEdges * 16] CSR positions for 4x4 block
    const int* fixed_dim,       // [dim] whether dimension is fixed
    // Per-vertex data
    const double* Q_compact,    // [3 * nQ] cross field vectors
    const double* N_compact,    // [3 * nQ] normals
    const double* V_compact,    // [3 * nQ] base mesh positions
    // Per-edge data
    const double* diffs,        // [3 * nDE] target edge vectors
    // Solution vector (for fixed column RHS adjustment)
    const double* x,            // [dim] current solution
    // Output
    double* csr_val,            // [nnz] CSR values (atomicAdd)
    double* rhs,                // [dim] right-hand side (atomicAdd)
    int nEdges
) {
    int ei = blockIdx.x * blockDim.x + threadIdx.x;
    if (ei >= nEdges) return;

    int vi = edge_i[ei], vj = edge_j[ei], de = edge_de[ei];

    // Cross field basis at vertex i
    double qix = Q_compact[vi * 3], qiy = Q_compact[vi * 3 + 1], qiz = Q_compact[vi * 3 + 2];
    double nix = N_compact[vi * 3], niy = N_compact[vi * 3 + 1], niz = N_compact[vi * 3 + 2];
    // q_y = n × q
    double qiyx = niy * qiz - niz * qiy;
    double qiyy = niz * qix - nix * qiz;
    double qiyz = nix * qiy - niy * qix;

    // Cross field basis at vertex j
    double qjx = Q_compact[vj * 3], qjy = Q_compact[vj * 3 + 1], qjz = Q_compact[vj * 3 + 2];
    double njx = N_compact[vj * 3], njy = N_compact[vj * 3 + 1], njz = N_compact[vj * 3 + 2];
    double qjyx = njy * qjz - njz * qjy;
    double qjyy = njz * qjx - njx * qjz;
    double qjyz = njx * qjy - njy * qjx;

    // Target offset
    double tx = diffs[de * 3], ty = diffs[de * 3 + 1], tz = diffs[de * 3 + 2];

    // C = target_offset - (V_j - V_i)
    double cx = tx - (V_compact[vj * 3]     - V_compact[vi * 3]);
    double cy = ty - (V_compact[vj * 3 + 1] - V_compact[vi * 3 + 1]);
    double cz = tz - (V_compact[vj * 3 + 2] - V_compact[vi * 3 + 2]);

    // weights[0] = qx2, weights[1] = qy2, weights[2] = -qx, weights[3] = -qy
    double wx[4] = {qjx, qjyx, -qix, -qiyx};
    double wy[4] = {qjy, qjyy, -qiy, -qiyy};
    double wz[4] = {qjz, qjyz, -qiz, -qiyz};

    // vid[0] = j*2, vid[1] = j*2+1, vid[2] = i*2, vid[3] = i*2+1
    int vid[4] = {vj * 2, vj * 2 + 1, vi * 2, vi * 2 + 1};

    for (int ii = 0; ii < 4; ii++) {
        int row = vid[ii];
        if (fixed_dim[row]) continue;

        // RHS contribution: weights[ii] . C
        double rhs_val = wx[ii] * cx + wy[ii] * cy + wz[ii] * cz;
        atomicAdd(&rhs[row], rhs_val);

        for (int jj = 0; jj < 4; jj++) {
            int col = vid[jj];
            // weights[ii] . weights[jj]
            double val = wx[ii] * wx[jj] + wy[ii] * wy[jj] + wz[ii] * wz[jj];

            if (fixed_dim[col]) {
                // Move fixed column to RHS
                atomicAdd(&rhs[row], -val * x[col]);
            } else {
                int pos = edge_csr_pos[ei * 16 + ii * 4 + jj];
                if (pos >= 0) atomicAdd(&csr_val[pos], val);
            }
        }
    }
}


// ---- Host wrappers ----

extern "C" {

// Convert vector<vector<int>> to CSR (ptr + list) on host
static void flatten_to_csr(const std::vector<std::vector<int>>& vv,
                           std::vector<int>& ptr, std::vector<int>& list) {
    int n = (int)vv.size();
    ptr.resize(n + 1);
    ptr[0] = 0;
    for (int i = 0; i < n; i++) ptr[i + 1] = ptr[i] + (int)vv[i].size();
    list.resize(ptr[n]);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < (int)vv[i].size(); j++)
            list[ptr[i] + j] = vv[i][j];
}

// Convert vector<list<int>> to CSR
static void flatten_list_to_csr(const std::vector<std::list<int>>& vl,
                                std::vector<int>& ptr, std::vector<int>& list) {
    int n = (int)vl.size();
    ptr.resize(n + 1);
    ptr[0] = 0;
    for (int i = 0; i < n; i++) ptr[i + 1] = ptr[i] + (int)vl[i].size();
    list.resize(ptr[n]);
    for (int i = 0; i < n; i++) {
        int j = 0;
        for (auto val : vl[i]) list[ptr[i] + j++] = val;
    }
}

struct DynGpuContext {
    // Graph data (uploaded once)
    int *d_adj_ptr, *d_adj_list;
    int *d_vset_ptr, *d_vset_list;
    int *d_dedge_ptr, *d_dedge_list;
    double *d_V, *d_N;  // base mesh V, N (col-major → row-major converted)
    int nV, nQ, nDE;

    // Per-iteration data
    double *d_O_compact;
    int *d_Vind;
    double *d_diffs;

    // FillCSR data (uploaded once)
    int *d_edge_i, *d_edge_j, *d_edge_de, *d_edge_csr_pos;
    int *d_fixed_dim;
    double *d_Q_compact, *d_N_compact, *d_V_compact;
    double *d_x;
    double *d_csr_val, *d_rhs;
    int nEdges, dim, csr_nnz;

    bool initialized;
};

void* cuda_dyn_init(
    // Base mesh (Eigen col-major: V[3×nV], N[3×nV])
    const double* V_colmaj, const double* N_colmaj, int nV,
    // Adjacency CSR
    const int* adj_ptr, const int* adj_list, int adj_nnz,
    // Vset CSR
    const int* vset_ptr, const int* vset_list, int vset_nnz,
    // Dedges CSR
    const int* dedge_ptr, const int* dedge_list, int dedge_nnz,
    // Quad data
    int nQ, int nDE
) {
    auto* ctx = new DynGpuContext();
    ctx->nV = nV; ctx->nQ = nQ; ctx->nDE = nDE;
    ctx->initialized = true;

    // Convert V, N from col-major [3×nV] to row-major [nV×3]
    std::vector<double> V_row(nV * 3), N_row(nV * 3);
    for (int i = 0; i < nV; i++) {
        V_row[i * 3]     = V_colmaj[i * 3];      // Eigen col-major for Vector3d cols:
        V_row[i * 3 + 1] = V_colmaj[i * 3 + 1];  // V.col(i) is at V.data() + 3*i
        V_row[i * 3 + 2] = V_colmaj[i * 3 + 2];
        N_row[i * 3]     = N_colmaj[i * 3];
        N_row[i * 3 + 1] = N_colmaj[i * 3 + 1];
        N_row[i * 3 + 2] = N_colmaj[i * 3 + 2];
    }

    // Upload graph data
    cudaMalloc(&ctx->d_V, nV * 3 * sizeof(double));
    cudaMalloc(&ctx->d_N, nV * 3 * sizeof(double));
    cudaMemcpy(ctx->d_V, V_row.data(), nV * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_N, N_row.data(), nV * 3 * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&ctx->d_adj_ptr, (nV + 1) * sizeof(int));
    cudaMalloc(&ctx->d_adj_list, adj_nnz * sizeof(int));
    cudaMemcpy(ctx->d_adj_ptr, adj_ptr, (nV + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_adj_list, adj_list, adj_nnz * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&ctx->d_vset_ptr, (nQ + 1) * sizeof(int));
    cudaMalloc(&ctx->d_vset_list, vset_nnz * sizeof(int));
    cudaMemcpy(ctx->d_vset_ptr, vset_ptr, (nQ + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_vset_list, vset_list, vset_nnz * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&ctx->d_dedge_ptr, (nQ + 1) * sizeof(int));
    cudaMalloc(&ctx->d_dedge_list, dedge_nnz * sizeof(int));
    cudaMemcpy(ctx->d_dedge_ptr, dedge_ptr, (nQ + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_dedge_list, dedge_list, dedge_nnz * sizeof(int), cudaMemcpyHostToDevice);

    // Per-iteration buffers
    cudaMalloc(&ctx->d_O_compact, nQ * 3 * sizeof(double));
    cudaMalloc(&ctx->d_Vind, nQ * sizeof(int));
    cudaMalloc(&ctx->d_diffs, nDE * 3 * sizeof(double));

    // FillCSR buffers (allocated later on first use)
    ctx->d_edge_i = nullptr;
    ctx->nEdges = 0;

    return ctx;
}

void cuda_dyn_find_nearest(
    void* context,
    double* h_O_compact,  // [nQ * 3], in/out
    int* h_Vind,          // [nQ], in/out
    double* h_diffs,      // [nDE * 3], in/out
    int iteration
) {
    auto* ctx = (DynGpuContext*)context;
    int nQ = ctx->nQ, nDE = ctx->nDE;

    // Upload current state
    cudaMemcpy(ctx->d_O_compact, h_O_compact, nQ * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_Vind, h_Vind, nQ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_diffs, h_diffs, nDE * 3 * sizeof(double), cudaMemcpyHostToDevice);

    int B = 256;
    k_find_nearest<<<(nQ + B - 1) / B, B>>>(
        ctx->d_V, ctx->d_N, ctx->d_O_compact, ctx->d_Vind,
        ctx->d_adj_ptr, ctx->d_adj_list,
        ctx->d_vset_ptr, ctx->d_vset_list,
        ctx->d_diffs, ctx->d_dedge_ptr, ctx->d_dedge_list,
        nQ, iteration);

    // Download results
    cudaMemcpy(h_O_compact, ctx->d_O_compact, nQ * 3 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Vind, ctx->d_Vind, nQ * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_diffs, ctx->d_diffs, nDE * 3 * sizeof(double), cudaMemcpyDeviceToHost);
}

void cuda_dyn_fill_csr_init(
    void* context,
    const int* h_edge_i, const int* h_edge_j, const int* h_edge_de,
    const int* h_edge_csr_pos,  // [nEdges * 16]
    const int* h_fixed_dim,
    int nEdges, int dim, int csr_nnz
) {
    auto* ctx = (DynGpuContext*)context;
    ctx->nEdges = nEdges;
    ctx->dim = dim;
    ctx->csr_nnz = csr_nnz;

    cudaMalloc(&ctx->d_edge_i, nEdges * sizeof(int));
    cudaMalloc(&ctx->d_edge_j, nEdges * sizeof(int));
    cudaMalloc(&ctx->d_edge_de, nEdges * sizeof(int));
    cudaMalloc(&ctx->d_edge_csr_pos, nEdges * 16 * sizeof(int));
    cudaMalloc(&ctx->d_fixed_dim, dim * sizeof(int));
    cudaMalloc(&ctx->d_Q_compact, ctx->nQ * 3 * sizeof(double));
    cudaMalloc(&ctx->d_N_compact, ctx->nQ * 3 * sizeof(double));
    cudaMalloc(&ctx->d_V_compact, ctx->nQ * 3 * sizeof(double));
    cudaMalloc(&ctx->d_x, dim * sizeof(double));
    cudaMalloc(&ctx->d_csr_val, csr_nnz * sizeof(double));
    cudaMalloc(&ctx->d_rhs, dim * sizeof(double));

    cudaMemcpy(ctx->d_edge_i, h_edge_i, nEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_edge_j, h_edge_j, nEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_edge_de, h_edge_de, nEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_edge_csr_pos, h_edge_csr_pos, nEdges * 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_fixed_dim, h_fixed_dim, dim * sizeof(int), cudaMemcpyHostToDevice);
}

void cuda_dyn_fill_csr(
    void* context,
    const double* h_Q_compact,  // [nQ * 3]
    const double* h_N_compact,  // [nQ * 3]
    const double* h_V_compact,  // [nQ * 3]
    const double* h_diffs,      // [nDE * 3]
    const double* h_x,          // [dim]
    double* h_csr_val,          // [csr_nnz], output
    double* h_rhs               // [dim], output
) {
    auto* ctx = (DynGpuContext*)context;

    // Upload per-iteration data
    cudaMemcpy(ctx->d_Q_compact, h_Q_compact, ctx->nQ * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_N_compact, h_N_compact, ctx->nQ * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_V_compact, h_V_compact, ctx->nQ * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_diffs, h_diffs, ctx->nDE * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_x, h_x, ctx->dim * sizeof(double), cudaMemcpyHostToDevice);

    // Zero output
    cudaMemset(ctx->d_csr_val, 0, ctx->csr_nnz * sizeof(double));
    cudaMemset(ctx->d_rhs, 0, ctx->dim * sizeof(double));

    int B = 256;
    k_fill_csr<<<(ctx->nEdges + B - 1) / B, B>>>(
        ctx->d_edge_i, ctx->d_edge_j, ctx->d_edge_de, ctx->d_edge_csr_pos,
        ctx->d_fixed_dim,
        ctx->d_Q_compact, ctx->d_N_compact, ctx->d_V_compact,
        ctx->d_diffs, ctx->d_x,
        ctx->d_csr_val, ctx->d_rhs,
        ctx->nEdges);

    // Download results
    cudaMemcpy(h_csr_val, ctx->d_csr_val, ctx->csr_nnz * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rhs, ctx->d_rhs, ctx->dim * sizeof(double), cudaMemcpyDeviceToHost);
}

void cuda_dyn_destroy(void* context) {
    auto* ctx = (DynGpuContext*)context;
    if (!ctx) return;
    cudaFree(ctx->d_V); cudaFree(ctx->d_N);
    cudaFree(ctx->d_adj_ptr); cudaFree(ctx->d_adj_list);
    cudaFree(ctx->d_vset_ptr); cudaFree(ctx->d_vset_list);
    cudaFree(ctx->d_dedge_ptr); cudaFree(ctx->d_dedge_list);
    cudaFree(ctx->d_O_compact); cudaFree(ctx->d_Vind);
    cudaFree(ctx->d_diffs);
    if (ctx->d_edge_i) {
        cudaFree(ctx->d_edge_i); cudaFree(ctx->d_edge_j);
        cudaFree(ctx->d_edge_de); cudaFree(ctx->d_edge_csr_pos);
        cudaFree(ctx->d_fixed_dim);
        cudaFree(ctx->d_Q_compact); cudaFree(ctx->d_N_compact); cudaFree(ctx->d_V_compact);
        cudaFree(ctx->d_x);
        cudaFree(ctx->d_csr_val); cudaFree(ctx->d_rhs);
    }
    delete ctx;
}

} // extern "C"
