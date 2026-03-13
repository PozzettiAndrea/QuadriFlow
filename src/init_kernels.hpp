#pragma once

#ifdef WITH_CUDA

extern "C" {

// Step 1: ComputeSmoothNormal + ComputeVertexArea on GPU
void cuda_compute_smooth_normal(
    const int* F, int nFaces,
    const double* V, int nVerts,
    const int* V2E,
    const int* E2E,
    const int* nonManifold,
    const int* sharp_edges,
    double* N,    // output: 3 * nVerts
    double* Nf    // output: 3 * nFaces
);

void cuda_compute_vertex_area(
    const int* F, int nFaces,
    const double* V, int nVerts,
    const int* V2E,
    const int* E2E,
    const int* nonManifold,
    double* A     // output: nVerts
);

// Step 2: compute_direct_graph on GPU
int cuda_compute_direct_graph(
    const double* V_data, int nVerts,
    const int* F_data, int nFaces,
    int* V2E,       // output: nVerts
    int* E2E,       // output: 3 * nFaces
    int* boundary,  // output: nVerts
    int* nonManifold // output: nVerts
);

// Step 3: generate_adjacency_matrix on GPU (CSR output)
// Caller must free() the output arrays
void cuda_generate_adjacency_matrix(
    const int* F, int nFaces,
    const int* V2E, const int* E2E,
    const int* nonManifold, int nVerts,
    int** rowPtr_out, int** colInd_out, double** weights_out, int* nnz_out);

// Step 4: rho smoothing on GPU
void cuda_rho_smooth(
    const int* rowPtr, const int* colInd,
    int nVerts, int nnz,
    double* rho,  // in/out
    int iterations);

// Step 5: DownsampleGraph on GPU
void cuda_downsample_graph(
    const int* adjRowPtr, const int* adjColInd, const double* adjWeights,
    int nVerts, int nnz,
    const double* V, const double* N, const double* A,
    double* V_p, double* N_p, double* A_p,
    int* to_upper,    // 2 * vertexCount_p (column-major)
    int* to_lower,    // nVerts
    int** adjRowPtr_p_out, int** adjColInd_p_out, double** adjWeights_p_out,
    int* vertexCount_p_out, int* nnz_p_out);

}

#endif // WITH_CUDA
