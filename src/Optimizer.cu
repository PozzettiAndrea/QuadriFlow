#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "adjacent-matrix.hpp"

using namespace qflow;

__device__ __host__ glm::dvec3
middle_point(const glm::dvec3 &p0, const glm::dvec3 &n0, const glm::dvec3 &p1, const glm::dvec3 &n1) {
	/* How was this derived?
	*
	* Minimize \|x-p0\|^2 + \|x-p1\|^2, where
	* dot(n0, x) == dot(n0, p0)
	* dot(n1, x) == dot(n1, p1)
	*
	* -> Lagrange multipliers, set derivative = 0
	*  Use first 3 equalities to write x in terms of
	*  lambda_1 and lambda_2. Substitute that into the last
	*  two equations and solve for the lambdas. Finally,
	*  add a small epsilon term to avoid issues when n1=n2.
	*/
	double n0p0 = glm::dot(n0, p0), n0p1 = glm::dot(n0, p1),
		n1p0 = glm::dot(n1, p0), n1p1 = glm::dot(n1, p1),
		n0n1 = glm::dot(n0, n1),
		denom = 1.0f / (1.0f - n0n1*n0n1 + 1e-4f),
		lambda_0 = 2.0f*(n0p1 - n0p0 - n0n1*(n1p0 - n1p1))*denom,
		lambda_1 = 2.0f*(n1p0 - n1p1 - n0n1*(n0p1 - n0p0))*denom;

	return 0.5 * (p0 + p1) - 0.25 * (n0 * lambda_0 + n1 * lambda_1);
}

__device__ __host__ glm::dvec3
position_round_4(const  glm::dvec3 &o, const  glm::dvec3 &q,
const  glm::dvec3 &n, const  glm::dvec3 &p,
double scale) {
	double inv_scale = 1.0 / scale;
	glm::dvec3 t = glm::cross(n, q);
	glm::dvec3 d = p - o;
	return o +
		q * round(glm::dot(q, d) * inv_scale) * scale +
		t * round(glm::dot(t, d) * inv_scale) * scale;
}

__device__ __host__ glm::dvec3
position_floor_4(const glm::dvec3 &o, const glm::dvec3 &q,
const glm::dvec3 &n, const glm::dvec3 &p,
double scale) {
	double inv_scale = 1.0 / scale;
	glm::dvec3 t = glm::cross(n,q);
	glm::dvec3 d = p - o;
	return o +
		q * floor(glm::dot(q, d) * inv_scale) * scale +
		t * floor(glm::dot(t, d) * inv_scale) * scale;
}


__device__ __host__ double cudaSignum(double value) {
	return copysign(1.0, value);
}

__device__ __host__ void
compat_orientation_extrinsic_4(const glm::dvec3 &q0, const glm::dvec3 &n0,
const glm::dvec3 &q1, const glm::dvec3 &n1, glm::dvec3& value1, glm::dvec3& value2) {
	const glm::dvec3 A[2] = { q0, glm::cross(n0, q0) };
	const glm::dvec3 B[2] = { q1, glm::cross(n1, q1) };

	double best_score = -1e10;
	int best_a = 0, best_b = 0;

	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			double score = fabs(glm::dot(A[i], B[j]));
			if (score > best_score + 1e-6) {
				best_a = i;
				best_b = j;
				best_score = score;
			}
		}
	}
	const double dp = glm::dot(A[best_a], B[best_b]);
	value1 = A[best_a];
	value2 = B[best_b] * cudaSignum(dp);
}

__device__ __host__ void
compat_position_extrinsic_4(
const glm::dvec3 &p0, const glm::dvec3 &n0, const glm::dvec3 &q0, const glm::dvec3 &o0,
const glm::dvec3 &p1, const glm::dvec3 &n1, const glm::dvec3 &q1, const glm::dvec3 &o1,
double scale, glm::dvec3& v1, glm::dvec3& v2) {

	glm::dvec3 t0 = glm::cross(n0, q0), t1 = glm::cross(n1, q1);
	glm::dvec3 middle = middle_point(p0, n0, p1, n1);
	glm::dvec3 o0p = position_floor_4(o0, q0, n0, middle, scale);
	glm::dvec3 o1p = position_floor_4(o1, q1, n1, middle, scale);

	double best_cost = 1e10;
	int best_i = -1, best_j = -1;

	for (int i = 0; i<4; ++i) {
		glm::dvec3 o0t = o0p + (q0 * ((i & 1) * scale) + t0 * (((i & 2) >> 1) * scale));
		for (int j = 0; j<4; ++j) {
			glm::dvec3 o1t = o1p + (q1 * ((j & 1) * scale) + t1 * (((j & 2) >> 1) * scale));
			glm::dvec3 t = o0t - o1t;
			double cost = glm::dot(t, t);

			if (cost < best_cost) {
				best_i = i;
				best_j = j;
				best_cost = cost;
			}
		}
	}

	v1 = o0p + (q0 * ((best_i & 1) * scale) + t0 * (((best_i & 2) >> 1) * scale)),
	v2 = o1p + (q1 * ((best_j & 1) * scale) + t1 * (((best_j & 2) >> 1) * scale));
}

__global__ 
void cudaUpdateOrientation(int* phase, int num_phases, glm::dvec3* N, glm::dvec3* Q, Link* adj, int* adjOffset, int num_adj) {
	int pi = blockIdx.x * blockDim.x + threadIdx.x;

//	for (int pi = 0; pi < num_phases; ++pi) {
		if (pi >= num_phases)
			return;
		int i = phase[pi];
		glm::dvec3 n_i = N[i];
		double weight_sum = 0.0f;
		glm::dvec3 sum = Q[i];

		for (int l = adjOffset[i]; l < adjOffset[i + 1]; ++l) {
			Link link = adj[l];
			const int j = link.id;
			const double weight = link.weight;
			if (weight == 0)
				continue;
			glm::dvec3 n_j = N[j];
			glm::dvec3 q_j = Q[j];
			glm::dvec3 value1, value2;
			compat_orientation_extrinsic_4(sum, n_i, q_j, n_j, value1, value2);
			sum = value1 * weight_sum + value2 * weight;
			sum -= n_i*glm::dot(n_i, sum);
			weight_sum += weight;

			double norm = glm::length(sum);
			if (norm > 2.93873587705571876e-39f)
				sum /= norm;
		}

		if (weight_sum > 0) {
			Q[i] = sum;
		}
//	}
}

__global__
void cudaPropagateOrientationUpper(glm::dvec3* srcField, glm::ivec2* toUpper, glm::dvec3* N, glm::dvec3* destField, int num_orientation) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	for (int i = 0; i < num_orientation; ++i) {
		if (i >= num_orientation)
			return;
		for (int k = 0; k < 2; ++k) {
			int dest = toUpper[i][k];
			if (dest == -1)
				continue;
			glm::dvec3 q = srcField[i];
			glm::dvec3 n = N[dest];
			destField[dest] = q - n * glm::dot(n, q);
		}
//	}
}

__global__
void cudaPropagateOrientationLower(glm::ivec2* toUpper, glm::dvec3* Q, glm::dvec3* N, glm::dvec3* Q_next, glm::dvec3* N_next, int num_toUpper) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	for (int i = 0; i < num_toUpper; ++i) {
		if (i >= num_toUpper)
			return;
		glm::ivec2 upper = toUpper[i];
		glm::dvec3 q0 = Q[upper[0]];
		glm::dvec3 n0 = N[upper[0]];

		glm::dvec3 q, q1, n1, value1, value2;
		if (upper[1] != -1) {
			q1 = Q[upper[1]];
			n1 = N[upper[1]];
			compat_orientation_extrinsic_4(q0, n0, q1, n1, value1, value2);
			q = value1 + value2;
		}
		else {
			q = q0;
		}
		glm::dvec3 n = N_next[i];
		q -= glm::dot(n, q) * n;

		double len = q.x * q.x + q.y * q.y + q.z * q.z;
		if (len > 2.93873587705571876e-39f)
			q /= sqrt(len);
		Q_next[i] = q;
//	}
}


__global__ 
void cudaUpdatePosition(int* phase, int num_phases, glm::dvec3* N, glm::dvec3* Q, Link* adj, int* adjOffset, int num_adj, glm::dvec3* V, glm::dvec3* O, double scale) {
	int pi = blockIdx.x * blockDim.x + threadIdx.x;

//	for (int pi = 0; pi < num_phases; ++pi) {
	if (pi >= num_phases)
		return;
		int i = phase[pi];
		glm::dvec3 n_i = N[i], v_i = V[i];
		glm::dvec3 q_i = Q[i];
		glm::dvec3 sum = O[i];
		double weight_sum = 0.0f;

		for (int l = adjOffset[i]; l < adjOffset[i + 1]; ++l) {
			Link link = adj[l];
			int j = link.id;
			const double weight = link.weight;
			if (weight == 0)
				continue;

			glm::dvec3 n_j = N[j], v_j = V[j];
			glm::dvec3 q_j = Q[j], o_j = O[j];
			glm::dvec3 v1, v2;
			compat_position_extrinsic_4(
				v_i, n_i, q_i, sum, v_j, n_j, q_j, o_j, scale, v1, v2);

			sum = v1*weight_sum +v2*weight;
			weight_sum += weight;
			if (weight_sum > 2.93873587705571876e-39f)
				sum /= weight_sum;
			sum -= glm::dot(n_i, sum - v_i)*n_i;
		}

		if (weight_sum > 0) {
			O[i] = position_round_4(sum, q_i, n_i, v_i, scale);
		}
//	}
}

__global__
void cudaPropagatePositionUpper(glm::dvec3* srcField, glm::ivec2* toUpper, glm::dvec3* N, glm::dvec3* V, glm::dvec3* destField, int num_position) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	for (int i = 0; i < num_position; ++i) {
	if (i >= num_position)
		return;
		for (int k = 0; k < 2; ++k) {
			int dest = toUpper[i][k];
			if (dest == -1)
				continue;
			glm::dvec3 o = srcField[i], n = N[dest], v = V[dest];
			o -= n * glm::dot(n, o - v);
			destField[dest] = o;
		}
//	}
}


namespace qflow {

void UpdateOrientation(int* phase, int num_phases, glm::dvec3* N, glm::dvec3* Q, Link* adj, int* adjOffset, int num_adj) {
	cudaUpdateOrientation << <(num_phases + 255) / 256, 256 >> >(phase, num_phases, N, Q, adj, adjOffset, num_adj);
//	cudaUpdateOrientation(phase, num_phases, N, Q, adj, adjOffset, num_adj);
}

void PropagateOrientationUpper(glm::dvec3* srcField, int num_orientation, glm::ivec2* toUpper, glm::dvec3* N, glm::dvec3* destField) {
	cudaPropagateOrientationUpper << <(num_orientation + 255) / 256, 256 >> >(srcField, toUpper, N, destField, num_orientation);
//	cudaPropagateOrientationUpper(srcField, toUpper, N, destField, num_orientation);
}

void PropagateOrientationLower(glm::ivec2* toUpper, glm::dvec3* Q, glm::dvec3* N, glm::dvec3* Q_next, glm::dvec3* N_next, int num_toUpper) {
	cudaPropagateOrientationLower << <(num_toUpper + 255) / 256, 256 >> >(toUpper, Q, N, Q_next, N_next, num_toUpper);
//	cudaPropagateOrientationLower(toUpper, Q, N, Q_next, N_next, num_toUpper);
}


void UpdatePosition(int* phase, int num_phases, glm::dvec3* N, glm::dvec3* Q, Link* adj, int* adjOffset, int num_adj, glm::dvec3* V, glm::dvec3* O, double scale) {
	cudaUpdatePosition << <(num_phases + 255) / 256, 256 >> >(phase, num_phases, N, Q, adj, adjOffset, num_adj, V, O, scale);
//	cudaUpdatePosition(phase, num_phases, N, Q, adj, adjOffset, num_adj, V, O, scale);
}

void PropagatePositionUpper(glm::dvec3* srcField, int num_position, glm::ivec2* toUpper, glm::dvec3* N, glm::dvec3* V, glm::dvec3* destField) {
	cudaPropagatePositionUpper << <(num_position + 255) / 256, 256 >> >(srcField, toUpper, N, V, destField, num_position);
//	cudaPropagatePositionUpper(srcField, toUpper, N, V, destField, num_position);
}

} // namespace qflow

// GPU sort for DownsampleGraph edge entries using Thrust
#include <thrust/device_vector.h>
#include <thrust/sort.h>

struct GpuSortEntry {
    int i, j;
    double order;
};

struct GpuSortComp {
    __host__ __device__
    bool operator()(const GpuSortEntry& a, const GpuSortEntry& b) const {
        return a.order > b.order;  // descending by order (matches Entry::operator<)
    }
};

extern "C" void cuda_sort_entries(void* data, int count) {
    GpuSortEntry* entries = (GpuSortEntry*)data;
    thrust::device_vector<GpuSortEntry> d_entries(entries, entries + count);
    thrust::sort(d_entries.begin(), d_entries.end(), GpuSortComp());
    thrust::copy(d_entries.begin(), d_entries.end(), entries);
}

// GPU IC0-Preconditioned Conjugate Gradient solver
#include <cusparse.h>

// Custom kernels for CG vector operations
__global__ void pcg_dot_partial(const double* a, const double* b, double* partial, int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? a[i] * b[i] : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

__global__ void pcg_dot_final(const double* partial, double* result, int nBlocks) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    sdata[tid] = (tid < nBlocks) ? partial[tid] : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) result[0] = sdata[0];
}

__global__ void pcg_update_xr(double* x, double* r, const double* p,
                               const double* Ap, double alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += alpha * p[i];
    r[i] -= alpha * Ap[i];
}

__global__ void pcg_update_p(double* p, const double* z, double beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i] = z[i] + beta * p[i];
}

__global__ void pcg_copy(double* dst, const double* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i];
}

static double gpu_dot(const double* a, const double* b, int n,
                      double* d_partial, double* d_result, double* h_result) {
    int blockSize = 256;
    int nBlocks = (n + blockSize - 1) / blockSize;
    pcg_dot_partial<<<nBlocks, blockSize, blockSize * sizeof(double)>>>(a, b, d_partial, n);
    int finalBlock = 1;
    while (finalBlock < nBlocks) finalBlock <<= 1;
    if (finalBlock > 1024) finalBlock = 1024;
    pcg_dot_final<<<1, finalBlock, finalBlock * sizeof(double)>>>(d_partial, d_result, nBlocks);
    cudaMemcpy(h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    return *h_result;
}

// SpMV: y = A*x using custom kernel
__global__ void pcg_spmv(const int* rowPtr, const int* colInd,
                          const double* val, const double* x, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double sum = 0.0;
    for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
        sum += val[j] * x[colInd[j]];
    }
    y[i] = sum;
}

// Persistent PCG solver context — reuses GPU buffers and IC0 analysis across solves
struct PcgContext {
    int n, nnz, nBlocks;
    int *d_rowPtr, *d_colInd;
    double *d_val, *d_ic_val, *d_b, *d_x;
    double *d_r, *d_z, *d_p, *d_Ap, *d_tmp;
    double *d_partial, *d_result;
    cusparseHandle_t spHandle;
    cusparseMatDescr_t descrA;
    csric02Info_t ic0Info;
    void* d_ic0Buf;
    int ic0BufSize;
    bool ic0_ok;
    // SpSV descriptors for triangular solves
    cusparseSpMatDescr_t matL;
    cusparseDnVecDescr_t vecIn, vecTmp, vecOut;
    cusparseSpSVDescr_t spsvDescrL, spsvDescrLt;
    void *d_bufSvL, *d_bufSvLt;
    bool initialized;
};

extern "C" void* cuda_pcg_init(
    int n, int nnz,
    const int* h_csrRowPtr,
    const int* h_csrColInd,
    const double* h_csrVal  // initial values for IC0 analysis
) {
    PcgContext* ctx = new PcgContext();
    ctx->n = n;
    ctx->nnz = nnz;
    ctx->nBlocks = (n + 255) / 256;
    ctx->ic0_ok = false;
    ctx->matL = nullptr;
    ctx->vecIn = ctx->vecTmp = ctx->vecOut = nullptr;
    ctx->spsvDescrL = ctx->spsvDescrLt = nullptr;
    ctx->d_bufSvL = ctx->d_bufSvLt = nullptr;
    ctx->initialized = true;

    int nBlocks = ctx->nBlocks;

    // Allocate all GPU buffers once
    cudaMalloc(&ctx->d_rowPtr, (n + 1) * sizeof(int));
    cudaMalloc(&ctx->d_colInd, nnz * sizeof(int));
    cudaMalloc(&ctx->d_val, nnz * sizeof(double));
    cudaMalloc(&ctx->d_ic_val, nnz * sizeof(double));
    cudaMalloc(&ctx->d_b, n * sizeof(double));
    cudaMalloc(&ctx->d_x, n * sizeof(double));
    cudaMalloc(&ctx->d_r, n * sizeof(double));
    cudaMalloc(&ctx->d_z, n * sizeof(double));
    cudaMalloc(&ctx->d_p, n * sizeof(double));
    cudaMalloc(&ctx->d_Ap, n * sizeof(double));
    cudaMalloc(&ctx->d_tmp, n * sizeof(double));
    cudaMalloc(&ctx->d_partial, nBlocks * sizeof(double));
    cudaMalloc(&ctx->d_result, sizeof(double));

    // Upload structure (once — pattern doesn't change)
    cudaMemcpy(ctx->d_rowPtr, h_csrRowPtr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_colInd, h_csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_val, h_csrVal, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_ic_val, h_csrVal, nnz * sizeof(double), cudaMemcpyHostToDevice);

    // cuSPARSE handle + descriptor
    cusparseCreate(&ctx->spHandle);
    cusparseCreateMatDescr(&ctx->descrA);
    cusparseSetMatType(ctx->descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(ctx->descrA, CUSPARSE_INDEX_BASE_ZERO);

    // IC0 symbolic analysis (once — depends only on sparsity pattern)
#pragma nv_diag_suppress 1444
    cusparseCreateCsric02Info(&ctx->ic0Info);
    cusparseDcsric02_bufferSize(ctx->spHandle, n, nnz, ctx->descrA,
        ctx->d_ic_val, ctx->d_rowPtr, ctx->d_colInd, ctx->ic0Info, &ctx->ic0BufSize);
    cudaMalloc(&ctx->d_ic0Buf, ctx->ic0BufSize);
    cusparseDcsric02_analysis(ctx->spHandle, n, nnz, ctx->descrA,
        ctx->d_ic_val, ctx->d_rowPtr, ctx->d_colInd, ctx->ic0Info,
        CUSPARSE_SOLVE_POLICY_USE_LEVEL, ctx->d_ic0Buf);

    int structural_zero;
    cusparseStatus_t st = cusparseXcsric02_zeroPivot(ctx->spHandle, ctx->ic0Info, &structural_zero);
    ctx->ic0_ok = (st != CUSPARSE_STATUS_ZERO_PIVOT);
    if (!ctx->ic0_ok) {
        printf("[PCG-IC0] structural zero at row %d, no IC0\n", structural_zero);
    }

    // Do initial numeric IC0 factorization
    if (ctx->ic0_ok) {
        cusparseDcsric02(ctx->spHandle, n, nnz, ctx->descrA,
            ctx->d_ic_val, ctx->d_rowPtr, ctx->d_colInd, ctx->ic0Info,
            CUSPARSE_SOLVE_POLICY_USE_LEVEL, ctx->d_ic0Buf);
        cusparseStatus_t ns = cusparseXcsric02_zeroPivot(ctx->spHandle, ctx->ic0Info, &structural_zero);
        if (ns == CUSPARSE_STATUS_ZERO_PIVOT) {
            printf("[PCG-IC0] numerical zero at row %d, no IC0\n", structural_zero);
            ctx->ic0_ok = false;
        }
    }

    // Setup SpSV for triangular solves
    if (ctx->ic0_ok) {
        cusparseCreateCsr(&ctx->matL, n, n, nnz,
            ctx->d_rowPtr, ctx->d_colInd, ctx->d_ic_val,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        cusparseFillMode_t fillMode = CUSPARSE_FILL_MODE_LOWER;
        cusparseDiagType_t diagType = CUSPARSE_DIAG_TYPE_NON_UNIT;
        cusparseSpMatSetAttribute(ctx->matL, CUSPARSE_SPMAT_FILL_MODE, &fillMode, sizeof(fillMode));
        cusparseSpMatSetAttribute(ctx->matL, CUSPARSE_SPMAT_DIAG_TYPE, &diagType, sizeof(diagType));

        cusparseCreateDnVec(&ctx->vecIn, n, ctx->d_r, CUDA_R_64F);
        cusparseCreateDnVec(&ctx->vecTmp, n, ctx->d_tmp, CUDA_R_64F);
        cusparseCreateDnVec(&ctx->vecOut, n, ctx->d_z, CUDA_R_64F);

        double one = 1.0;
        cusparseSpSV_createDescr(&ctx->spsvDescrL);
        size_t bufSizeL;
        cusparseSpSV_bufferSize(ctx->spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, ctx->matL, ctx->vecIn, ctx->vecTmp, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, ctx->spsvDescrL, &bufSizeL);
        cudaMalloc(&ctx->d_bufSvL, bufSizeL);
        cusparseSpSV_analysis(ctx->spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, ctx->matL, ctx->vecIn, ctx->vecTmp, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, ctx->spsvDescrL, ctx->d_bufSvL);

        cusparseSpSV_createDescr(&ctx->spsvDescrLt);
        size_t bufSizeLt;
        cusparseSpSV_bufferSize(ctx->spHandle, CUSPARSE_OPERATION_TRANSPOSE,
            &one, ctx->matL, ctx->vecTmp, ctx->vecOut, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, ctx->spsvDescrLt, &bufSizeLt);
        cudaMalloc(&ctx->d_bufSvLt, bufSizeLt);
        cusparseSpSV_analysis(ctx->spHandle, CUSPARSE_OPERATION_TRANSPOSE,
            &one, ctx->matL, ctx->vecTmp, ctx->vecOut, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, ctx->spsvDescrLt, ctx->d_bufSvLt);
    }

    printf("[PCG-IC0] init: n=%d nnz=%d ic0=%s\n", n, nnz, ctx->ic0_ok ? "yes" : "no");
    return (void*)ctx;
}

extern "C" int cuda_pcg_solve(
    void* context,
    const double* h_csrVal,  // new values (same pattern)
    const double* h_b,
    double* h_x
) {
    PcgContext* ctx = (PcgContext*)context;
    int n = ctx->n, nnz = ctx->nnz;
    const int MAX_ITER = 1000;
    const double TOL = 1e-6;
    const int BS = 256;
    int nBlocks = ctx->nBlocks;

    // Upload only values and RHS (pattern already on GPU)
    cudaMemcpy(ctx->d_val, h_csrVal, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_b, h_b, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(ctx->d_x, 0, n * sizeof(double));

    // Re-factorize IC0 with new values (reuse analysis)
    if (ctx->ic0_ok) {
        cudaMemcpy(ctx->d_ic_val, h_csrVal, nnz * sizeof(double), cudaMemcpyHostToDevice);
#pragma nv_diag_suppress 1444
        cusparseDcsric02(ctx->spHandle, n, nnz, ctx->descrA,
            ctx->d_ic_val, ctx->d_rowPtr, ctx->d_colInd, ctx->ic0Info,
            CUSPARSE_SOLVE_POLICY_USE_LEVEL, ctx->d_ic0Buf);
        // Re-analyze SpSV with new factored values
        double one = 1.0;
        cusparseSpSV_destroyDescr(ctx->spsvDescrL);
        cusparseSpSV_createDescr(&ctx->spsvDescrL);
        cusparseSpSV_analysis(ctx->spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, ctx->matL, ctx->vecIn, ctx->vecTmp, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, ctx->spsvDescrL, ctx->d_bufSvL);
        cusparseSpSV_destroyDescr(ctx->spsvDescrLt);
        cusparseSpSV_createDescr(&ctx->spsvDescrLt);
        cusparseSpSV_analysis(ctx->spHandle, CUSPARSE_OPERATION_TRANSPOSE,
            &one, ctx->matL, ctx->vecTmp, ctx->vecOut, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, ctx->spsvDescrLt, ctx->d_bufSvLt);
    }

    // CG iteration
    pcg_copy<<<nBlocks, BS>>>(ctx->d_r, ctx->d_b, n);

    if (ctx->ic0_ok) {
        double one = 1.0;
        cusparseDnVecSetValues(ctx->vecIn, ctx->d_r);
        cusparseDnVecSetValues(ctx->vecTmp, ctx->d_tmp);
        cusparseDnVecSetValues(ctx->vecOut, ctx->d_z);
        cusparseSpSV_solve(ctx->spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, ctx->matL, ctx->vecIn, ctx->vecTmp, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, ctx->spsvDescrL);
        cusparseSpSV_solve(ctx->spHandle, CUSPARSE_OPERATION_TRANSPOSE,
            &one, ctx->matL, ctx->vecTmp, ctx->vecOut, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, ctx->spsvDescrLt);
    } else {
        pcg_copy<<<nBlocks, BS>>>(ctx->d_z, ctx->d_r, n);
    }

    pcg_copy<<<nBlocks, BS>>>(ctx->d_p, ctx->d_z, n);

    double h_result_val;
    double rz_old = gpu_dot(ctx->d_r, ctx->d_z, n, ctx->d_partial, ctx->d_result, &h_result_val);
    double r0_norm = gpu_dot(ctx->d_r, ctx->d_r, n, ctx->d_partial, ctx->d_result, &h_result_val);
    double tol_sq = TOL * TOL * r0_norm;

    int iter;
    for (iter = 0; iter < MAX_ITER; iter++) {
        pcg_spmv<<<nBlocks, BS>>>(ctx->d_rowPtr, ctx->d_colInd, ctx->d_val, ctx->d_p, ctx->d_Ap, n);

        double pAp = gpu_dot(ctx->d_p, ctx->d_Ap, n, ctx->d_partial, ctx->d_result, &h_result_val);
        if (fabs(pAp) < 1e-30) break;
        double alpha = rz_old / pAp;

        pcg_update_xr<<<nBlocks, BS>>>(ctx->d_x, ctx->d_r, ctx->d_p, ctx->d_Ap, alpha, n);

        if ((iter & 15) == 0) {
            double rr = gpu_dot(ctx->d_r, ctx->d_r, n, ctx->d_partial, ctx->d_result, &h_result_val);
            if (rr < tol_sq) { iter++; break; }
        }

        if (ctx->ic0_ok) {
            double one = 1.0;
            cusparseDnVecSetValues(ctx->vecIn, ctx->d_r);
            cusparseDnVecSetValues(ctx->vecTmp, ctx->d_tmp);
            cusparseDnVecSetValues(ctx->vecOut, ctx->d_z);
            cusparseSpSV_solve(ctx->spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one, ctx->matL, ctx->vecIn, ctx->vecTmp, CUDA_R_64F,
                CUSPARSE_SPSV_ALG_DEFAULT, ctx->spsvDescrL);
            cusparseSpSV_solve(ctx->spHandle, CUSPARSE_OPERATION_TRANSPOSE,
                &one, ctx->matL, ctx->vecTmp, ctx->vecOut, CUDA_R_64F,
                CUSPARSE_SPSV_ALG_DEFAULT, ctx->spsvDescrLt);
        } else {
            pcg_copy<<<nBlocks, BS>>>(ctx->d_z, ctx->d_r, n);
        }

        double rz_new = gpu_dot(ctx->d_r, ctx->d_z, n, ctx->d_partial, ctx->d_result, &h_result_val);
        double beta = rz_new / rz_old;
        rz_old = rz_new;
        pcg_update_p<<<nBlocks, BS>>>(ctx->d_p, ctx->d_z, beta, n);
    }

    printf("[PCG-IC0] iters=%d\n", iter);
    cudaMemcpy(h_x, ctx->d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
    return (iter < MAX_ITER) ? 0 : -1;
}

extern "C" void cuda_pcg_destroy(void* context) {
    PcgContext* ctx = (PcgContext*)context;
    if (!ctx) return;
    if (ctx->spsvDescrL) cusparseSpSV_destroyDescr(ctx->spsvDescrL);
    if (ctx->spsvDescrLt) cusparseSpSV_destroyDescr(ctx->spsvDescrLt);
    if (ctx->matL) cusparseDestroySpMat(ctx->matL);
    if (ctx->vecIn) cusparseDestroyDnVec(ctx->vecIn);
    if (ctx->vecTmp) cusparseDestroyDnVec(ctx->vecTmp);
    if (ctx->vecOut) cusparseDestroyDnVec(ctx->vecOut);
    if (ctx->d_bufSvL) cudaFree(ctx->d_bufSvL);
    if (ctx->d_bufSvLt) cudaFree(ctx->d_bufSvLt);
#pragma nv_diag_suppress 1444
    cusparseDestroyCsric02Info(ctx->ic0Info);
    cudaFree(ctx->d_ic0Buf);
    cusparseDestroyMatDescr(ctx->descrA);
    cusparseDestroy(ctx->spHandle);
    cudaFree(ctx->d_rowPtr); cudaFree(ctx->d_colInd);
    cudaFree(ctx->d_val); cudaFree(ctx->d_ic_val);
    cudaFree(ctx->d_b); cudaFree(ctx->d_x);
    cudaFree(ctx->d_r); cudaFree(ctx->d_z);
    cudaFree(ctx->d_p); cudaFree(ctx->d_Ap); cudaFree(ctx->d_tmp);
    cudaFree(ctx->d_partial); cudaFree(ctx->d_result);
    delete ctx;
}

// One-shot API for callers that don't need persistence (optimize_positions_fixed, optimize_scale)
extern "C" int cuda_cholesky_solve(
    int n, int nnz,
    const int* h_csrRowPtr,
    const int* h_csrColInd,
    const double* h_csrVal,
    const double* h_b,
    double* h_x
) {
    void* ctx = cuda_pcg_init(n, nnz, h_csrRowPtr, h_csrColInd, h_csrVal);
    int ret = cuda_pcg_solve(ctx, h_csrVal, h_b, h_x);
    cuda_pcg_destroy(ctx);
    return ret;
}
