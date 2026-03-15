#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_
#include "config.hpp"
#include "field-math.hpp"
#include "hierarchy.hpp"
#include <unordered_map>

struct PairHash {
    size_t operator()(const std::pair<int,int>& p) const {
        return std::hash<long long>()(((long long)p.first << 32) | (unsigned int)p.second);
    }
};

namespace qflow {

class Optimizer {
   public:
    Optimizer();
    static void optimize_orientations(Hierarchy& mRes);
    static void optimize_scale(Hierarchy& mRes, VectorXd& rho, int adaptive);
    static void optimize_positions(Hierarchy& mRes, int with_scale = 0);
    static void optimize_integer_constraints(Hierarchy& mRes, std::map<int, int>& singularities,
                                             bool use_minimum_cost_flow);
    static void optimize_positions_fixed(
        Hierarchy& mRes, std::vector<DEdge>& edge_values, std::vector<Vector2i>& edge_diff,
        std::set<int>& sharp_vertices,
        std::map<int, std::pair<Vector3d, Vector3d>>& sharp_constraints, int with_scale = 0);
    static void optimize_positions_sharp(
        Hierarchy& mRes, std::vector<DEdge>& edge_values, std::vector<Vector2i>& edge_diff,
        std::vector<int>& sharp_edges, std::set<int>& sharp_vertices,
        std::map<int, std::pair<Vector3d, Vector3d>>& sharp_constraints, int with_scale = 0);
    static void optimize_positions_dynamic(
        MatrixXi& F, MatrixXd& V, MatrixXd& N, MatrixXd& Q, std::vector<std::vector<int>>& Vset,
        std::vector<Vector3d>& O_compact, std::vector<Vector4i>& F_compact,
        std::vector<int>& V2E_compact, std::vector<int>& E2E_compact, double mScale,
        std::vector<Vector3d>& diffs, std::vector<int>& diff_count,
        std::unordered_map<std::pair<int, int>, int, PairHash>& o2e, std::vector<int>& sharp_o,
        std::map<int, std::pair<Vector3d, Vector3d>>& compact_sharp_constraints, int with_scale);
#ifdef WITH_CUDA
    static void optimize_orientations_cuda(Hierarchy& mRes);
    static void optimize_positions_cuda(Hierarchy& mRes);
#endif
};

#ifdef WITH_CUDA
extern void UpdateOrientation(int* phase, int num_phases, glm::dvec3* N, glm::dvec3* Q, Link* adj,
                              int* adjOffset, int num_adj);
extern void PropagateOrientationUpper(glm::dvec3* srcField, int num_orientation,
                                      glm::ivec2* toUpper, glm::dvec3* N, glm::dvec3* destField);
extern void PropagateOrientationLower(glm::ivec2* toUpper, glm::dvec3* Q, glm::dvec3* N,
                                      glm::dvec3* Q_next, glm::dvec3* N_next, int num_toUpper);

extern void UpdatePosition(int* phase, int num_phases, glm::dvec3* N, glm::dvec3* Q, Link* adj,
                           int* adjOffset, int num_adj, glm::dvec3* V, glm::dvec3* O,
                           double scale);
extern void PropagatePositionUpper(glm::dvec3* srcField, int num_position, glm::ivec2* toUpper,
                                   glm::dvec3* N, glm::dvec3* V, glm::dvec3* destField);

// Dynamic optimization GPU kernels
extern "C" void* cuda_dyn_init(
    const double* V_colmaj, const double* N_colmaj, int nV,
    const int* adj_ptr, const int* adj_list, int adj_nnz,
    const int* vset_ptr, const int* vset_list, int vset_nnz,
    const int* dedge_ptr, const int* dedge_list, int dedge_nnz,
    int nQ, int nDE);
extern "C" void cuda_dyn_find_nearest(
    void* context, double* h_O_compact, int* h_Vind, double* h_diffs, int iteration);
extern "C" void cuda_dyn_fill_csr_init(
    void* context, const int* h_edge_i, const int* h_edge_j, const int* h_edge_de,
    const int* h_edge_csr_pos, const int* h_fixed_dim,
    int nEdges, int dim, int csr_nnz);
extern "C" void cuda_dyn_fill_csr(
    void* context, const double* h_Q_compact, const double* h_N_compact,
    const double* h_V_compact, const double* h_diffs, const double* h_x,
    double* h_csr_val, double* h_rhs);
extern "C" void cuda_dyn_destroy(void* context);

#endif

} // namespace qflow

#endif
