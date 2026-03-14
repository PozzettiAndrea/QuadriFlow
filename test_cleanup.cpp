// Standalone test for flow cleanup strategies
// Loads /tmp/qf-dinic-flow.bin and tries different augmentation approaches
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <vector>
#include <queue>
#include <chrono>

int main() {
    FILE* fp = fopen("/tmp/qf-dinic-flow.bin", "rb");
    if (!fp) { printf("Can't open flow state\n"); return 1; }

    int num_nodes, num_edges, source, sink, initial_flow;
    fread(&num_nodes, sizeof(int), 1, fp);
    fread(&num_edges, sizeof(int), 1, fp);
    fread(&source, sizeof(int), 1, fp);
    fread(&sink, sizeof(int), 1, fp);
    fread(&initial_flow, sizeof(int), 1, fp);

    std::vector<int> flow(num_edges), nindex(num_nodes+1), nlist(num_edges),
                     cap(num_edges), rnindex(num_nodes+1), rnlist(num_edges), retoe(num_edges);

    fread(flow.data(), sizeof(int), num_edges, fp);
    fread(nindex.data(), sizeof(int), num_nodes+1, fp);
    fread(nlist.data(), sizeof(int), num_edges, fp);
    fread(cap.data(), sizeof(int), num_edges, fp);
    fread(rnindex.data(), sizeof(int), num_nodes+1, fp);
    fread(rnlist.data(), sizeof(int), num_edges, fp);
    fread(retoe.data(), sizeof(int), num_edges, fp);
    fclose(fp);

    printf("Loaded: %d nodes, %d edges, source=%d, sink=%d, initial_flow=%d\n",
           num_nodes, num_edges, source, sink, initial_flow);

    // ---- Strategy 2: CPU EK with vector BFS + iteration cap ----
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<int> par(num_nodes);
    std::vector<int> par_e(num_nodes);
    std::vector<int> par_d(num_nodes);
    std::vector<int> bfs_q;
    bfs_q.reserve(num_nodes);

    int total_flow = initial_flow;
    int augs = 0;
    int max_augs = 2000;  // safety cap

    while (augs < max_augs) {
        // BFS using vector (no malloc)
        std::fill(par.begin(), par.end(), -1);
        par[source] = source;
        bfs_q.clear();
        bfs_q.push_back(source);

        bool found = false;
        for (int qi = 0; qi < (int)bfs_q.size() && !found; qi++) {
            int u = bfs_q[qi];
            for (int e = nindex[u]; e < nindex[u+1]; e++) {
                int v = nlist[e];
                if (par[v] != -1) continue;
                if (cap[e] - flow[e] <= 0) continue;
                par[v] = u; par_e[v] = e; par_d[v] = 1;
                bfs_q.push_back(v);
                if (v == sink) { found = true; break; }
            }
            if (found) break;
            for (int re = rnindex[u]; re < rnindex[u+1]; re++) {
                int v = rnlist[re];
                if (par[v] != -1) continue;
                int ef = retoe[re];
                if (flow[ef] <= 0) continue;
                par[v] = u; par_e[v] = ef; par_d[v] = -1;
                bfs_q.push_back(v);
                if (v == sink) { found = true; break; }
            }
        }

        if (!found) break;

        // Bottleneck
        int bn = INT_MAX;
        for (int v = sink; v != source; v = par[v]) {
            int e = par_e[v];
            int r = (par_d[v] == 1) ? (cap[e] - flow[e]) : flow[e];
            if (r < bn) bn = r;
        }

        // Push
        for (int v = sink; v != source; v = par[v]) {
            int e = par_e[v];
            if (par_d[v] == 1) flow[e] += bn; else flow[e] -= bn;
        }
        total_flow += bn;
        augs++;

        if (augs % 50 == 0) {
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t0).count();
            printf("  aug %d: flow=%d, BFS visited %d nodes, elapsed=%.3fs\n",
                   augs, total_flow, (int)bfs_q.size(), elapsed);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    printf("CPU EK cleanup: %d augmentations, final flow=%d, %.3f s\n",
           augs, total_flow, elapsed);
    printf("  (needed %d more flow from initial %d)\n", total_flow - initial_flow, initial_flow);

    return 0;
}
