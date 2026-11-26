// graph.h -- two fixed CSR graph inputs, and the CPU reference BFS.
// Do NOT change this file 

#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <queue>
#include <utility>
#include <vector>

struct CSRGraph {
    int V = 0;                  // number of vertices
    long long E = 0;            // number of directed edges (each undirected edge stored twice)
    std::vector<int> rowPtr;    // size V+1: neighbors of v are colIdx[rowPtr[v] .. rowPtr[v+1]-1]
    std::vector<int> colIdx;    // size E
};

// Build an undirected CSR graph from an edge list
static CSRGraph buildCSR(int V, std::vector<std::pair<int, int>>& edges) {
    std::vector<std::pair<int, int>> both;
    both.reserve(edges.size() * 2);
    for (auto& e : edges)
        if (e.first != e.second) { both.push_back(e); both.push_back({e.second, e.first}); }
    std::sort(both.begin(), both.end());
    both.erase(std::unique(both.begin(), both.end()), both.end());

    CSRGraph g;
    g.V = V;
    g.E = (long long)both.size();
    g.rowPtr.assign(V + 1, 0);
    g.colIdx.resize(both.size());
    for (auto& e : both) g.rowPtr[e.first + 1]++;
    for (int v = 0; v < V; ++v) g.rowPtr[v + 1] += g.rowPtr[v];
    for (size_t i = 0; i < both.size(); ++i) g.colIdx[i] = both[i].second;
    return g;
}

// Input 1:  a road network or a mesh.
static CSRGraph makeGrid(int W, int H) {
    std::vector<std::pair<int, int>> edges;
    edges.reserve(2LL * W * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            int v = y * W + x;
            if (x + 1 < W) edges.push_back({v, v + 1});
            if (y + 1 < H) edges.push_back({v, v + W});
        }
    return buildCSR(W * H, edges);
}

// Input 2: an R-MAT (social networks) graph with Graph500 parameters, skewed degrees, small diameter
static CSRGraph makeRMAT(int scale, int edgeFactor, uint64_t seed) {
    const int V = 1 << scale;
    const double a = 0.57, b = 0.19, c = 0.19;      // d = 0.05
    uint64_t s = seed;
    auto rnd = [&s]() {                              // xorshift64*, uniform in [0,1)
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return (double)((s * 2685821657736338717ULL) >> 11) * (1.0 / 9007199254740992.0);
    };
    std::vector<std::pair<int, int>> edges;
    edges.reserve((size_t)V * edgeFactor);
    for (long long i = 0; i < (long long)V * edgeFactor; ++i) {
        int u = 0, v = 0;
        for (int bit = scale - 1; bit >= 0; --bit) {
            double r = rnd();
            if (r < a) {}
            else if (r < a + b) v |= 1 << bit;
            else if (r < a + b + c) u |= 1 << bit;
            else { u |= 1 << bit; v |= 1 << bit; }
        }
        edges.push_back({u, v});
    }
    return buildCSR(V, edges);
}

// Your GPU kernel should complete with this CPU kernel
// Use it to check correctness
static int bfsCPU(const CSRGraph& g, int src, std::vector<int>& level) {
    level.assign(g.V, -1);
    std::queue<int> q;
    level[src] = 0;
    q.push(src);
    int maxLevel = 0;
    while (!q.empty()) {
        int v = q.front(); q.pop();
        for (int i = g.rowPtr[v]; i < g.rowPtr[v + 1]; ++i) {
            int u = g.colIdx[i];
            if (level[u] == -1) {
                level[u] = level[v] + 1;
                maxLevel = std::max(maxLevel, level[u]);
                q.push(u);
            }
        }
    }
    return maxLevel + 1;
}

static bool verify(const std::vector<int>& ref, const std::vector<int>& got) {
    for (size_t v = 0; v < ref.size(); ++v)
        if (ref[v] != got[v]) {
            printf("MISMATCH at vertex %zu: CPU level %d, GPU level %d\n", v, ref[v], got[v]);
            return false;
        }
    return true;
}

static CSRGraph loadInput(const char* name) {
    if (strcmp(name, "grid") == 0) return makeGrid(1024, 1024);      // 1,048,576 vertices
    if (strcmp(name, "rmat") == 0) return makeRMAT(18, 16, 2360);    // 262,144 vertices
    printf("unknown input '%s' (use grid or rmat)\n", name);
    return CSRGraph{};
}

static double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
