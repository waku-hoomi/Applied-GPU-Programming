// You are free to modify this file
// but do not change the function signatures of bfsLevelKernel or bfsGPU.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "graph.h"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_),        \
                   __FILE__, __LINE__);                                         \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// TODO (1): one level of a level-synchronous BFS.
//
//   One thread per vertex v. If v is on the current frontier, visit every neighbor u of v:
//       neighbors of v are colIdx[rowPtr[v]] ... colIdx[rowPtr[v+1] - 1]
//   If u is unvisited (level[u] == -1), set level[u] = curLevel + 1 so that u will be on the next frontier
//  If any vertex was newly visited, set *changed = 1 so that the host (in TODO(2)) knows to continue to the next level.
// ---------------------------------------------------------------------------
__global__ void bfsLevelKernel(const int* rowPtr, const int* colIdx, int* level,
                               int V, int curLevel, int* changed) {
    // Map each thread to a unique vertex ID v
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;

    // Only vertices on the current BFS frontier expand their neighbors
    if (level[v] == curLevel) {
        int startEdge = rowPtr[v];
        int endEdge = rowPtr[v + 1];
        for (int e = startEdge; e < endEdge; ++e) {
            int u = colIdx[e];
            // Idempotent write: all threads discovering u in this level write the exact same value (curLevel + 1)
            if (level[u] == -1) {
                level[u] = curLevel + 1;
                *changed = 1;
            }
        }
    }
}

float bfsGPU(const CSRGraph& g, int src, std::vector<int>& level, int& numLevels) {
    int *dRowPtr, *dColIdx, *dLevel, *dChanged;
    CUDA_CHECK(cudaMalloc(&dRowPtr, (g.V + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dColIdx, g.E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dLevel, g.V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dChanged, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dRowPtr, g.rowPtr.data(), (g.V + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dColIdx, g.colIdx.data(), g.E * sizeof(int), cudaMemcpyHostToDevice));

    // All vertices unvisited (-1), except the source (level 0).
    level.assign(g.V, -1);
    level[src] = 0;
    CUDA_CHECK(cudaMemcpy(dLevel, level.data(), g.V * sizeof(int), cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // TODO (2): the level loop on the host.
    //   curLevel = 0
    //   repeat:
    //     set *dChanged = 0                      (cudaMemset or cudaMemcpy)
    //     launch bfsLevelKernel with enough blocks of 256 threads to cover V
    //     copy dChanged back to the host
    //     curLevel++
    //   until nothing changed
    //   numLevels = number of levels that contain at least one vertex
    // -----------------------------------------------------------------------
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (g.V + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    int curLevel = 0;
    int hChanged = 0;

    // Record start time of the GPU level-synchronous kernel loop
    CUDA_CHECK(cudaEventRecord(startEvent));
    do {
        hChanged = 0;
        CUDA_CHECK(cudaMemset(dChanged, 0, sizeof(int)));

        bfsLevelKernel<<<blocksPerGrid, threadsPerBlock>>>(
            dRowPtr, dColIdx, dLevel, g.V, curLevel, dChanged);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(&hChanged, dChanged, sizeof(int), cudaMemcpyDeviceToHost));
        if (hChanged) {
            curLevel++;
        }
    } while (hChanged != 0);

    // Record end time after all frontier levels have completed
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    // Levels 0 .. curLevel each contain at least one vertex
    numLevels = curLevel + 1;

    // -----------------------------------------------------------------------
    // TODO (3):
    // copy dLevel back into level.
    // print the timing information and return the elapsed time
    // -----------------------------------------------------------------------
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    CUDA_CHECK(cudaMemcpy(level.data(), dLevel, g.V * sizeof(int), cudaMemcpyDeviceToHost));

    printf("GPU BFS kernel loop elapsed: %.3f ms (%d kernel launches)\n", ms, numLevels);

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    cudaFree(dRowPtr);
    cudaFree(dColIdx);
    cudaFree(dLevel);
    cudaFree(dChanged);
    return ms;
}

int main(int argc, char** argv) {
    const char* input = argc > 1 ? argv[1] : "grid";
    // Optional second argument: number of runs to compute median time (default = 1)
    const int runs = argc > 2 ? std::max(1, std::atoi(argv[2])) : 1;

    CSRGraph g = loadInput(input);
    if (g.V == 0) return 1;
    const int src = 0;
    printf("Input %s: %d vertices, %lld directed edges (avg degree %.1f, max degree %d)\n",
           input, g.V, g.E, (double)g.E / g.V,
           [&] { int m = 0; for (int v = 0; v < g.V; ++v) m = std::max(m, g.rowPtr[v + 1] - g.rowPtr[v]); return m; }());

    std::vector<int> refLevel, gpuLevel;
    std::vector<double> cpuTimes, gpuTimes;
    int cpuLevels = 0, gpuLevels = 0;
    bool ok = true;

    for (int r = 0; r < runs; ++r) {
        double c0 = nowMs();
        cpuLevels = bfsCPU(g, src, refLevel);
        double cpuMs = nowMs() - c0;

        float gpuMs = bfsGPU(g, src, gpuLevel, gpuLevels);

        cpuTimes.push_back(cpuMs);
        gpuTimes.push_back(static_cast<double>(gpuMs));
        ok = ok && verify(refLevel, gpuLevel);
    }

    // Compute median times across runs
    std::sort(cpuTimes.begin(), cpuTimes.end());
    std::sort(gpuTimes.begin(), gpuTimes.end());
    double medCpuMs = (runs % 2 == 1)
                          ? cpuTimes[runs / 2]
                          : 0.5 * (cpuTimes[runs / 2 - 1] + cpuTimes[runs / 2]);
    double medGpuMs = (runs % 2 == 1)
                          ? gpuTimes[runs / 2]
                          : 0.5 * (gpuTimes[runs / 2 - 1] + gpuTimes[runs / 2]);

    long long reached = 0;
    for (int l : refLevel) reached += (l >= 0);
    printf("CPU BFS: %8.2f ms, %d levels, %lld vertices reached\n", medCpuMs, cpuLevels, reached);
    printf("GPU BFS: %8.2f ms, %d levels\n", medGpuMs, gpuLevels);

    printf("Result: %s", ok ? "PASS" : "FAIL");
    if (ok) printf("   speedup %.2fx, GPU %.1f MTEPS", medCpuMs / medGpuMs, g.E / (medGpuMs * 1e3));
    printf("\n");
    if (runs > 1) {
        printf("(Median over %d runs: CPU = %.3f ms, GPU = %.3f ms)\n", runs, medCpuMs, medGpuMs);
    }
    return ok ? 0 : 1;
}
