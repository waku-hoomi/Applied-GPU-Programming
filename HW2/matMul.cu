#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef USE_FLOAT
using DataType = float;
constexpr const char *TYPE_NAME = "float";
constexpr double EPSILON = 1e-3;
#else
using DataType = double;
constexpr const char *TYPE_NAME = "double";
constexpr double EPSILON = 1e-6;
#endif

constexpr int BLOCK_SIZE = 16;
constexpr int TILE_K = 16;
constexpr int GPU_RUNS = 10;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                 \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

__global__ void gemm(const DataType *A, const DataType *B, DataType *C,
                     int M, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        DataType sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[static_cast<size_t>(row) * K + k] *
                   B[static_cast<size_t>(k) * N + col];
        }
        C[static_cast<size_t>(row) * N + col] = sum;
    }
}

__global__ void tiled_gemm(const DataType *A, const DataType *B, DataType *C,
                           int M, int K, int N, int tileX, int tileY) {
    extern __shared__ DataType shared[];
    DataType *tileA = shared;
    DataType *tileB = shared + tileY * TILE_K;

    int firstRow = blockIdx.y * tileY;
    int firstCol = blockIdx.x * tileX;
    int row = firstRow + threadIdx.y;
    int col = firstCol + threadIdx.x;
    int thread = threadIdx.y * tileX + threadIdx.x;
    int threads = tileX * tileY;

    DataType sum = 0;
    for (int firstK = 0; firstK < K; firstK += TILE_K) {
        for (int i = thread; i < tileY * TILE_K; i += threads) {
            int inputRow = firstRow + i / TILE_K;
            int k = firstK + i % TILE_K;
            tileA[i] = inputRow < M && k < K
                           ? A[static_cast<size_t>(inputRow) * K + k]
                           : DataType{0};
        }
        for (int i = thread; i < TILE_K * tileX; i += threads) {
            int k = firstK + i / tileX;
            int inputCol = firstCol + i % tileX;
            tileB[i] = k < K && inputCol < N
                           ? B[static_cast<size_t>(k) * N + inputCol]
                           : DataType{0};
        }
        __syncthreads();

        for (int k = 0; k < TILE_K; ++k) {
            sum += tileA[threadIdx.y * TILE_K + k] *
                   tileB[k * tileX + threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[static_cast<size_t>(row) * N + col] = sum;
    }
}

// Row-major traversal of B and C. Each C element still accumulates in K order.
void cpu_gemm(const DataType *A, const DataType *B, DataType *C,
              int M, int K, int N) {
    for (int row = 0; row < M; ++row) {
        DataType *output = C + static_cast<size_t>(row) * N;
        std::fill_n(output, N, DataType{0});
        for (int k = 0; k < K; ++k) {
            DataType a = A[static_cast<size_t>(row) * K + k];
            const DataType *input = B + static_cast<size_t>(k) * N;
            for (int col = 0; col < N; ++col) {
                output[col] += a * input[col];
            }
        }
    }
}

bool validate(const std::vector<DataType> &output,
              const std::vector<DataType> &reference, double &maxError) {
    size_t mismatches = 0;
    maxError = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        double expected = reference[i];
        double error = std::fabs(static_cast<double>(output[i]) - expected);
        maxError = std::isfinite(error) ? std::max(maxError, error) : HUGE_VAL;
        if (!(error <= EPSILON * (1.0 + std::fabs(expected)))) {
            ++mismatches;
        }
    }
    if (mismatches != 0) {
        fprintf(stderr, "%zu GPU output elements failed validation\n", mismatches);
    }
    return mismatches == 0;
}

struct Tile {
    int x;
    int y;
};

bool parse_dimension(const char *arg, int &value) {
    char *end;
    errno = 0;
    long parsed = strtol(arg, &end, 10);
    if (errno != 0 || end == arg || *end != '\0' || parsed < 1 ||
        parsed > INT_MAX - 32) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

int main(int argc, char **argv) {
    int M, K, N;
    bool csv = argc == 5 && strcmp(argv[4], "--csv") == 0;
    bool profile = argc == 5 && strcmp(argv[4], "--profile") == 0;
    if ((argc != 4 && !csv && !profile) ||
        !parse_dimension(argv[1], M) || !parse_dimension(argv[2], K) ||
        !parse_dimension(argv[3], N)) {
        fprintf(stderr, "Usage: %s M K N [--csv|--profile]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int device;
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    // The smallest Y tile defines the largest grid height.
    if ((M + 7) / 8 > properties.maxGridSize[1] ||
        (N + 7) / 8 > properties.maxGridSize[0]) {
        fprintf(stderr, "Matrix dimensions exceed the supported CUDA grid size\n");
        return EXIT_FAILURE;
    }

    std::vector<DataType> A(static_cast<size_t>(M) * K);
    std::vector<DataType> B(static_cast<size_t>(K) * N);
    std::vector<DataType> reference(static_cast<size_t>(M) * N);
    std::vector<DataType> output(reference.size());
    srand(42);
    for (DataType &a : A) a = static_cast<DataType>((rand() % 100) / 100.0);
    for (DataType &b : B) b = static_cast<DataType>((rand() % 100) / 100.0);

    auto start = std::chrono::steady_clock::now();
    cpu_gemm(A.data(), B.data(), reference.data(), M, K, N);
    double cpuMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    DataType *dA, *dB, *dC;
    size_t bytesA = A.size() * sizeof(DataType);
    size_t bytesB = B.size() * sizeof(DataType);
    size_t bytesC = output.size() * sizeof(DataType);
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytesB, cudaMemcpyHostToDevice));

    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));

    const Tile configurations[] = {{BLOCK_SIZE, BLOCK_SIZE}, {8, 8}, {16, 16}, {32, 8}};
    float gpuMs[4];
    bool allValid = true;
    if (!csv) {
        printf("A(%d x %d) * B(%d x %d) = C(%d x %d), %s\n", M, K, K, N, M, N, TYPE_NAME);
        printf("GPU: %s | CPU reference: %.6f ms\n", properties.name, cpuMs);
        if (profile) {
            puts("GPU timing: one launch under profiling");
        } else {
            printf("GPU timing: mean of %d launches after warm-up (CUDA events)\n", GPU_RUNS);
        }
    }
    for (int variant = 0; variant < 4; ++variant) {
        Tile tile = configurations[variant];
        dim3 block(tile.x, tile.y);
        dim3 grid((N + tile.x - 1) / tile.x, (M + tile.y - 1) / tile.y);
        size_t sharedBytes = variant == 0 ? 0 :
                             TILE_K * (tile.x + tile.y) * sizeof(DataType);

        auto launch = [&] {
            if (variant == 0) {
                gemm<<<grid, block>>>(dA, dB, dC, M, K, N);
            } else {
                tiled_gemm<<<grid, block, sharedBytes>>>(dA, dB, dC, M, K, N, tile.x, tile.y);
            }
        };

        if (!profile) {
            launch();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        // Seed C so an unwritten element fails validation.
        CUDA_CHECK(cudaMemset(dC, 0xff, bytesC));
        int runs = profile ? 1 : GPU_RUNS;
        CUDA_CHECK(cudaEventRecord(begin));
        for (int run = 0; run < runs; ++run) launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaEventElapsedTime(&gpuMs[variant], begin, end));
        gpuMs[variant] /= runs;

        CUDA_CHECK(cudaMemcpy(output.data(), dC, bytesC, cudaMemcpyDeviceToHost));
        double maxError;
        bool valid = validate(output, reference, maxError);
        allValid = allValid && valid;
        if (!csv || !valid) {
            printf("%-10s block %2d x %-2d | shared %4zu B | %10.6f ms | %s | max abs error %.3g\n",
                   variant == 0 ? "gemm" : "tiled_gemm", tile.x, tile.y, sharedBytes,
                   gpuMs[variant], valid ? "PASS" : "FAIL", maxError);
        }
    }

    if (csv && allValid) {
        printf("%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
               M, K, N, cpuMs, gpuMs[0], gpuMs[1], gpuMs[2], gpuMs[3]);
    }
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return allValid ? EXIT_SUCCESS : EXIT_FAILURE;
}
