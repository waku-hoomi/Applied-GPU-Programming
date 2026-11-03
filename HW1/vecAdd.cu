#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <sys/time.h>

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                    \
    }                                                    \
} while (0)

// 返回当前时间，单位为秒
double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, nullptr);

    return static_cast<double>(tp.tv_sec)
         + static_cast<double>(tp.tv_usec) * 1.e-6;
}

/* Our over-simplified CUDA kernel */
__global__ void add(
    const float* a,
    const float* b,
    float* c,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        c[i] = a[i] + b[i];
    }
} 

int main(int argc, char** argv)
{
    int N = 263149;

    if (argc > 1) {
        char* end = nullptr;
        long value = strtol(argv[1], &end, 10);

        if (end == argv[1] || *end != '\0' ||
            value < 5 || value > 67108864L) {
            fprintf(stderr,
                    "N must be an integer from 5 to 67108864\n");
            return EXIT_FAILURE;
        }

        N = static_cast<int>(value);
    }

    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    // Create input and output arrays on the CPU
    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> c(N, -1.0f);
    std::vector<float> c_cpu(N);

  
    for (int i = 0; i < N; ++i) {
        a[i] = 0.5f * i;
        b[i] = 0.25f * i;
    }
    // CPU calculation results
    for (int i = 0; i < N; ++i) {
    c_cpu[i] = a[i] + b[i];
    }
    

    // Allocate space for three vectors on the GPU
    float *d_a, *d_b, *d_c;

    CHECK(cudaMalloc((void**)&d_a, bytes));
    CHECK(cudaMalloc((void**)&d_b, bytes));
    CHECK(cudaMalloc((void**)&d_c, bytes));
    // Configure GPU threads
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // One warm-up, followed by 10 measured repetitions
    const int repeats = 10;

    double totalH2D = 0.0;
    double totalKernel = 0.0;
    double totalD2H = 0.0;

    for (int trial = -1; trial < repeats; ++trial) {
        // trial == -1 is the warm-up; its timing is discarded
        CHECK(cudaDeviceSynchronize());

        // 1. Host -> Device: copy both input vectors
        double start = cpuSecond();

        CHECK(cudaMemcpy(
            d_a, a.data(), bytes, cudaMemcpyHostToDevice));

        CHECK(cudaMemcpy(
            d_b, b.data(), bytes, cudaMemcpyHostToDevice));

        CHECK(cudaDeviceSynchronize());

        double h2d = cpuSecond() - start;

        // 2. Kernel: launch and wait for completion
        start = cpuSecond();

        add<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        double kernel = cpuSecond() - start;

        // 3. Device -> Host: copy the output vector
        start = cpuSecond();

        CHECK(cudaMemcpy(
            c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

        double d2h = cpuSecond() - start;

        // Accumulate only the measured repetitions
        if (trial >= 0) {
            totalH2D += h2d;
            totalKernel += kernel;
            totalD2H += d2h;
        }
    }

    // Average times, converted from seconds to milliseconds
    double h2d_ms = totalH2D / repeats * 1000.0;
    double kernel_ms = totalKernel / repeats * 1000.0;
    double d2h_ms = totalD2H / repeats * 1000.0;

    printf("\nAverage timing over %d repetitions:\n", repeats);
    printf("H2D = %.6f ms\n", h2d_ms);
    printf("Kernel = %.6f ms\n", kernel_ms);
    printf("D2H = %.6f ms\n", d2h_ms);

    // Used by the Python script to collect timing data
    printf("TIMING,%d,%.9f,%.9f,%.9f\n",
           N, h2d_ms, kernel_ms, d2h_ms);
 
    // Compare GPU and CPU results（Q1.4）
    double maxAbsError = 0.0;
    int exactMismatches = 0;
    int failedElements = 0;

    const double atol = 1e-6;
    const double rtol = 1e-6;

    for (int i = 0; i < N; ++i) {
        double ref = static_cast<double>(c_cpu[i]);
        double got = static_cast<double>(c[i]);

        if (got != ref) {
            ++exactMismatches;
        }

        if (!std::isfinite(got)) {
            ++failedElements;
            maxAbsError = INFINITY;
            continue;
        }

        double error = std::fabs(got - ref);

        if (error > maxAbsError) {
            maxAbsError = error;
        }

        if (error > atol + rtol * std::fabs(ref)) {
            ++failedElements;
        }
    }

    printf("\nCPU/GPU validation:\n");
    printf("Exact mismatches = %d\n", exactMismatches);
    printf("Max absolute error = %.9g\n", maxAbsError);
    printf("Failed elements = %d\n", failedElements);
    printf("Validation = %s\n",
          failedElements == 0 ? "PASS" : "FAIL");

    printf("N = %d\n", N);
    printf("Threads per block = %d\n", threadsPerBlock);
    printf("Blocks = %d\n", blocks);
    printf("Total launched threads = %d\n",
       blocks * threadsPerBlock);

    printf("\nFirst 5 results:\n");

    for (int i = 0; i < 5; ++i) {
        printf("c[%d] = %.2f + %.2f = %.2f\n",
               i, a[i], b[i], c[i]);
    }

    printf("\nLast result:\n");

    printf("c[%d] = %.2f + %.2f = %.2f\n",
           N - 1, a[N - 1], b[N - 1], c[N - 1]);

    // release GPU memory
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    return failedElements == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}