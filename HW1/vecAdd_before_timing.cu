#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <vector>
#include <cmath>

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                    \
    }                                                    \
} while (0)

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

int main()
{

    const int N = 263149;
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

    // Copy the two input vectors from CPU to GPU
    CHECK(cudaMemcpy(
        d_a, a.data(), bytes, cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(
        d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    // Configure and start GPU threads
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    add<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Copy results from GPU back to CPU
    CHECK(cudaMemcpy(
        c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    
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