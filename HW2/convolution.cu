#include <cuda_runtime.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define gpuCheck(stmt)                                               \
  do {                                                               \
      cudaError_t err = stmt;                                        \
      if (err != cudaSuccess) {                                      \
          fprintf(stderr, "CUDA error in %s: %s\n", #stmt,            \
                  cudaGetErrorString(err));                          \
          exit(EXIT_FAILURE);                                       \
      }                                                              \
  } while (0)

struct timeval t_start, t_end;
void cputimer_start(){
  gettimeofday(&t_start, 0);
}

double cputimer_elapsed_us(){
  gettimeofday(&t_end, 0);
  return 1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec;
}

void cputimer_stop(const char* info){
  double time = cputimer_elapsed_us();
  printf("Timing - %s. \t\tElapsed %.0f microseconds \n", info, time);
}

// Report the average time of one run when `runs` runs were timed together.
void cputimer_stop_avg(const char* info, int runs){
  double time = cputimer_elapsed_us();
  printf("Timing - %s. \t\tAverage %.2f microseconds over %d runs \n",
         info, time / runs, runs);
}

#define MASK_WIDTH 5
// Number of timed kernel launches; the average hides launch jitter and the
// microsecond resolution of gettimeofday for small N.
#define NUM_RUNS 100
#ifndef TILE_WIDTH
#define TILE_WIDTH 512  // chosen from the nsys tile sweep (results/sweep_nsys.txt)
#endif
#if TILE_WIDTH < 1 || TILE_WIDTH > 1024
#error "TILE_WIDTH must be between 1 and 1024"
#endif

__global__ void convolution_1D_basic(float *N, float *M, float *P, int width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) {
        return;
    }

    int start = i - MASK_WIDTH / 2;
    float sum = 0.0f;
    for (int j = 0; j < MASK_WIDTH; ++j) {
        int input_index = start + j;
        if (input_index >= 0 && input_index < width) {
            sum += N[input_index] * M[j];
        }
    }
    P[i] = sum;
}

__global__ void convolution_1D_tiled(float *N, float *M, float *P, int width)
{
    __shared__ float input_tile[TILE_WIDTH + MASK_WIDTH - 1];
    int tx = threadIdx.x;
    int block_start = blockIdx.x * TILE_WIDTH;

    // Load the output tile plus both halos. The loop also supports tiles < 4.
    for (int s = tx; s < TILE_WIDTH + MASK_WIDTH - 1; s += TILE_WIDTH) {
        int input_index = block_start - MASK_WIDTH / 2 + s;
        input_tile[s] = (input_index >= 0 && input_index < width)
                            ? N[input_index] : 0.0f;
    }
    // Every thread must finish loading before any thread reads its neighbors.
    __syncthreads();

    int i = block_start + tx;
    if (i < width) {
        float sum = 0.0f;
        for (int j = 0; j < MASK_WIDTH; ++j) {
            sum += input_tile[tx + j] * M[j];
        }
        P[i] = sum;
    }
}

void convolution_1D_cpu(const float *N, const float *M, float *P, int width)
{
    for (int i = 0; i < width; ++i) {
        int start = i - MASK_WIDTH / 2;
        float sum = 0.0f;
        for (int j = 0; j < MASK_WIDTH; ++j) {
            int input_index = start + j;
            if (input_index >= 0 && input_index < width) {
                sum += N[input_index] * M[j];
            }
        }
        P[i] = sum;
    }
}

bool validate(const float *output, const float *reference, int width, const char *name)
{
    const float tolerance = 1e-5f;
    int mismatches = 0;
    float max_error = 0.0f;
    for (int i = 0; i < width; ++i) {
        float error = fabsf(output[i] - reference[i]);
        if (error > max_error) {
            max_error = error;
        }
        // The negated comparison also rejects NaN results.
        if (!(error < tolerance)) {
            if (mismatches < 5) {
                fprintf(stderr, "%s mismatch at %d: GPU=%.9g CPU=%.9g\n",
                        name, i, output[i], reference[i]);
            }
            ++mismatches;
        }
    }
    printf("%s validation: %s (width=%d, mismatches=%d, max abs error=%.9g)\n",
           name, mismatches == 0 ? "PASS" : "FAIL", width, mismatches, max_error);

    return mismatches == 0;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <positive input length>\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *end;
    errno = 0;
    long parsed_width = strtol(argv[1], &end, 10);
    // Leave room for padded thread indices and halo indices in both kernels.
    const int max_block_size = TILE_WIDTH > 256 ? TILE_WIDTH : 256;
    if (errno != 0 || end == argv[1] || *end != '\0' ||
        parsed_width <= 0 || parsed_width > INT_MAX - max_block_size - MASK_WIDTH) {
        fprintf(stderr, "Invalid input length: %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    int width = (int)parsed_width;
    size_t bytes = (size_t)width * sizeof(float);
    size_t mask_bytes = MASK_WIDTH * sizeof(float);

    cputimer_start();
    float *hostN = (float *)malloc(bytes);
    float *hostM = (float *)malloc(mask_bytes);
    float *hostP = (float *)malloc(bytes);
    float *hostReference = (float *)malloc(bytes);
    if (!hostN || !hostM || !hostP || !hostReference) {
        fprintf(stderr, "Failed to allocate host memory\n");
        free(hostN);
        free(hostM);
        free(hostP);
        free(hostReference);
        return EXIT_FAILURE;
    }
    cputimer_stop("Allocated host memory");

    float *deviceN;
    float *deviceM;
    float *deviceP;
    cputimer_start();
    gpuCheck(cudaMalloc((void **)&deviceN, bytes));
    gpuCheck(cudaMalloc((void **)&deviceM, mask_bytes));
    gpuCheck(cudaMalloc((void **)&deviceP, bytes));
    cputimer_stop("Allocated device memory");

    cputimer_start();
    for (int i = 0; i < width; ++i) {
        hostN[i] = rand() / (float)RAND_MAX;
        hostP[i] = 0.0f;
    }
    const float mask[MASK_WIDTH] = {-0.25f, 0.5f, 1.0f, 0.5f, 0.25f};
    for (int j = 0; j < MASK_WIDTH; ++j) {
        hostM[j] = mask[j];
    }
    cputimer_stop("Initialized host arrays");

    cputimer_start();
    gpuCheck(cudaMemcpy(deviceN, hostN, bytes, cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(deviceM, hostM, mask_bytes, cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(deviceP, hostP, bytes, cudaMemcpyHostToDevice));
    cputimer_stop("Copying data to the GPU");

    const int blockSize = 256;
    const int gridSize = (width + blockSize - 1) / blockSize;
    // Warm-up launch (not timed): the first launch of a kernel also pays
    // one-time costs such as lazy module loading.
    convolution_1D_basic<<<gridSize, blockSize>>>(deviceN, deviceM, deviceP, width);
    gpuCheck(cudaGetLastError());
    gpuCheck(cudaDeviceSynchronize());

    cputimer_start();
    for (int run = 0; run < NUM_RUNS; ++run) {
        convolution_1D_basic<<<gridSize, blockSize>>>(deviceN, deviceM, deviceP, width);
    }
    gpuCheck(cudaGetLastError());
    gpuCheck(cudaDeviceSynchronize());
    cputimer_stop_avg("Finished 1D convolution (basic)", NUM_RUNS);

    cputimer_start();
    gpuCheck(cudaMemcpy(hostP, deviceP, bytes, cudaMemcpyDeviceToHost));
    cputimer_stop("Copying output P to the CPU");

    convolution_1D_cpu(hostN, hostM, hostReference, width);
    bool basic_valid = validate(hostP, hostReference, width, "Basic");

    // Clear P on both sides so the tiled run cannot pass validation by
    // leaving the basic kernel's results in place.
    gpuCheck(cudaMemset(deviceP, 0, bytes));
    memset(hostP, 0, bytes);

    const int tiledGridSize = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    printf("Tiled configuration: TILE_WIDTH=%d, blocks=%d\n", TILE_WIDTH, tiledGridSize);
    convolution_1D_tiled<<<tiledGridSize, TILE_WIDTH>>>(deviceN, deviceM, deviceP, width);
    gpuCheck(cudaGetLastError());
    gpuCheck(cudaDeviceSynchronize());

    cputimer_start();
    for (int run = 0; run < NUM_RUNS; ++run) {
        convolution_1D_tiled<<<tiledGridSize, TILE_WIDTH>>>(deviceN, deviceM, deviceP, width);
    }
    gpuCheck(cudaGetLastError());
    gpuCheck(cudaDeviceSynchronize());
    cputimer_stop_avg("Finished 1D convolution (tiled)", NUM_RUNS);

    cputimer_start();
    gpuCheck(cudaMemcpy(hostP, deviceP, bytes, cudaMemcpyDeviceToHost));
    cputimer_stop("Copying tiled output P to the CPU");
    bool tiled_valid = validate(hostP, hostReference, width, "Tiled");

    cputimer_start();
    gpuCheck(cudaFree(deviceN));
    gpuCheck(cudaFree(deviceM));
    gpuCheck(cudaFree(deviceP));
    free(hostN);
    free(hostM);
    free(hostP);
    free(hostReference);
    cputimer_stop("Free memory resources");

    return basic_valid && tiled_valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
