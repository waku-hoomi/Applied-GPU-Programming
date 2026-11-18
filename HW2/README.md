# HW2 — Q1: 1D Convolution

## Files

- `convolution.cu`: basic and tiled (shared memory) 1D convolution kernels, CPU validation, and timing.
- `Q1.md`: report section for Q1 (Q1.1–Q1.4).
- `image/q1_final_run_1.png`, `image/q1_final_run_2.png`: output of the final run used in Q1.3.

## Compile and run

Recorded experiments used one MIG instance of an NVIDIA H100 (16 SMs) on the KTH JupyterLab server with CUDA 12.5. Adjust the architecture flag for other GPUs (for example `-arch=sm_75` on a Tesla T4).

```bash
nvcc -O2 -lineinfo -arch=sm_90 convolution.cu -o convolution
for n in 1024 2048 4096 8192 16384; do ./convolution $n; done
```

The program takes N as a command-line argument. The mask has 5 elements. Both kernels are validated against a CPU reference. Each kernel is launched once as a warm-up, and then the average host-side time of 100 launches is printed. The tile size is 512 by default and can be changed at compile time with `-DTILE_WIDTH=<1..1024>`.

## Tile size and profiling

To compare tile sizes, compile with different `-DTILE_WIDTH` values and read the GPU kernel times with Nsight Systems:

```bash
nvcc -O2 -lineinfo -arch=sm_90 -DTILE_WIDTH=256 convolution.cu -o conv_t256
nsys profile -o nsys_t256 ./conv_t256 16384
nsys stats --report cuda_gpu_kern_sum nsys_t256.nsys-rep
```

Nsight Compute was run on N = 16384. On a MIG instance the GPU clock cannot be locked, so `--clock-control none` is needed:

```bash
ncu --set full --clock-control none -k convolution_1D_tiled -c 1 -o ncu_tiled ./convolution 16384
```

Timings measured while ncu is attached include profiler overhead and should not be used as kernel times.
