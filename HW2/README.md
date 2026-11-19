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

## Bonus — Particle Simulation

Build from the HW2 directory with `make`, `nvcc`, and a compatible `g++`. The recorded runs used a Tesla T4 with CUDA 13.0 and GCC 13.3. Use `sm_75` for a T4 and set `ARCH` to match the GPU on other machines.

```bash
cd iPIC3D-mini
make -B ARCH=sm_75
```

Run the following from the same `iPIC3D-mini` directory. `GEM_2D` is the required case and `GEM_3D` is an additional experiment. Each CPU/GPU run uses the same executable and input for its case. Separate working directories keep the simulator's `./data/` outputs from overwriting one another.

```bash
(
    set -euo pipefail
    project="$PWD"
    mkdir -p bonus_runs
    results=$(mktemp -d "$project/bonus_runs/run.XXXXXXXX")
    for case in GEM_2D GEM_3D; do
        for mode in cpu gpu; do
            run="$results/$case/$mode"
            mkdir -p "$run/data"
            (
                cd "$run"
                MINIPIC_MOVER="$mode" "$project/bin/miniPIC.out" \
                    "$project/inputfiles/$case.inp" 2>&1 | tee simulation.log
            )
        done
    done
    printf 'Results directory: %s\n' "$results"
)
```

Compare the CPU and GPU `data/rho_net_10.vtk` files for each case. Compute mover speedup as CPU `Mover Time / Cycle` divided by GPU `Mover Time / Cycle` in the simulation logs. Compute overall speedup using `Tot. Simulation Time` in the same way.

For an automated run and comparison, open the [experiment notebook](iPIC3D-mini/HW2_Bonus_A%20Particle%20Simulation%20Application.ipynb) with its kernel working directory set to `iPIC3D-mini` and run all cells. It saves the logs, VTK files, and a `summary.json` with maximum absolute error, RMSE, and speedups under `bonus_runs/`.
