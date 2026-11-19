# Bonus Report — GPU Particle Mover

## 1. Environment, build, and experiment

The experiments ran on a Google Colab Linux runtime (kernel 6.6.159+, x86-64, glibc 2.39). It exposed two logical CPUs on an Intel Xeon at 2.00 GHz and one NVIDIA Tesla T4 with 15,360 MiB of device memory. The T4 has compute capability 7.5 and used NVIDIA driver 580.82.07. The compiler was CUDA 13.0 (\`nvcc\` 13.0.88) with GNU C++ 13.3.0.

The Makefile used \`ARCH=sm_75\`, passed it to \`nvcc\` as \`-arch=sm_75\`, used \`g++\` as the CUDA host compiler, and built \`bin/miniPIC.out\` with \`-O3\`. The run used the existing Makefile with \`make -B ARCH=sm_75\` to force a rebuild. Both mover modes used that same executable.

For each input, the Notebook ran the simulator sequentially in CPU and GPU modes, using \`MINIPIC_MOVER=cpu\` and \`MINIPIC_MOVER=gpu\` respectively. Each run had its own working directory and \`data/\` subdirectory. The initializer seeds its pseudo-random generator with the species ID, giving each pair the same initial particle distribution.

| Input | Grid | Total particles | Cycles | Time step |
| --- | --- | ---: | ---: | ---: |
| \`GEM_2D.inp\` (required) | 256 × 128 × 1 | 16,384,000 | 10 | 0.25 |
| \`GEM_3D.inp\` (additional) | 128 × 64 × 64 | 56,623,104 | 10 | 0.25 |

The Colab run saved the build log, simulation logs, VTK files, and \`summary.json\` under \`/content/bonus_runs/20261119-111945-233569\`.

## 2. CUDA mover design

\`mover_PC_gpu()\` processes one species per call. It allocates device buffers and uploads the six particle arrays (\`x\`, \`y\`, \`z\`, \`u\`, \`v\`, \`w\`), the three grid-coordinate arrays, and the six electric and magnetic field arrays. The grid arrays use flat storage in device global memory. The kernel receives their device addresses and scalar grid parameters through \`MoverMesh\`. It maps the node indices \`(ix, iy, iz)\` to the offset \`(ix * ny + iy) * nz + iz\`.

Each block has 256 threads, and each active thread advances one particle. Inside each thread, subcycles and midpoint iterations run in order. The thread interpolates the field from eight surrounding nodes, computes the midpoint velocity, advances the particle, and applies periodic or reflecting boundaries. Field reads can be shared across threads because the mover leaves the grid fields unchanged, and each thread writes only its own particle state.

The host checks for CUDA errors, waits for the kernel to finish, copies the updated particle arrays back, and releases the temporary device buffers. The existing CPU \`interpP2G()\` then deposits particle contributions on the grid.

## 3. Correctness

The Notebook compared the CPU and GPU \`rho_net_10.vtk\` files point by point after verifying that both variants produced the same set of output cycles and that the compared files had matching grid dimensions, origin, spacing, and point count. The per-point acceptance criterion was \`|GPU − CPU| ≤ 1e-5 + 1e-4 × |CPU|\`.

| Input | Cycle | Grid points | Max absolute error | RMSE | Points outside tolerance |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEM_2D | 10 | 32,768 | 3.0e-8 | 1.08763e-9 | 0 |
| GEM_3D | 10 | 524,288 | 1.1e-7 | 1.67721e-9 | 0 |

Both cases passed. The observed discrepancies are consistent with differences in floating-point rounding between the CPU and GPU. This check validates the saved net charge density for these configurations at cycle 10. It does not establish particle-by-particle agreement or quantify differences at earlier cycles.

## 4. Performance

Each value comes from one run. Speedup is CPU time divided by GPU time. Mover time is averaged over the 10 cycles and includes all four species per cycle. In GPU mode it includes allocation, host-to-device copies, kernel execution and synchronization, device-to-host copies, and device deallocation. Total time is the simulator's wall-clock interval from before grid setup to after cleanup.

| Input | Metric | CPU | GPU | Speedup |
| --- | --- | ---: | ---: | ---: |
| GEM_2D | Mover (s/cycle) | 3.40729 | 0.252729 | 13.482× |
| GEM_2D | Interpolation and output (s/cycle) | 2.53858 | 2.48942 | |
| GEM_2D | Total (s) | 62.2649 | 30.2167 | 2.061× |
| GEM_3D | Mover (s/cycle) | 14.9592 | 0.759637 | 19.693× |
| GEM_3D | Interpolation and output (s/cycle) | 10.7984 | 11.3088 | |
| GEM_3D | Total (s) | 267.754 | 130.148 | 2.057× |

The GPU mover reduced total wall time by about 51% in each case. Once the mover was accelerated, CPU-side deposition and output accounted for most of the measured time. The program's \`Interp. Time / Cycle\` counter also covers density boundary conditions, species summation, and VTK writes when scheduled. GEM_2D writes VTK at cycle 10, while GEM_3D writes at every cycle, so this counter cannot isolate interpolation or explain its run-to-run variation. Additional repetitions would be needed to measure timing variability.

## Appendix — Colab simulation log excerpts

These excerpts come from the four runs used in the report. The charge-to-mass ratios for species 0–3 were −64, +1, −64, and +1, respectively. Both inputs used a time step of 0.25 and ran for 10 cycles.

| Input | Particles per species | Grid | Saved VTK cycles |
| --- | ---: | --- | --- |
| GEM_2D | 4,096,000 | 256 × 128 × 1 | 10 |
| GEM_3D | 14,155,776 | 128 × 64 × 64 | 1–10 |

### GEM_2D

**CPU**

```text
Particle mover: CPU
Opening file: ./data/rho_net_10.vtk
Tot. Simulation Time (s) = 62.2649
Mover Time / Cycle   (s) = 3.40729
Interp. Time / Cycle (s) = 2.53858
```

**GPU**

```text
Particle mover: GPU
Opening file: ./data/rho_net_10.vtk
Tot. Simulation Time (s) = 30.2167
Mover Time / Cycle   (s) = 0.252729
Interp. Time / Cycle (s) = 2.48942
```

### GEM_3D

**CPU**

```text
Particle mover: CPU
Opening file: ./data/rho_net_10.vtk
Tot. Simulation Time (s) = 267.754
Mover Time / Cycle   (s) = 14.9592
Interp. Time / Cycle (s) = 10.7984
```

**GPU**

```text
Particle mover: GPU
Opening file: ./data/rho_net_10.vtk
Tot. Simulation Time (s) = 130.148
Mover Time / Cycle   (s) = 0.759637
Interp. Time / Cycle (s) = 11.3088
```
