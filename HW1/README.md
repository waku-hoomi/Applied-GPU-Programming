# HW1 — Q1: CUDA Vector Addition

## Files

- `vecAdd.cu`: final vector addition program, CPU validation, and timing.
- `vecAdd_before_timing.cu`: single-launch version before timing was added (N=263149).
- `HW1_Q1_VectorAdd.ipynb`: notebook containing answers, saved outputs, batch experiments, and plotting code.
- `q1_7_timings.csv`: recorded timing data for 12 vector lengths, in milliseconds.
- `q1_7_time_breakdown.png`: stacked timing chart.

## Compile and run

Use an NVIDIA GPU with CUDA. Recorded experiments used a Tesla T4 on Google Colab with CUDA Toolkit 13.0. Apple GPUs cannot run this CUDA code.

```bash
nvcc -O2 -lineinfo -arch=sm_75 vecAdd.cu -o vecAdd
./vecAdd 512
./vecAdd 263149
```

The final program accepts N as a command-line argument and defaults to 263149. Each block has 256 threads. It compares every output element against the CPU reference and reports validation results. Adjust the architecture flag for other GPUs.

## Occupancy

The recorded results are 24.76% for N=512 and 80.17% for N=263149. These measurements used the program before repeated timing was added.

```bash
nvcc -O2 -lineinfo -arch=sm_75 vecAdd_before_timing.cu -o vecAdd_single
ncu --section Occupancy ./vecAdd_single
```

The backup source fixes N at 263149. For the 512-element experiment, make a separate copy, change `const int N = 263149;` to `const int N = 512;`, then compile and profile that copy. Performance counter access is required. New runs may produce different occupancy values.

## Timing and plotting

Each final-program invocation performs one warm-up and 10 measured repetitions. It reports mean H2D, kernel, and D2H times using the course's gettimeofday host-side timer. H2D includes both inputs and a completion synchronization. Kernel timing includes launch and synchronization overhead. Allocation, initialization, CPU reference calculation, validation, and printing are excluded. Validation checks the final repetition.

In Colab, upload `vecAdd.cu` to `/content`, compile it using the notebook's command, and run the batch and plotting cells in Q1.7. They generate the CSV and PNG. Timing runs should be separate from profiler runs. To redraw the saved chart without new measurements, upload the supplied CSV and run only the plotting cell.

## Notebook and storage notes

The notebook preserves development history. Early saved outputs came from earlier source versions. Re-running old cells uses the current source, not its historical version; use explicit N arguments with the final program. Independent `.cu` files are not embedded in the notebook. Colab runtime files are temporary, and mounting Google Drive does not automatically back them up.

This folder contains Q1 materials. The final group PDF and assignment submission ZIP are separate deliverables.

# HW1 — Q2: 2D Dense Matrix Multiplication

## Files

- `matMul.cu`: CUDA C++ source code implementing 2D dense matrix multiplication. Supports both `double` and `float` types via preprocessor macros, includes CPU validation (with sampled checks for huge matrices), and host-side timing.
- `HW1_Q2_MatrixMul.ipynb`: notebook containing environment setup, code generation, profiling commands, batch experiments, and plotting scripts.
- `q2_4_timings_double.csv` & `q2_5_timings_float.csv`: recorded timing data for 12 square matrix dimensions (M=N=P), in milliseconds.
- `q2_4_time_breakdown_double.png` & `q2_5_time_breakdown_float.png`: stacked timing charts comparing absolute and relative time breakdowns.

## Compile and run

Similar to Q1, use an NVIDIA GPU with CUDA (e.g., Tesla T4 on Google Colab). The program takes matrix dimensions `M N P` as arguments, computing $C(M \times P) = A(M \times N) \times B(N \times P)$. 

```bash
# Compile for double-precision (default, for Q2.2 - Q2.4)
nvcc -O3 -arch=sm_75 matMul.cu -o matMul_double

# Compile for single-precision float (for Q2.5)
nvcc -O3 -arch=sm_75 -DUSE_FLOAT matMul.cu -o matMul_float

# Run specific matrix sizes
./matMul_double 128 256 32
./matMul_double 1024 8191 8197
```

Each thread block is configured to $16 \times 16$ (256 threads). The kernel uses ceiling division and boundary checks to safely handle non-multiple dimensions like 8191. The program automatically verifies the GPU result against a CPU reference. Adjust the `-arch` flag if testing on a different GPU.

## Occupancy

The recorded Achieved Occupancy results on a Tesla T4 are 24.75% for the $128 \times 256 \times 32$ matrices, and 99.83% for the $1024 \times 8191 \times 8197$ matrices. 

```bash
# Profile small matrices (Q2.2)
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./matMul_double 128 256 32 1 0

# Profile large matrices (Q2.3)
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./matMul_double 1024 8191 8197 1 0
```

*(Note: The trailing `1 0` arguments used in the profiling commands tell the program to skip CPU verification and CSV mode respectively, significantly speeding up the profiling process).*

## Timing and plotting

For Q2.4 and Q2.5, a Python script inside the notebook drives the compiled executables (`matMul_double` and `matMul_float`), iterating through 12 square matrix sizes ($N=128$ to $N=4096$). It extracts the Host-to-Device (H2D), Kernel, and Device-to-Host (D2H) timings reported directly by the C++ program.

To reproduce the charts, run the final code cell in the Colab notebook. This will execute the scaling sweeps, dump the data into the two `.csv` files, and generate the corresponding stacked bar `.png` charts. As with Q1, timing runs are strictly separated from `ncu` profiling runs to avoid overhead interference.
