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

# HW1 — Q3: Quantum Fourier Transformation

## Files and saved evidence

- [`HW1_Q3_Quantum Fourier Transformation.ipynb`](<HW1_Q3_Quantum Fourier Transformation.ipynb>): six-question analysis, original saved measurements, source mapping, and GUI screenshots.
- [`qft_bench_naive.c`](qft_bench_naive.c): supplied naïve QFT benchmark; its algorithm and timing logic are unchanged.
- `images/q3_5_24q_timeline.png`: 24-qubit GUI timeline and selected Hadamard kernel.
- `images/q3_6_16q_timeline.png` and `images/q3_6_28q_timeline.png`: scaling screenshots.
- `report.md`: existing group report. For Q3, distinguish its separate standalone 24-qubit timing from the notebook's profiled run; its CPU launch-duration claim and its use of a statistics average as a timeline duration still require reconciliation with the reviewed notebook.

## Environment and build

The saved experiment used a Tesla T4 (compute capability 7.5) in a Linux/Colab GPU runtime with QuEST's CUDA backend. Install CUDA, CMake, Nsight Systems CLI, and GNU `time` in that environment. Install the Nsight Systems desktop GUI on your own computer for report inspection. The notebook and image folder should remain together for relative image links.

In Colab, upload the original `qft_bench_naive.c` to `/content/`, then run:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
git clone https://github.com/QuEST-Kit/QuEST.git /content/QuEST
cp /content/qft_bench_naive.c /content/QuEST/qft_bench_naive.c

cmake -S /content/QuEST -B /content/QuEST/build \
  -DQUEST_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_CUDA_FLAGS="-lineinfo" \
  -DUSER_SOURCE_NAMES=qft_bench_naive.c \
  -DUSER_OUTPUT_EXE_NAME=qft_naive
cmake --build /content/QuEST/build -j4
/content/QuEST/build/qft_naive 24 5
```

Clone only once; skip the clone when `/content/QuEST` already exists. Change `75` to match the GPU (the assignment specifies `90` for its school cluster). Explicit `-S` and `-B` resolve the working-directory issue recorded in the report. Confirm that output says **GPU backend** and total probability is close to 1. Normalization is only a sanity check.

The original QuEST checkout SHA and CUDA/Nsight tool versions were not saved. The notebook contains an optional environment-record cell for future runs; record the commit and tool versions before repeating because upstream `main` can change. Use the course-compatible QuEST v4 revision when reproducing.

## Q3.2 — Memory measurement

Run the notebook's original 28-qubit memory cell without the profiler. It launches `/usr/bin/time -v` around the benchmark and polls `nvidia-smi --query-gpu=memory.used` until completion.

| Measurement | Recorded value |
|---|---:|
| GPU baseline | 0 MiB |
| Maximum observed GPU memory | 4203 MiB (4.1045 GiB) |
| Process peak host RSS | 183492 KiB (179.19 MiB) |
| Theoretical 28-qubit state vector | 4096 MiB (4 GiB) |

The device sample is device-wide and may include other processes. The loop waits 100 ms plus query overhead, so it can miss brief peaks. RSS is host process memory, not device memory or machine-wide RAM. The excess 107 MiB on the device is consistent with runtime overhead and additional allocations, without assigning an exact cause to every byte.

## Q3.3–Q3.4 — Profile and identify hotspots

```bash
cd /content/QuEST
nsys profile --trace=cuda,osrt --stats=true -o qft_24q ./build/qft_naive 24 5
nsys profile --trace=cuda,osrt --stats=true -o qft_16q ./build/qft_naive 16 5
nsys profile --trace=cuda,osrt --stats=true -o qft_28q ./build/qft_naive 28 5
```

Use fresh output names when repeating to preserve existing reports. Profiling uses the unmodified benchmark and Nsight Systems only. Inspect `cuda_gpu_kern_sum`; default ns values must be converted when reporting ms or µs. The 24-qubit top three are:

| Kernel base name | Gate | Total GPU time (ms) | Instances | Average (µs) |
|---|---|---:|---:|---:|
| `kernel_statevec_anyCtrlOneTargDiagMatr_sub<1>` | Controlled phase shift | 1891.753348 | 1656 | 1142.3631 |
| `kernel_statevec_anyCtrlOneTargDenseMatr_subA<0>` | Hadamard | 335.207961 | 144 | 2327.8331 |
| `kernel_statevec_anyCtrlSwap_subA<0>` | SWAP | 96.363784 | 72 | 1338.3859 |

The phase kernel accounts for 80.8% of summed GPU kernel time in this capture. Counts include **one warm-up plus five timed QFTs**: 276×6 phase gates, 24×6 Hadamards, and 12×6 swaps. The notebook explains the mapping using [QuEST kernel definitions](https://github.com/QuEST-Kit/QuEST/blob/main/quest/src/gpu/gpu_kernels.cuh) and [API operations](https://github.com/QuEST-Kit/QuEST/blob/main/quest/src/api/operations.cpp).

## Q3.5 — Inspect the CPU/GPU timeline

Download the `.nsys-rep` files and open them in the desktop GUI. Expand the process/thread CUDA API row and GPU kernel rows. Skip the warm-up sequence, zoom into a timed QFT, select a gate kernel, and follow its correlation to **`cudaLaunchKernel`** on the launching CPU thread. `cudaDeviceSynchronize` waits for completion rather than launching a gate.

The existing 24-qubit screenshot shows a Hadamard GPU duration of **2.303 ms**, thread **23889**, Correlation ID **806**, and launch latency **14.095 µs**. It does not show the matching CPU call duration. Capture that CPU event/details panel to finish the individual CPU/GPU evidence. CPU API duration, launch latency, and GPU execution duration are distinct quantities. Refer to the [Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) for GUI navigation.

## Q3.6 — Scaling results

| Qubits | Best / mean complete QFT (ms) | Selected kernel in supplied GUI screenshot | GUI duration |
|---|---:|---|---:|
| 16 | 1.741 / 1.911 | Hadamard | 13.728 µs |
| 28 | 8177.909 / 8189.069 | Controlled phase shift | 18.048 ms |

The screenshots select different gate types. For comparison of the same gate, the phase-kernel statistics averages are 6.9480 µs at 16 qubits and 18.1531318 ms at 28 qubits. These averages are separate from individual GUI measurements. The notebook includes both screenshots, all five printed repetition times, and their calculated sums.

The benchmark's `best / mean` is per-QFT synchronized host elapsed time; initialization is outside that interval. Profiling totals additionally include warm-up and utility work. The separate, unprofiled 28-qubit memory run gave 8169.742 / 8185.428 ms, so keep it separate from the table above.

## Remaining evidence and storage

- Add the actual 24-qubit output screenshot as `images/q3_1_24q_output.png`; it is still missing. Saved profiler output already proves the GPU run succeeded, but does not replace the requested screenshot.
- Capture the matched CPU `cudaLaunchKernel` event and, if needed, a view locating the selected timed repetition. Current images are short timeline crops.
- Preserve the source revision, tool versions, memory logs from `/tmp/qft28_output.txt` and `/tmp/qft28_memory.txt`, and all three `.nsys-rep` files outside the temporary Colab runtime. Raw reports are not currently tracked here.

This review used saved outputs and existing screenshots. The added setup/environment cells have not been executed in a fresh GPU runtime, and CUDA/Nsight experiments were not rerun locally.
