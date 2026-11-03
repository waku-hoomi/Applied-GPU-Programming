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

