import matplotlib.pyplot as plt
import numpy as np

sizes = ["512x512", "1024x1024", "2048x2048", "4096x4096", "8192x8192"]

# Actual measured runtimes (ms) from Google Colab Tesla T4
cpu_ms        = [50.185878, 412.606772, 5123.276870, 45672.277708, 391734.038359]
gemm_ms       = [1.779302,  11.778032,  85.894150,   731.825012,   6550.681641]
tiled_gemm_ms = [1.811251,  10.407424,  81.017242,   674.290710,   5291.535645]
wmma_ms       = [0.133123,  0.686579,   6.711910,    99.167488,    853.676086]

x = np.arange(len(sizes))
width = 0.20

fig, ax = plt.subplots(figsize=(11, 6), dpi=150)

r1 = ax.bar(x - 1.5 * width, cpu_ms,        width, label="CPU GEMM (FP64, OpenMP)",        color="#4C72B0")
r2 = ax.bar(x - 0.5 * width, gemm_ms,       width, label="GPU gemm() (FP64, 16x16)",       color="#DD8452")
r3 = ax.bar(x + 0.5 * width, tiled_gemm_ms, width, label="GPU tiled_gemm() (FP64, 16x16)", color="#C44E52")
r4 = ax.bar(x + 1.5 * width, wmma_ms,       width, label="GPU wmma_gemm() (Tensor Core)",  color="#55A868")

ax.set_yscale("log")
ax.set_xlabel("Matrix Size (M = K = N)", fontsize=12, fontweight="bold")
ax.set_ylabel("Execution Time (ms, log scale)", fontsize=12, fontweight="bold")
ax.set_title("GEMM Runtime Comparison: CPU vs. gemm() vs. tiled_gemm() vs. wmma_gemm() (Tesla T4)", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(sizes, fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.6, which="both")
ax.legend(fontsize=10, loc="upper left")

def annotate_bars(rects):
    for rect in rects:
        h = rect.get_height()
        txt = f"{h:.2f}" if h < 10 else (f"{h:.1f}" if h < 1000 else f"{h:.0f}")
        ax.annotate(
            txt,
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8, rotation=45
        )

for rects in (r1, r2, r3, r4):
    annotate_bars(rects)

plt.ylim(top=max(cpu_ms) * 5)
plt.tight_layout()
plt.savefig("gemm_runtime_comparison.png", dpi=300)
plt.show()
