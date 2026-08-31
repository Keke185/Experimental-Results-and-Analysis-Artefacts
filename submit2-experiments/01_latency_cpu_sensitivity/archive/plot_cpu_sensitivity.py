"""

Plot the sensitivity curves of CPU and latency at the sandbox edge in the deployment configuration/sandbox overhead section of the paper.

Read the cpu_sensitivity_results.csv file.

Summarize the average/minimum/maximum end-to-end latency for each CPU level and save the double logarithmic line plot as cpu_sensitivity_curve.png.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# File path
CSV_PATH = "/kaggle/input/datasets/kehuang5/cpu-sensitivity-results/cpu_sensitivity_results.csv"

# Average latency (milliseconds) of the unsandboxed edge baseline,
# with 58 milliseconds used as the representative midpoint here
UNSANDBOXED_BASELINE_MS = 58

OUTPUT_BASENAME = "cpu_sensitivity_curve"  # save as .png (600dpi), .pdf and .svg

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def main():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

    # Delete the temporary test run and force the remaining labels to be converted to numerical CPU core values.
    df = df[df["cpu_label"].apply(lambda x: str(x).replace(".", "", 1).isdigit())].copy()
    df["cpu_cores"] = df["cpu_label"].astype(float)

    summary = (
        df.groupby("cpu_cores")["end_to_end_ms"]
        .agg(mean="mean", min="min", max="max", n="count")
        .reset_index()
        .sort_values("cpu_cores")
    )

    print(summary)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(
        summary["cpu_cores"], summary["mean"],
        marker="o", markersize=5, linewidth=2,
        color="#2a78d6", label="Sandboxed edge (measured mean)",
    )

    ax.fill_between(
        summary["cpu_cores"], summary["min"], summary["max"],
        color="#2a78d6", alpha=0.12, label="min-max range",
    )

    ax.axhline(
        UNSANDBOXED_BASELINE_MS, color="#898781", linestyle="--", linewidth=1.5,
        label=f"Unsandboxed edge baseline (~{UNSANDBOXED_BASELINE_MS}ms)",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CPU cores allocated (--cpus)")
    ax.set_ylabel("End-to-end latency, ms (log scale)")
    ax.set_title("Sandboxed edge: latency vs. CPU allocation")
    ax.set_xticks(summary["cpu_cores"])
    ax.set_xticklabels([str(c) for c in summary["cpu_cores"]])
    ax.grid(True, which="both", linestyle="-", linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()

    png_path = f"{OUTPUT_BASENAME}.png"
    pdf_path = f"{OUTPUT_BASENAME}.pdf"
    svg_path = f"{OUTPUT_BASENAME}.svg"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    print(f"\n Saved:")
    print(f"     {png_path}  (600dpi raster -- use in Word/PowerPoint)")
    print(f"     {pdf_path}  (vector -- use in LaTeX with \\includegraphics)")
    print(f"     {svg_path}  (vector, editable)")


if __name__ == "__main__":
    main()
