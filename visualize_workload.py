from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path("outputs")
SUMMARY_FILE = OUTPUT_DIR / "workload_summary.csv"


def main():
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {SUMMARY_FILE}. Run workload_analysis.py first."
        )

    OUTPUT_DIR.mkdir(exist_ok=True)

    data = pd.read_csv(SUMMARY_FILE)

    required = {
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data.sort_values("timestamp").reset_index(drop=True)

    if data.empty:
        raise ValueError("Workload summary contains no rows.")

    figures = [
        (
            "active_vm_count",
            "Active Virtual Machines Over Time",
            "Active VM Count",
            "active_vm_count.png",
        ),
        (
            "aggregate_cpu_utilization",
            "Aggregate CPU Utilization Over Time",
            "Aggregate CPU Utilization",
            "aggregate_cpu_utilization.png",
        ),
    ]

    for column, title, ylabel, filename in figures:
        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(data["timestamp"], data[column], linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        output_path = OUTPUT_DIR / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        print(f"Saved: {output_path}")

    print(f"Analyzed rows: {len(data)}")


if __name__ == "__main__":
    main()