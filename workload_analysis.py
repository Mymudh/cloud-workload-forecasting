from pathlib import Path

import pandas as pd

DATA_DIR = Path("challenge_data")

cpu = pd.read_csv(DATA_DIR / "vm_cpu_readings.csv.gz")
metadata = pd.read_csv(DATA_DIR / "vm_metadata.csv.gz")

cpu["timestamp"] = pd.to_numeric(cpu["timestamp"], errors="raise")

workload = (
    cpu.groupby("timestamp")
    .agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("avg_cpu", "mean"),
        total_cpu_readings=("vm_id", "size"),
        mean_min_cpu=("min_cpu", "mean"),
        mean_max_cpu=("max_cpu", "mean"),
    )
    .reset_index()
    .sort_values("timestamp")
)

print("\nWORKLOAD SUMMARY")
print(workload.describe().round(3).to_string())

print("\nFIRST 10 TIMESTAMPS")
print(workload.head(10).to_string(index=False))

print("\nLAST 30 TIMESTAMPS")
print(workload.tail(30).to_string(index=False))

print("\nTIMESTAMP INTERVALS")
print(workload["timestamp"].diff().value_counts().sort_index())

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

workload.to_csv(output_dir / "workload_summary.csv", index=False)

print("\nSaved outputs/workload_summary.csv")
