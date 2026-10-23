# Cloud Workload Forecasting Challenge

AI/ML take-home challenge built from the **Microsoft Azure Public Dataset V2** VM trace.

## Start here

1. Read [`ASSIGNMENT.md`](ASSIGNMENT.md).
2. Read [`DATASET.md`](DATASET.md).
3. Download the **`data-v1` GitHub Release** dataset and extract it as `challenge_data/`.
4. Read [`SUBMISSION.md`](SUBMISSION.md).
5. Implement your analysis and forecasting pipeline.

The challenge dataset is distributed as a versioned GitHub Release rather than committed to the repository. The `data-v1` release contains the fixed curated dataset used for this assignment.

## Environment setup

Python **3.11 or newer** is required.

Create a virtual environment:

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
py -m venv .venv
.venv\Scripts\activate.bat
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the installation:

```bash
python -m pytest -q
```

The repository includes a small test suite covering the starter forecasting utilities. Run it before beginning the assignment.

## Dataset download

The hiring dataset is distributed as the fixed `data-v1` release of this repository.

**Download:** [Azure Dataset v1 Release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)

Download `data-v1.tar.gz` from the release and extract it into the repository as `challenge_data/`:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
```

The resulting layout is:

```text
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

Verify the dataset before starting the assignment:

```bash
cd challenge_data
sha256sum -c SHA256SUMS
```

You should see:

```text
vm_cpu_readings.csv.gz: OK
vm_metadata.csv.gz: OK
DATA_CARD.md: OK
```

The release is a real, curated subset of the Microsoft Azure Public Dataset V2. It is not synthetic data.

The dataset contains 10,000 VMs with 134 CPU measurements per VM at five-minute intervals, for a total of 1,340,000 CPU observations.

Use the exact `data-v1` release provided for this assignment. Do not substitute another dataset or a different version of the Azure Public Dataset.

## What you are solving

The underlying data contains anonymized VM workload telemetry sampled every five minutes. Your job is to understand the workload and build a defensible forecasting pipeline for near-term aggregate demand.

The challenge is intentionally open-ended. We evaluate reasoning, time-series methodology, leakage prevention, error analysis, reproducibility, and the quality of your operational conclusions—not just model complexity.

## Expected environment

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Model dependencies are intentionally not prescribed. Add only what you need and document them.

## Source

Microsoft Azure Public Dataset V2:
https://github.com/Azure/AzurePublicDataset

The official V2 documentation describes a 30-day VM trace with five-minute CPU readings, VM metadata, and encrypted subscription/deployment/VM identifiers. The full upstream trace is much larger than this challenge release.

## Submission

Submit your work as a repository or archive according to [`SUBMISSION.md`](SUBMISSION.md).

## Candidate Implementation

This project implements a backend machine-learning pipeline for forecasting cloud workload metrics using historical virtual-machine CPU readings.

### Objectives

* Analyze historical cloud workload data.
* Aggregate CPU readings into time-series workload metrics.
* Engineer lag-based and rolling statistical features.
* Compare baseline forecasting methods with machine-learning models.
* Evaluate models using chronological validation.
* Generate 24 future workload predictions at five-minute intervals.
* Validate the prediction file before submission.

### Forecast Targets

1. `active_vm_count` — the number of active virtual machines at each timestamp.
2. `aggregate_cpu_utilization` — the mean CPU utilization across virtual-machine readings at each timestamp.

### Technology Stack

* Python 3.11+
* Pandas
* NumPy
* Scikit-learn
* Pytest

### Project Structure

```text
forecasting-challenge-main/
├── challenge_data/
│   ├── vm_cpu_readings.csv.gz
│   ├── vm_metadata.csv.gz
│   ├── DATA_CARD.md
│   └── README.md
├── outputs/
│   ├── workload_summary.csv
│   ├── model_comparison.csv
│   └── predictions.csv
├── src/
├── tests/
├── forecast_pipeline.py
├── workload_analysis.py
├── pyproject.toml
└── README.md
```

### Environment Setup

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Run the Workload Analysis

```powershell
python workload_analysis.py
```

### Run the Forecasting Pipeline

```powershell
python forecast_pipeline.py
```

The pipeline writes the following files to the `outputs/` directory:

* `model_comparison.csv` — chronological validation metrics for the evaluated models.
* `predictions.csv` — the final 24 workload predictions.
* `workload_summary.csv` — generated by the workload analysis script.

### Run Tests

```powershell
python -m pytest -q
```

### Prediction Format

The final `predictions.csv` file contains exactly these columns:

```text
timestamp,active_vm_count,aggregate_cpu_utilization
```

It contains 24 chronological rows, with timestamps spaced five minutes apart.

The pipeline checks for missing values, duplicate timestamps, non-finite predictions, negative VM counts, and CPU utilization outside the 0–100 range.

### Validation and Limitations

Model selection is based on a chronological validation window. The final 24 target values are excluded from training and model selection.

Historical active VM counts in the analyzed dataset were constant at 10,000. Consequently, predictions for this target remain constant under the selected persistence baseline. This observation should not be interpreted as proof that VM counts will remain constant under future workloads.

Forecast accuracy depends on the characteristics of the supplied dataset. Validation results do not guarantee future production performance.

### Cloud Operations Recommendations

* Monitor forecast errors as new workload observations become available.
* Reevaluate the models when workload patterns change.
* Consider CPU utilization forecasts alongside service latency, request volume, and resource limits before scaling infrastructure.
* Use appropriate safety margins and operational limits when planning cloud capacity.
* Evaluate the forecasting pipeline on additional time periods before production use.
