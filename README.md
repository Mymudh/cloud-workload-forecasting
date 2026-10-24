# Cloud Workload Forecasting Challenge

A reproducible machine-learning project for forecasting cloud workload metrics using the Microsoft Azure Public Dataset V2.

The project analyzes historical virtual-machine CPU readings, compares baseline and machine-learning forecasting methods, evaluates models using chronological validation, and generates predictions for the next 24 timestamps at five-minute intervals.

## 1. Project Overview

Cloud infrastructure teams need reliable workload forecasts to plan capacity, monitor resource utilization, and make informed scaling decisions.

This project builds a forecasting pipeline for two cloud workload metrics:

1. **Active VM Count:** The number of active virtual machines at each timestamp.
2. **Aggregate CPU Utilization:** The mean CPU utilization across virtual-machine readings at each timestamp.

The pipeline includes data analysis, feature engineering, baseline forecasting, machine-learning model comparison, prediction generation, output validation, and automated tests.

### Project Objectives

* Analyze historical cloud workload telemetry.
* Understand workload patterns and data characteristics.
* Prepare time-series data for forecasting.
* Engineer lag-based and rolling statistical features where appropriate.
* Compare baseline methods with machine-learning models.
* Evaluate forecasting performance using chronological validation.
* Generate 24 future predictions at five-minute intervals.
* Prevent the final 24 target values from being used during training and model selection.
* Validate the final prediction file.
* Provide recommendations for cloud capacity planning and operational monitoring.

## 2. Forecast Targets

### Active VM Count

**Column:** `active_vm_count`

Represents the number of active virtual machines at each timestamp.

The analyzed dataset contains 10,000 VMs, and the observed active VM count is constant at 10,000 across the 134 observed timestamps.

Consequently, the selected persistence baseline predicts 10,000 for each forecast timestamp.

This is an observation about the supplied dataset, not a guarantee that future VM counts will remain constant.

### Aggregate CPU Utilization

**Column:** `aggregate_cpu_utilization`

Represents the mean CPU utilization across virtual-machine readings at each timestamp, as implemented in this project.

The pipeline compares multiple forecasting methods to identify an appropriate model for this target.

CPU utilization predictions must remain within the valid range of 0 to 100.

## 3. Dataset

The project uses the fixed `data-v1` release of the Microsoft Azure Public Dataset V2 curated for this challenge.

**Official challenge repository:**

https://github.com/atomityhq/forecasting-challenge

**Dataset release:**

https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1

**Dataset documentation:**

https://github.com/atomityhq/forecasting-challenge/blob/main/DATASET.md

**Original Microsoft dataset:**

https://github.com/Azure/AzurePublicDataset

### Dataset Characteristics

| Property                   | Description                                   |
| -------------------------- | --------------------------------------------- |
| Dataset source             | Microsoft Azure Public Dataset V2             |
| Challenge release          | `data-v1`                                     |
| Virtual machines           | 10,000                                        |
| Observed timestamps per VM | 134                                           |
| Sampling interval          | 5 minutes                                     |
| CPU observations           | 1,340,000                                     |
| Forecast horizon           | 24 timestamps                                 |
| Forecast interval          | 5 minutes                                     |
| Total forecast duration    | 2 hours                                       |
| Target variables           | Active VM count and aggregate CPU utilization |

Use the exact dataset release supplied for this assignment. Do not substitute another dataset or version.

### Dataset Files

The `challenge_data/` directory contains the following files:

```text
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
├── SHA256SUMS
└── README.md
```

The compressed CSV files should remain compressed. Python can read them directly using pandas.

The dataset is distributed separately from the source repository, so the large dataset files should not be committed to GitHub unless the assignment explicitly requires otherwise.

## 4. Technology Stack

| Technology   | Purpose                                          |
| ------------ | ------------------------------------------------ |
| Python 3.11+ | Main programming language                        |
| Pandas       | Data loading, transformation, and analysis       |
| NumPy        | Numerical operations                             |
| Scikit-learn | Machine-learning models and evaluation utilities |
| Pytest       | Automated testing                                |
| Git          | Version control                                  |
| GitHub       | Source-code hosting and submission               |

## 5. Project Structure

```text
forecasting-challenge-main/
│
├── challenge_data/
│   ├── vm_cpu_readings.csv.gz
│   ├── vm_metadata.csv.gz
│   ├── DATA_CARD.md
│   ├── SHA256SUMS
│   └── README.md
│
├── outputs/
│   ├── model_comparison.csv
│   └── predictions.csv
│
├── src/
│   └── cwfc/
│       ├── __init__.py
│       ├── baselines.py
│       ├── metrics.py
│       └── validation.py
│
├── tests/
│   ├── test_metrics.py
│   └── test_validation.py
│
├── forecast_pipeline.py
├── workload_analysis.py
├── visualize_workload.py
├── pyproject.toml
├── ASSIGNMENT.md
├── DATASET.md
├── DATA_CARD_TEMPLATE.md
├── DATA_RELEASE.md
├── README.md
├── SOURCES.md
├── SUBMISSION.md
├── TECHNICAL_REPORT.md
└── .gitignore
```

The `outputs/` directory may contain additional files generated by your analysis scripts. The exact output files depend on which scripts have been run.

## 6. Environment Setup

Python 3.11 or newer is required.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Mymudh/cloud-workload-forecasting.git
cd cloud-workload-forecasting
```

Alternatively, download the repository ZIP and open the extracted project folder in PyCharm.

### Step 2: Create a Virtual Environment

On Windows PowerShell:

```powershell
py -m venv .venv
```

### Step 3: Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell activation is unavailable, use the Python interpreter configured for your project in PyCharm.

### Step 4: Install Dependencies

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the project and development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

### Step 5: Verify the Installation

Run the automated tests:

```powershell
python -m pytest -q
```

The current test run completed successfully with two tests passing.

## 7. Running the Project

Run the commands from the project root directory.

### Workload Analysis

```powershell
python workload_analysis.py
```

This script performs the workload analysis implemented in the project.

### Workload Visualization

```powershell
python visualize_workload.py
```

This script runs the project's workload visualization functionality.

### Forecasting Pipeline

```powershell
python forecast_pipeline.py
```

The pipeline evaluates forecasting models, selects models using validation MAE, generates future predictions, validates the output, and writes the prediction and model-comparison files.

### Automated Tests

```powershell
python -m pytest -q
```

The latest observed test result was:

```text
2 passed
```

This confirms that the two tests in the current test suite passed. It does not establish that every possible input or scenario has been tested.

## 8. Forecasting Methodology

The forecasting pipeline compares baseline methods and machine-learning models using historical workload observations.

The final 24 target values are excluded from model training and model selection.

### Models Evaluated

The current model comparison includes:

1. Persistence
2. Seasonal Naive with a seasonal period of 12
3. Random Forest
4. Histogram-based Gradient Boosting

### Persistence Baseline

The persistence baseline uses the most recent observed target value as the forecast.

This provides a simple reference point for evaluating more complex forecasting methods.

### Seasonal Naive Baseline

The Seasonal Naive (12) method uses observations from the corresponding positions in the previous 12-step seasonal cycle.

At five-minute sampling intervals, 12 observations represent one hour.

This baseline is useful for comparing forecasts when recent workload patterns repeat over time.

### Random Forest

Random Forest is an ensemble learning method that combines predictions from multiple decision trees.

Its performance depends on the available training data, feature engineering, and model configuration.

### Histogram-Based Gradient Boosting

Histogram-based Gradient Boosting is a tree-based ensemble method that builds successive trees to improve predictions.

Its performance is evaluated using the same reported validation metrics as the other models.

## 9. Model Evaluation Results

The following results were recorded in `outputs/model_comparison.csv`.

| Target                      | Model                |      MAE |     RMSE |
| --------------------------- | -------------------- | -------: | -------: |
| `active_vm_count`           | Persistence          | 0.000000 | 0.000000 |
| `active_vm_count`           | Seasonal Naive (12)  | 0.000000 | 0.000000 |
| `active_vm_count`           | Random Forest        | 0.000000 | 0.000000 |
| `active_vm_count`           | HistGradientBoosting | 0.000000 | 0.000000 |
| `aggregate_cpu_utilization` | Seasonal Naive (12)  | 0.086357 | 0.104226 |
| `aggregate_cpu_utilization` | HistGradientBoosting | 0.086502 | 0.107515 |
| `aggregate_cpu_utilization` | Persistence          | 0.087578 | 0.103766 |
| `aggregate_cpu_utilization` | Random Forest        | 0.123885 | 0.144383 |

**Important:** These are the metrics recorded by the current project. They should not be interpreted as a guarantee of future forecasting performance.

### Interpretation

**Active VM count**

All evaluated models have zero reported error for this target. The observed VM count is constant at 10,000 throughout the analyzed timestamps.

**CPU utilization**

Seasonal Naive (12) has the lowest reported MAE among the evaluated models.

Persistence has the lowest reported RMSE among the evaluated CPU models.

The current pipeline selects models using validation MAE. Therefore, it selects Seasonal Naive (12) for CPU utilization.

MAE and RMSE measure different aspects of prediction error. MAE measures the average absolute error, while RMSE gives relatively greater weight to larger errors.

Model selection should also consider chronological validation design, the forecast horizon, and whether the validation data represents the intended forecasting task.

## 10. Prediction Output

The forecasting pipeline generates:

```text
outputs/
├── model_comparison.csv
└── predictions.csv
```

### Prediction File

The file `outputs/predictions.csv` contains the final 24 predictions.

Its required columns are:

```csv
timestamp,active_vm_count,aggregate_cpu_utilization
```

The file must contain:

* Exactly 24 data rows.
* Three columns with the required names.
* Chronologically increasing timestamps.
* Timestamps spaced five minutes apart.
* No missing values.
* No duplicate timestamps.
* Finite numerical predictions.
* Non-negative active VM counts.
* CPU utilization values between 0 and 100.

### Observed Output Validation

The prediction file was checked in the current development environment.

| Check                      | Observed result         |
| -------------------------- | ----------------------- |
| Number of rows             | 24                      |
| Required columns           | Present                 |
| Missing values             | 0                       |
| Duplicate timestamps       | 0                       |
| Chronological order        | Passed                  |
| Active VM count range      | 10,000–10,000           |
| CPU utilization range      | Approximately 9.11–9.44 |
| Pipeline output validation | Passed                  |

These checks confirm the structure and basic numerical validity of the observed output. They do not independently prove forecast accuracy.

### Reproduce the Output

Run:

```powershell
python forecast_pipeline.py
```

The pipeline reports the selected models, previews predictions, validates the output, and saves the generated files.

## 11. Data Leakage Prevention

Time-series forecasting must preserve the chronological relationship between observations and future targets.

The current pipeline reports that the final 24 target values were excluded from model training and model selection.

This separation is important because using the held-out target values to train a model or choose a model could make the reported results misleading.

The pipeline's reported behavior should still be reviewed against its implementation to confirm that feature engineering, validation splits, and model selection all preserve the intended separation.

## 12. Cloud Operations Recommendations

Workload forecasts can support capacity planning, but they should not be used as the sole basis for production scaling decisions.

### Monitor Forecast Errors

Compare predictions against actual workload measurements as new observations become available.

Track MAE and RMSE over time to identify deteriorating performance.

### Detect Workload Changes

Monitor changes in CPU utilization and VM counts. Reevaluate the forecasting approach when the workload pattern changes significantly.

### Plan Capacity Carefully

Use workload predictions alongside service latency, request volume, resource limits, and application-specific service-level objectives.

### Maintain Operational Safety Margins

Account for forecast uncertainty and unexpected demand when planning resource capacity.

### Validate Before Production

Evaluate the pipeline on additional time periods and representative workloads before relying on it for automated infrastructure decisions.

## 13. Limitations

The project has several important limitations.

1. The supplied dataset contains 134 observed timestamps per VM.
2. The observed active VM count is constant at 10,000 across the analyzed timestamps.
3. A constant historical VM count does not prove that future VM demand will remain constant.
4. The CPU forecasting models are evaluated using the project's reported validation metrics.
5. Results on this dataset do not guarantee performance on other cloud workloads.
6. The project does not establish production readiness for automated cloud scaling.
7. The current automated test suite contains only two tests, so additional testing would be beneficial.

These limitations should be considered when interpreting the results.

## 14. Reproducibility

To reproduce the analysis:

1. Use Python 3.11 or newer.
2. Install the dependencies defined by the project.
3. Obtain the exact challenge dataset release.
4. Place the dataset files in `challenge_data/`.
5. Run the workload analysis script.
6. Run the forecasting pipeline.
7. Run the automated tests.
8. Inspect the generated prediction and model-comparison files.

Commands:

```powershell
python workload_analysis.py
python forecast_pipeline.py
python -m pytest -q
```

The outputs should be regenerated using the same dataset and project configuration.

## 15. References

* Atomity forecasting challenge repository: https://github.com/atomityhq/forecasting-challenge
* Challenge assignment: https://github.com/atomityhq/forecasting-challenge/blob/main/ASSIGNMENT.md
* Dataset documentation: https://github.com/atomityhq/forecasting-challenge/blob/main/DATASET.md
* Submission requirements: https://github.com/atomityhq/forecasting-challenge/blob/main/SUBMISSION.md
* Dataset release: https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1
* Microsoft Azure Public Dataset: https://github.com/Azure/AzurePublicDataset

## 16. Final Summary

This project implements a cloud workload forecasting pipeline using historical virtual-machine CPU readings.

It compares baseline and machine-learning forecasting approaches, evaluates model performance using reported chronological validation metrics, generates 24 future predictions, and validates the prediction file.

The project also documents the observed dataset characteristics, limitations, and recommendations for monitoring forecast performance and planning cloud capacity.

The results provide a reproducible starting point for workload forecasting, while additional validation is necessary before applying the approach to other datasets or production infrastructure.
