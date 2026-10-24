# Cloud Workload Forecasting — Technical Report

## 1. Executive Summary

This project develops a backend forecasting pipeline for cloud workload metrics. It processes historical virtual-machine CPU readings, constructs workload time-series data, evaluates forecasting methods, and generates predictions for the next 24 timestamps at five-minute intervals.

The two forecast targets are active virtual-machine count and aggregate CPU utilization.

The pipeline uses Python, Pandas, NumPy, and Scikit-learn. It compares baseline forecasting methods with machine-learning models and saves the final predictions in CSV format. The project also includes workload visualizations, automated output validation, and tests to support reproducibility.

## 2. Problem Statement

Cloud infrastructure teams need estimates of future resource demand to support capacity planning, workload management, and operational decisions.

This project forecasts the following metrics:

* **Active VM count:** The number of distinct virtual machines observed at each timestamp.
* **Aggregate CPU utilization:** The mean CPU utilization across the virtual-machine readings at each timestamp.

The forecasting horizon is 24 timestamps, equivalent to two hours when observations are spaced five minutes apart.

The objective is to develop a reproducible forecasting pipeline that processes historical observations, compares forecasting methods using chronological validation, and produces a valid prediction file for downstream analysis.

## 3. Data Processing and Analysis

The input dataset contains virtual-machine CPU readings and VM metadata.

The CPU readings are grouped by timestamp to calculate:

* The number of distinct VM identifiers.
* Mean CPU utilization.
* The total number of CPU readings.

The resulting time series is sorted chronologically before forecasting.

Exploratory analysis of the workload summary found that the active VM count remained constant at 10,000 across the analyzed timestamps. CPU utilization varied over time, with observed values in the approximate range of 9.11 to 9.65 in the inspected output.

The constant VM count limits the ability to evaluate how well a model predicts changes in VM demand. CPU values should be interpreted according to the dataset's documented units and aggregation methodology.

The workload visualizations are saved in the `outputs/` directory:

* `active_vm_count.png` — shows active virtual-machine counts over time.
* `aggregate_cpu_utilization.png` — shows aggregate CPU utilization over time.

These visualizations provide a graphical summary of the observed workload patterns and support the initial data analysis.

## 4. Forecasting Methodology

The pipeline evaluates baseline forecasting methods and machine-learning approaches.

### 4.1 Persistence Baseline

The latest observed target value is used as the forecast for future timestamps. This provides a simple reference method against which more complex approaches can be evaluated.

### 4.2 Seasonal-Naive Baseline

Historical values from a specified seasonal lag are reused to construct forecasts. This method provides a reference for evaluating whether historical periodic patterns are useful for forecasting.

### 4.3 Random Forest

A Random Forest regressor combines predictions from multiple decision trees. It can model nonlinear relationships between engineered features and the forecast target.

### 4.4 Histogram-Based Gradient Boosting

Histogram-based Gradient Boosting builds an ensemble of trees sequentially, with each stage improving on the previous model. It provides an alternative machine-learning approach for evaluating workload forecasting performance.

### 4.5 Feature Engineering

The forecasting pipeline can use:

* Historical lag values.
* Rolling means.
* Rolling standard deviations.
* Time-related features.

All features used for a prediction must be available at the prediction time. Future target values must not be used to construct features or train the forecasting models.

## 5. Validation and Model Selection

Forecasting models should be evaluated using chronological splits rather than randomly shuffled observations. Chronological validation better reflects the task of predicting future workload values from past observations.

The project compares models using two evaluation metrics:

* **Mean Absolute Error (MAE):** Measures the average absolute difference between observed and predicted values.
* **Root Mean Squared Error (RMSE):** Measures the square root of the average squared prediction error and gives greater weight to larger errors.

Model selection must use only historical training data and chronological validation periods. The final 24 target values must remain excluded from model training and model selection.

The pipeline generates a model comparison file and a prediction file. It also performs output validation to check the generated predictions. In the current run, the output-validation check passed, and the automated test suite reported two passing tests.

Passing the tests confirms that the tested conditions succeeded; it does not, by itself, establish that the forecasting models will perform well on future workloads.

## 6. Results and Interpretation

The forecasting pipeline compares four forecasting approaches for each target: Persistence, Seasonal Naive (12), Random Forest, and Histogram-Based Gradient Boosting.

### 6.1 Model Comparison Results

The current model comparison output contains the following validation metrics:

| Target                    | Model                             |      MAE |     RMSE |
| ------------------------- | --------------------------------- | -------: | -------: |
| Active VM Count           | Persistence                       | 0.000000 | 0.000000 |
| Active VM Count           | Seasonal Naive (12)               | 0.000000 | 0.000000 |
| Active VM Count           | Random Forest                     | 0.000000 | 0.000000 |
| Active VM Count           | Histogram-Based Gradient Boosting | 0.000000 | 0.000000 |
| Aggregate CPU Utilization | Seasonal Naive (12)               | 0.086357 | 0.104226 |
| Aggregate CPU Utilization | Histogram-Based Gradient Boosting | 0.086502 | 0.107515 |
| Aggregate CPU Utilization | Persistence                       | 0.087578 | 0.103766 |
| Aggregate CPU Utilization | Random Forest                     | 0.123885 | 0.144383 |

### 6.2 Interpretation

**Active VM Count:** All four models recorded zero MAE and RMSE in the reported comparison. The observed active VM count remained constant at 10,000 across the analyzed timestamps. Therefore, these results reflect the characteristics of this dataset and do not demonstrate that the models can accurately forecast changing VM demand.

**Aggregate CPU Utilization:** Seasonal Naive (12) achieved the lowest reported MAE of 0.086357. Persistence achieved the lowest reported RMSE of 0.103766. Because the current pipeline selects models using validation MAE, Seasonal Naive (12) is selected for CPU utilization.

MAE and RMSE measure different aspects of prediction error. MAE measures average absolute error, while RMSE gives relatively greater weight to larger errors.

These results should be interpreted in the context of the validation procedure and the limited workload history. They do not guarantee equivalent performance on future workloads or other cloud environments.

The pipeline also generated 24 future prediction rows and reported that output validation passed. Successful output validation confirms basic structural and numerical checks; it does not independently establish forecast accuracy.

## 7. Cloud Operations Recommendations

The forecasting pipeline can support cloud capacity-planning workflows. The following recommendations describe ways to use and improve the results:

1. **Monitor actual workload observations:** Compare newly observed workload values against the forecasts.
2. **Reevaluate model performance:** Reassess forecast errors as additional historical data becomes available.
3. **Use multiple operational metrics:** Consider CPU utilization together with request volume, latency, memory usage, and service-level objectives when planning capacity.
4. **Maintain safety margins:** Apply appropriate operational limits and minimum or maximum scaling thresholds.
5. **Investigate unexpected changes:** Examine sudden workload shifts and large prediction errors.
6. **Validate on additional periods:** Test the pipeline on additional historical time windows before production deployment.

Forecasts should support, rather than independently determine, infrastructure scaling decisions.

## 8. Reproducibility

The implementation is organized as a Python project containing input data, source code, automated tests, and generated outputs.

Run the following commands from the project root in the configured Python environment:

```powershell
python workload_analysis.py
python forecast_pipeline.py
python -m pytest -q
```

The workload analysis script generates the aggregated workload summary and visualizations. The forecasting pipeline generates the model comparison and prediction files. The test command runs the project's automated tests.

Generated CSV files and visualization images are stored in the `outputs/` directory.

For reproducible results, use the documented project dependencies, preserve the supplied dataset, and run the scripts from the project root.

## 9. Limitations and Future Work

The current dataset exhibits a constant active VM count in the inspected time series. This limits the usefulness of this target for evaluating dynamic demand forecasting.

Additional limitations include the restricted range of observed workload patterns and the need to evaluate forecasting performance across more historical periods.

Future work could investigate:

* Additional forecasting features.
* Longer chronological validation periods.
* Prediction uncertainty estimates.
* Forecast-drift monitoring.
* Evaluation on additional workload patterns.
* Integration with operational monitoring and resource-management safeguards.

Before production deployment, the pipeline should be tested against additional workload patterns and integrated with appropriate monitoring and resource-management controls.

## 10. Conclusion

This project establishes a reproducible backend workflow for cloud workload analysis, forecasting, evaluation, and prediction-file generation.

The pipeline aggregates historical CPU readings, compares forecasting methods, generates workload visualizations, and produces a 24-row forecast file for two workload metrics.

The resulting artifacts provide a foundation for further validation and capacity-planning research. Final conclusions about forecasting quality should be based on verified chronological validation results and the documented characteristics of the supplied dataset.
