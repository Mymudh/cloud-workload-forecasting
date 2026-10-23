from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "challenge_data"
OUTPUT_DIR = BASE_DIR / "outputs"

HORIZON = 24
VALIDATION_HORIZON = 12
TIMESTAMP_INTERVAL = 300

TARGETS = [
    "active_vm_count",
    "aggregate_cpu_utilization",
]

LAGS = [1, 2, 3, 6, 12, 24, 48]
WINDOWS = [3, 6, 12, 24]


def load_workload():
    cpu_path = DATA_DIR / "vm_cpu_readings.csv.gz"

    if not cpu_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {cpu_path.resolve()}"
        )

    cpu = pd.read_csv(cpu_path)

    required_columns = {
        "timestamp",
        "vm_id",
        "avg_cpu",
    }

    missing = required_columns - set(cpu.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    cpu["timestamp"] = pd.to_numeric(
        cpu["timestamp"],
        errors="raise",
    )

    cpu["avg_cpu"] = pd.to_numeric(
        cpu["avg_cpu"],
        errors="raise",
    )

    if not np.isfinite(cpu["timestamp"]).all():
        raise ValueError("Invalid timestamps found.")

    if not np.isfinite(cpu["avg_cpu"]).all():
        raise ValueError("Invalid CPU readings found.")

    if not cpu["avg_cpu"].between(0, 100).all():
        raise ValueError(
            "CPU readings must be between 0 and 100."
        )

    workload = (
        cpu.groupby("timestamp")
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if workload["timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps found after aggregation."
        )

    intervals = workload["timestamp"].diff().dropna()

    if not intervals.eq(TIMESTAMP_INTERVAL).all():
        raise ValueError(
            "Expected consecutive timestamps at 5-minute intervals."
        )

    if (workload["active_vm_count"] < 0).any():
        raise ValueError(
            "Active VM counts cannot be negative."
        )

    if not workload[
        "aggregate_cpu_utilization"
    ].between(0, 100).all():
        raise ValueError(
            "Aggregate CPU utilization must be between 0 and 100."
        )

    return workload


def add_features(frame):
    data = frame.copy()

    for target in TARGETS:
        for lag in LAGS:
            data[f"{target}_lag_{lag}"] = (
                data[target].shift(lag)
            )

        shifted = data[target].shift(1)

        for window in WINDOWS:
            data[f"{target}_rolling_mean_{window}"] = (
                shifted.rolling(window).mean()
            )

            data[f"{target}_rolling_std_{window}"] = (
                shifted.rolling(window).std()
            )

    data["time_index"] = np.arange(len(data))

    data["time_of_day"] = (
        data["timestamp"] // TIMESTAMP_INTERVAL
    ) % 288

    return data


def get_feature_columns(frame):
    excluded = TARGETS + ["timestamp"]

    return [
        column
        for column in frame.columns
        if column not in excluded
    ]


def make_model(name):
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=42,
        )

    raise ValueError(f"Unknown model: {name}")


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError(
            "Actual and predicted values must have the same shape."
        )

    if not np.isfinite(actual).all():
        raise ValueError(
            "Actual values contain non-finite values."
        )

    if not np.isfinite(predicted).all():
        raise ValueError(
            "Predictions contain non-finite values."
        )

    return {
        "MAE": float(
            mean_absolute_error(actual, predicted)
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
    }


def build_forecast_features(history, timestamp):
    row = {}

    for target in TARGETS:
        values = history[target].to_numpy(dtype=float)

        if len(values) < max(LAGS):
            raise ValueError(
                f"At least {max(LAGS)} historical observations "
                "are required."
            )

        for lag in LAGS:
            row[f"{target}_lag_{lag}"] = float(
                values[-lag]
            )

        for window in WINDOWS:
            recent = values[-window:]

            row[f"{target}_rolling_mean_{window}"] = (
                float(np.mean(recent))
            )

            row[f"{target}_rolling_std_{window}"] = (
                float(np.std(recent, ddof=1))
            )

    row["time_index"] = len(history)

    row["time_of_day"] = (
        timestamp // TIMESTAMP_INTERVAL
    ) % 288

    return row


def forecast_recursive(
    history,
    future_timestamps,
    models,
    feature_columns,
):
    history = history[TARGETS].copy().reset_index(drop=True)

    predictions = {
        target: []
        for target in TARGETS
    }

    for timestamp in future_timestamps:
        row = build_forecast_features(
            history,
            timestamp,
        )

        X = pd.DataFrame(
            [row],
            columns=feature_columns,
        )

        next_values = {}

        for target in TARGETS:
            prediction = float(
                models[target].predict(X)[0]
            )

            if not np.isfinite(prediction):
                raise ValueError(
                    f"Non-finite prediction for {target}."
                )

            if target == "active_vm_count":
                prediction = max(0.0, prediction)
                prediction = float(np.rint(prediction))
            else:
                prediction = float(
                    np.clip(prediction, 0.0, 100.0)
                )

            predictions[target].append(prediction)
            next_values[target] = prediction

        history = pd.concat(
            [
                history,
                pd.DataFrame([next_values]),
            ],
            ignore_index=True,
        )

    return {
        target: np.asarray(values, dtype=float)
        for target, values in predictions.items()
    }


def persistence_forecast(history, target, horizon):
    if history.empty:
        raise ValueError(
            "History cannot be empty."
        )

    last_value = float(
        history[target].iloc[-1]
    )

    return np.repeat(
        last_value,
        horizon,
    )


def seasonal_naive_forecast(
    history,
    target,
    horizon,
    season_length=12,
):
    values = history[target].to_numpy(dtype=float)

    if len(values) < season_length:
        raise ValueError(
            "Insufficient history for seasonal-naive forecasting."
        )

    seasonal_values = values[-season_length:]

    return np.resize(
        seasonal_values,
        horizon,
    )


def validate_predictions(output):
    required_columns = [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]

    if list(output.columns) != required_columns:
        raise ValueError(
            f"Prediction columns must be exactly {required_columns}."
        )

    if len(output) != HORIZON:
        raise ValueError(
            f"Expected exactly {HORIZON} prediction rows."
        )

    if output.isna().any().any():
        raise ValueError(
            "Predictions contain missing values."
        )

    if output["timestamp"].duplicated().any():
        raise ValueError(
            "Prediction timestamps must be unique."
        )

    if not output["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Prediction timestamps must be chronological."
        )

    numeric_values = output[
        TARGETS
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Predictions contain non-finite values."
        )

    if (output["active_vm_count"] < 0).any():
        raise ValueError(
            "Active VM predictions cannot be negative."
        )

    if not np.equal(
        output["active_vm_count"],
        np.rint(output["active_vm_count"]),
    ).all():
        raise ValueError(
            "Active VM predictions must be integers."
        )

    if not output[
        "aggregate_cpu_utilization"
    ].between(0, 100).all():
        raise ValueError(
            "CPU predictions must be between 0 and 100."
        )

    return True


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading workload data...")

    workload = load_workload()

    print(f"Total timestamps: {len(workload)}")
    print(
        "First timestamp:",
        workload["timestamp"].iloc[0],
    )
    print(
        "Last timestamp:",
        workload["timestamp"].iloc[-1],
    )

    minimum_required = (
        max(LAGS)
        + VALIDATION_HORIZON
        + HORIZON
        + 1
    )

    if len(workload) < minimum_required:
        raise ValueError(
            "Insufficient data for training, validation, "
            "and final forecasting."
        )

    development = workload.iloc[:-HORIZON].copy()
    heldout = workload.iloc[-HORIZON:].copy()

    feature_data = add_features(
        development
    )

    usable = (
        feature_data
        .dropna()
        .reset_index(drop=True)
    )

    if len(usable) <= VALIDATION_HORIZON:
        raise ValueError(
            "Insufficient rows for chronological validation."
        )

    validation_start = (
        len(usable) - VALIDATION_HORIZON
    )

    train = usable.iloc[
        :validation_start
    ].copy()

    validation = usable.iloc[
        validation_start:
    ].copy()

    if train.empty:
        raise ValueError(
            "Training data is empty."
        )

    if len(validation) != VALIDATION_HORIZON:
        raise ValueError(
            "Unexpected validation window size."
        )

    validation_history_end = (
        int(train.index[-1]) + max(LAGS) + 1
    )

    validation_history = development.iloc[
        :validation_history_end
    ].copy()

    feature_columns = get_feature_columns(
        usable
    )

    report = []

    print(
        f"Development timestamps: {len(development)}"
    )
    print(
        f"Training feature rows: {len(train)}"
    )
    print(
        f"Validation timestamps: {len(validation)}"
    )
    print(
        f"Held-out timestamps: {len(heldout)}"
    )

    print("\nEvaluating models using chronological validation...")

    for target in TARGETS:
        actual = validation[
            target
        ].to_numpy(dtype=float)

        baseline_predictions = {
            "Persistence": persistence_forecast(
                validation_history,
                target,
                VALIDATION_HORIZON,
            ),
            "SeasonalNaive_12": seasonal_naive_forecast(
                validation_history,
                target,
                VALIDATION_HORIZON,
                season_length=12,
            ),
        }

        for model_name, predicted in baseline_predictions.items():
            scores = calculate_metrics(
                actual,
                predicted,
            )

            report.append({
                "target": target,
                "model": model_name,
                **scores,
            })

        trained_models = {}

        X_train = train[
            feature_columns
        ]

        for model_name in [
            "RandomForest",
            "HistGradientBoosting",
        ]:
            model = make_model(
                model_name
            )

            model.fit(
                X_train,
                train[target],
            )

            trained_models[model_name] = model

        for model_name, model in trained_models.items():
            recursive_models = {
                target_name: (
                    model
                    if target_name == target
                    else make_model("RandomForest")
                )
                for target_name in TARGETS
            }

            for other_target in TARGETS:
                if other_target != target:
                    recursive_models[other_target].fit(
                        X_train,
                        train[other_target],
                    )

            predicted = forecast_recursive(
                validation_history,
                validation["timestamp"].to_numpy(),
                recursive_models,
                feature_columns,
            )[target]

            scores = calculate_metrics(
                actual,
                predicted,
            )

            report.append({
                "target": target,
                "model": model_name,
                **scores,
            })

    results = pd.DataFrame(
        report
    )

    results = results.sort_values(
        ["target", "MAE", "RMSE"]
    ).reset_index(drop=True)

    results.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\nCHRONOLOGICAL VALIDATION RESULTS")
    print(
        results.to_string(index=False)
    )

    print(
        "\nSelecting the best model for each target "
        "using validation MAE only..."
    )

    selected_models = {}

    final_features = (
        add_features(development)
        .dropna()
        .reset_index(drop=True)
    )

    final_feature_columns = get_feature_columns(
        final_features
    )

    final_models = {}

    for target in TARGETS:
        target_results = results[
            results["target"] == target
        ].sort_values(
            ["MAE", "RMSE"]
        )

        selected_name = target_results.iloc[0]["model"]

        selected_models[target] = selected_name

        if selected_name in [
            "Persistence",
            "SeasonalNaive_12",
        ]:
            continue

        model = make_model(
            selected_name
        )

        model.fit(
            final_features[final_feature_columns],
            final_features[target],
        )

        final_models[target] = model

    print("\nSELECTED MODELS")

    for target, model_name in selected_models.items():
        print(
            f"{target}: {model_name}"
        )

    predictions_by_target = {}

    for target in TARGETS:
        selected_name = selected_models[target]

        if selected_name == "Persistence":
            predictions_by_target[target] = (
                persistence_forecast(
                    development,
                    target,
                    HORIZON,
                )
            )

        elif selected_name == "SeasonalNaive_12":
            predictions_by_target[target] = (
                seasonal_naive_forecast(
                    development,
                    target,
                    HORIZON,
                    season_length=12,
                )
            )

        else:
            recursive_models = {}

            for target_name in TARGETS:
                if target_name == target:
                    recursive_models[target_name] = (
                        final_models[target_name]
                    )
                elif target_name in final_models:
                    recursive_models[target_name] = (
                        final_models[target_name]
                    )
                else:
                    fallback_model = make_model(
                        "RandomForest"
                    )

                    fallback_model.fit(
                        final_features[
                            final_feature_columns
                        ],
                        final_features[target_name],
                    )

                    recursive_models[target_name] = (
                        fallback_model
                    )

            recursive_predictions = forecast_recursive(
                development,
                heldout["timestamp"].to_numpy(),
                recursive_models,
                final_feature_columns,
            )

            predictions_by_target[target] = (
                recursive_predictions[target]
            )

    output = pd.DataFrame({
        "timestamp": heldout["timestamp"].to_numpy(),
        "active_vm_count": predictions_by_target[
            "active_vm_count"
        ],
        "aggregate_cpu_utilization": predictions_by_target[
            "aggregate_cpu_utilization"
        ],
    })

    output["active_vm_count"] = np.rint(
        output["active_vm_count"].clip(lower=0)
    ).astype(int)

    output["aggregate_cpu_utilization"] = (
        output["aggregate_cpu_utilization"]
        .clip(lower=0, upper=100)
    )

    validate_predictions(
        output
    )

    output_path = OUTPUT_DIR / "predictions.csv"

    output.to_csv(
        output_path,
        index=False,
    )

    print("\nPREDICTIONS PREVIEW")
    print(
        output.head(10).to_string(index=False)
    )

    print(
        "\nFinal output validation: PASSED"
    )

    print(
        f"Saved: {OUTPUT_DIR / 'model_comparison.csv'}"
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Prediction rows: {len(output)}"
    )

    print(
        "\nIMPORTANT: The final 24 target values were excluded "
        "from model training and model selection."
    )


if __name__ == "__main__":
    main()