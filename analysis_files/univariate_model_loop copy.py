import timeit
import os.path
import numpy as np
import pandas as pd
import torch.nn as nn
from analysis_files.modelling_functions import (
    load_and_preprocess_data,
    evaluate_linear_model,
    evaluate_lstm_model,
)

# Define constants
COMPLETENESS_THRESHOLDS = [1, 0.98, 0.96, 0.94]
COMPLETENESS_THRESHOLDS = [1]
WINDOW_SIZES = [3, 6, 12, 24, 48, 96]
WINDOW_SIZES = [3]
HORIZONS = [3, 6, 12, 24]
HORIZONS = [3]
STRIDE = 1
INPUT_INDEX = 0
TARGET_INDEX = 0
BATCH_SIZE = 8
EPOCHS = 50
ERROR_STD = 4
device = "cpu"


def main():
    runtime_dict = {}  # To store the runtime for each iteration

    for completeness in COMPLETENESS_THRESHOLDS:
        # Load and preprocess data
        east_df, _ = load_data()
        east_dict = preprocess_data(east_df, completeness, show_prints=False)
        east_timeseries = east_dict["data"]

        for window_size in WINDOW_SIZES:
            for horizon in HORIZONS:
                start_time = timeit.default_timer()
                print(
                    f"Completeness: {completeness} | Window Size: {window_size} | Horizon: {horizon}"
                )

                inputs, targets = sliding_windows(
                    east_timeseries,
                    window_size,
                    INPUT_INDEX,
                    TARGET_INDEX,
                    horizon,
                    STRIDE,
                )

                # Baseline Model Evaluation
                start_time = timeit.default_timer()
                baseline_model = BaselineModel()
                baseline_predictions = baseline_model(inputs.squeeze())
                end_time = timeit.default_timer()

                metrics = compute_performance_metrics(baseline_predictions, targets)
                metrics.update(
                    {
                        "Model": "Baseline",
                        "Training Time (s)": end_time - start_time,
                        "Completeness": completeness,
                        "WindowSize": window_size,
                        "Horizon": horizon,
                    }
                )
                performance_df = pd.DataFrame([metrics])

                # Setup and Train LSTM Model
                model = LSTMModel().to(device)
                train_size = int(0.8 * len(inputs))
                train_inputs, test_inputs = inputs[:train_size], inputs[train_size:]
                train_targets, test_targets = targets[:train_size], targets[train_size:]
                train_dataset = TimeSeriesDataset(train_inputs, train_targets)
                test_dataset = TimeSeriesDataset(test_inputs, test_targets)
                train_dataloader = DataLoader(
                    train_dataset, batch_size=BATCH_SIZE, shuffle=False
                )
                test_dataloader = DataLoader(
                    test_dataset, batch_size=BATCH_SIZE, shuffle=False
                )

                start_time = timeit.default_timer()
                train_metrics, test_metrics = train_lstm_model(
                    model, train_dataloader, test_dataloader
                )
                end_time = timeit.default_timer()

                train_metrics_df = pd.DataFrame(
                    train_metrics, columns=["Epoch", "MSE", "RMSE"]
                )
                test_metrics_df = pd.DataFrame(
                    test_metrics, columns=["Epoch", "MSE", "RMSE"]
                )

                # Prediction and Error Calculation for LSTM
                model.eval()
                with torch.no_grad():
                    predictions = model(inputs.to(device).unsqueeze(dim=-1).float())
                errors = (
                    (predictions.squeeze(dim=-1) - targets.squeeze(dim=-1))
                    .abs()
                    .numpy()
                )

                metrics = compute_performance_metrics(predictions, targets)
                metrics.update(
                    {
                        "Model": "LSTM",
                        "Training Time (s)": end_time - start_time,
                        "Completeness": completeness,
                        "WindowSize": window_size,
                        "Horizon": horizon,
                    }
                )

                metrics_df = pd.DataFrame([metrics])  # Convert metrics to a DataFrame
                # Calculate anomaly threshold based on error statistics
                threshold = errors.mean() + ERROR_STD * errors.std()

                # Detect anomalies based on error threshold
                anomalies = errors > threshold
                total_anomalies = np.sum(anomalies)

                # Calculate the anomaly percentage
                anomaly_percentage = (total_anomalies / len(targets)) * 100

                # Add the columns to train_metrics_df and test_metrics_df
                train_metrics_df["Total_Anomalies"] = total_anomalies
                train_metrics_df["Anomaly_Percentage"] = anomaly_percentage

                test_metrics_df["Total_Anomalies"] = total_anomalies
                test_metrics_df["Anomaly_Percentage"] = anomaly_percentage

                key = f"Completeness_{completeness}_WindowSize_{window_size}_Horizon_{horizon}"
                metrics_dict[key] = {
                    "Train Metrics": train_metrics_df,
                    "Test Metrics": test_metrics_df,
                }

                # Add loop parameters to metrics and append to performance_df
                metrics["Completeness"] = completeness
                metrics["WindowSize"] = window_size
                metrics["Horizon"] = horizon
                performance_df = pd.concat(
                    [performance_df, metrics_df], ignore_index=True
                )

                # Stop the timer for this iteration
                end_time = timeit.default_timer()
                elapsed_time = end_time - start_time

                # Store the elapsed time in the dictionary using loop parameters as key
                key = f"Completeness_{completeness}_WindowSize_{window_size}_Horizon_{horizon}"
                runtime_dict[key] = elapsed_time
                print(f"Elapsed Time: {elapsed_time:.2f} seconds")

                print(
                    f"Finished Loop {len(runtime_dict)}/{len(COMPLETENESS_THRESHOLDS)*len(WINDOW_SIZES)*len(HORIZONS)}"
                )

                performance_df.to_csv(
                    "performance_metrics_um.csv",
                    mode="a",
                    index=False,
                    header=not os.path.exists("performance_metrics_um.csv"),
                )
                runtime_df = pd.DataFrame(
                    list(runtime_dict.items()), columns=["Parameters", "Runtime (s)"]
                )
                runtime_df.to_csv(
                    "runtime_um.csv",
                    mode="w",
                    index=False,
                    header=not os.path.exists("runtime_um.csv"),
                )

    return performance_df, runtime_df


if __name__ == "__main__":
    main()
