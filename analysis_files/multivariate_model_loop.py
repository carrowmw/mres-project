from analysis_files.modelling_functions import *

import timeit
import os.path
import numpy as np
import pandas as pd
import torch

# Define the loop parameters
COMPLETENESS_THRESHOLDS = [1, 0.98, 0.96, 0.94]
WINDOW_SIZES = [3, 6, 12, 24, 48, 96, 192]
# WINDOW_SIZES = [1]
HORIZONS = [3, 6, 12, 24, 48, 96]

# Constants
STRIDE = 1
INPUT_INDICES = list(range(0, 11))
TARGET_INDEX = 0
BATCH_SIZE = 8
EPOCHS = 20
ERROR_STD = 4

device = "cpu"


def main():
    total_runtime_start = timeit.default_timer()
    performance_df = pd.DataFrame()  # To store performance metrics
    weights_df = pd.DataFrame()  # To store weights of linear models
    runtime_dict = {}  # To store the runtime for each iteration

    for completeness in COMPLETENESS_THRESHOLDS:
        east_timeseries = load_and_preprocess_data(completeness)

        for horizon in HORIZONS:
            start_time_overall = timeit.default_timer()
            print()
            print(
                f"Completeness: {completeness} | Window Size: {1} | Horizon: {horizon}"
            )

            train_dataloader, test_dataloader, test_targets = prepare_dataloaders(
                data=east_timeseries,
                window_size=1,
                input_feature_indices=INPUT_INDICES,
                target_feature_index=TARGET_INDEX,
                horizon=horizon,
                stride=STRIDE,
                batch_size=1,
                shuffle=False,
            )

            # Baseline Linear Model Evaluation
            linear_model = LinearModel(input_size=len(INPUT_INDICES)).to(device)
            print()
            print("Training Linear Model...")
            start_time = timeit.default_timer()
            (
                linear_weights,
                train_metrics,
                test_metrics,
            ) = train_and_evaluate_linear_model(
                linear_model, train_dataloader, test_dataloader, epochs=EPOCHS
            )
            end_time = timeit.default_timer()
            print()
            print("Evaluating Linear Model...")

            all_predictions = []

            all_Linear_predictions = []
            test_mse, test_rmse = 0, 0

            linear_model.eval()
            criterion = nn.MSELoss()

            for X, y in test_dataloader:
                with torch.inference_mode():
                    X = X.float().to(device)
                    y = y.unsqueeze(-1).unsqueeze(-1).float().to(device)
                    linear_predictions = linear_model(X)
                    all_Linear_predictions.append(linear_predictions.cpu().numpy())
                    # Model prediction and loss calculation
                    loss_MSE = criterion(linear_predictions.unsqueeze(dim=1), y)

                    # Accumulate loss for metrics
                    test_mse += loss_MSE.item()
                    test_rmse += sqrt(loss_MSE.item())

            test_mse /= len(test_dataloader)
            test_rmse /= len(test_dataloader)

            print(f"Validation | Test MSE: {test_mse:.4f} | Test RMSE: {test_rmse:.4f}")

            all_predictions = np.concatenate(all_Linear_predictions, axis=0).flatten()
            all_Linear_predictions = np.array(all_Linear_predictions)
            test_targets_numpy = test_targets.detach().cpu().numpy().flatten()

            print(
                f"Linear | Predictions: {len(all_Linear_predictions)} | Targets: {test_targets_numpy.shape}"
            )
            print()

            weights = linear_model.linear.weight.detach().numpy()
            weights_entry = {
                "Completeness": completeness,
                "WindowSize": 1,
                "Horizon": horizon,
                "Weights": weights,
            }
            weights_df = weights_df._append(weights_entry, ignore_index=True)

            metrics = compute_performance_metrics(
                all_Linear_predictions, test_targets_numpy
            )

            save_model_state(
                linear_model,
                f"Linear | Completeness {completeness} | Window Size 1 | Horizon {horizon}",
            )

            metrics.update(
                {
                    "Model": "Multivariate linear",
                    "Training Time (s)": end_time - start_time,
                    "Completeness": completeness,
                    "WindowSize": 1,
                    "Horizon": horizon,
                }
            )
            performance_df = performance_df._append(metrics, ignore_index=True)

            end_time_overall = timeit.default_timer()
            elapsed_time = end_time_overall - start_time_overall
            key = f"Completeness_{completeness}_WindowSize_{1}_Horizon_{horizon}"
            runtime_dict[key] = elapsed_time
            print(f"Elapsed Time: {elapsed_time:.2f} seconds")

            # Save outputs to CSV
            performance_df.to_csv(
                "performance_metrics_mm.csv",
                mode="a",
                index=False,
                header=not os.path.exists("performance_metrics_mm.csv"),
            )
            weights_df.to_csv(
                "weights_mm.csv",
                mode="a",
                index=False,
                header=not os.path.exists("weights_mm.csv"),
            )
            runtime_df = pd.DataFrame(
                list(runtime_dict.items()), columns=["Parameters", "Runtime (s)"]
            )
            runtime_df.to_csv(
                "runtime_mm.csv",
                mode="w",
                index=False,
                header=not os.path.exists("runtime_mm.csv"),
            )

            # Setup and Train LSTM Model
            for window_size in WINDOW_SIZES:
                start_time_overall = timeit.default_timer()
                print()
                print(
                    f"Completeness: {completeness} | Window Size: {window_size} | Horizon: {horizon}"
                )

                train_dataloader, test_dataloader, test_targets = prepare_dataloaders(
                    east_timeseries,
                    window_size,
                    INPUT_INDICES,
                    TARGET_INDEX,
                    horizon,
                    stride=STRIDE,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                )

                lstm_model = LSTMModel(feature_dim=len(INPUT_INDICES)).to(device)
                print()
                print("Training LSTM Model...")
                start_time = timeit.default_timer()
                train_metrics, test_metrics = train_and_evaluate_lstm_model(
                    lstm_model, train_dataloader, test_dataloader, epochs=EPOCHS
                )
                end_time = timeit.default_timer()
                print()
                print("Evaluating LSTM Model...")

                all_lstm_predictions = []
                test_mse, test_rmse = 0, 0

                linear_model.eval()
                criterion = nn.MSELoss()

                for X, y in test_dataloader:
                    with torch.inference_mode():
                        X = X.float().to(device)
                        y = y.unsqueeze(-1).unsqueeze(-1).float().to(device)
                        lstm_predictions = lstm_model(X)
                        all_lstm_predictions.append(lstm_predictions.cpu().numpy())

                        loss_MSE = criterion(linear_predictions.unsqueeze(dim=1), y)

                        # Accumulate loss for metrics
                        test_mse += loss_MSE.item()
                        test_rmse += sqrt(loss_MSE.item())

                test_mse /= len(test_dataloader)
                test_rmse /= len(test_dataloader)

                print(
                    f"Validation | Test MSE: {test_mse:.4f} | Test RMSE: {test_rmse:.4f}"
                )
                print()
                print(f"LSTM Prediction Type: {type(all_lstm_predictions)}")
                print(f"LSTM Predictions Length: {len(all_lstm_predictions)}")
                print(f"LSTM Predictions Shape: {all_lstm_predictions[0].shape}")
                all_lstm_predictions_np = np.concatenate(
                    all_lstm_predictions
                )  # This will have shape (22*8, 1)
                all_lstm_predictions_flat = (
                    all_lstm_predictions_np.ravel()
                )  # This will flatten the array

                test_targets_numpy = test_targets.detach().cpu().numpy().flatten()
                print(f"LSTM Prediction Type: {type(all_lstm_predictions_flat)}")
                print(f"LSTM Predictions Length: {len(all_lstm_predictions_flat)}")
                print(f"LSTM Predictions Shape: {all_lstm_predictions_flat[0].shape}")

                metrics = compute_performance_metrics(
                    all_lstm_predictions_flat, test_targets_numpy
                )

                save_model_state(
                    lstm_model,
                    f"LSTM | Completeness {completeness} | Window Size {window_size} | Horizon {horizon}",
                )

                metrics.update(
                    {
                        "Model": "Multivariate LSTM",
                        "Training Time (s)": end_time - start_time,
                        "Completeness": completeness,
                        "WindowSize": window_size,
                        "Horizon": horizon,
                    }
                )
                print(
                    f"Type metrics: {type(metrics)}, Type performance_df: {type(performance_df)}"
                )
                performance_df = performance_df._append(metrics, ignore_index=True)

                # Stop the timer for this loop
                end_time_overall = timeit.default_timer()
                elapsed_time = end_time_overall - start_time_overall

                # Store the elapsed time in the dictionary using loop parameters as key
                key = f"Completeness_{completeness}_WindowSize_{window_size}_Horizon_{horizon}"
                runtime_dict[key] = elapsed_time
                print(f"Elapsed Time: {elapsed_time:.2f} seconds")

                # Save outputs to CSV
                performance_df.to_csv(
                    "performance_metrics_mm.csv",
                    mode="a",
                    index=False,
                    header=not os.path.exists("performance_metrics_mm.csv"),
                )
                weights_df.to_csv(
                    "weights_mm.csv",
                    mode="a",
                    index=False,
                    header=not os.path.exists("weights_mm.csv"),
                )
                runtime_df = pd.DataFrame(
                    list(runtime_dict.items()), columns=["Parameters", "Runtime (s)"]
                )
                runtime_df.to_csv(
                    "runtime_mm.csv",
                    mode="w",
                    index=False,
                    header=not os.path.exists("runtime_mm.csv"),
                )
    total_runtime_end = timeit.default_timer()
    total_runtime = total_runtime_end - total_runtime_start
    print(f"Done! Total time elapsed: {total_runtime} seconds")
    return performance_df, weights_df, runtime_df


if __name__ == "__main__":
    main()
