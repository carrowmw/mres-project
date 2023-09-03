import timeit
import os.path
import numpy as np
import pandas as pd
import torch.nn as nn
from modelling_functions import (
    load_and_preprocess_data,
    evaluate_linear_model,
    evaluate_lstm_model,
)

# Define constants
COMPLETENESS_THRESHOLDS = [1, 0.98, 0.96, 0.94]
# COMPLETENESS_THRESHOLDS = [1]
WINDOW_SIZES = [3, 6, 12, 24, 48, 96]
# WINDOW_SIZES = [3]
HORIZONS = [1, 3, 6, 12, 24]
# HORIZONS = [3]
STRIDE = 1
INPUT_INDICES = [0]
TARGET_INDEX = 0
BATCH_SIZE = 8
EPOCHS = 50
ERROR_STD = 4
device = "cpu"
model_input_type = "mm" if len(INPUT_INDICES) > 1 else "uv"


def main():
    """
    Evaluates linear and LSTM models across different thresholds and horizons.

    For each combination of completeness threshold and horizon, this function:
    1. Loads the data associated with the given completeness.
    2. Evaluates the linear model and logs the weights and performance metrics.
    3. Evaluates the LSTM model for different window sizes and logs the performance metrics.

    Performance metrics, weights, and runtime information are saved in CSV files:
    - `performance_metrics.csv`: Contains performance metrics for both models.
    - `weights.csv`: Contains weights information from the linear model.
    - `runtime_metrics.csv`: Contains runtime information for each combination.

    Existing CSV files will be cleaned at the beginning of the function execution.

    Note:
    This function depends on global constants `COMPLETENESS_THRESHOLDS`, `HORIZONS`, and `WINDOW_SIZES`.
    It also calls external functions `load_data`, `evaluate_linear_model`, and `evaluate_lstm_model`.

    Returns:
        None
    """

    try:
        total_runtime_start = timeit.default_timer()

        # If the files exist, remove them
        if os.path.exists(f"performance_metrics_{model_input_type}.csv"):
            os.remove(f"performance_metrics_{model_input_type}.csv")

        if os.path.exists(f"weights_{model_input_type}.csv"):
            os.remove(f"weights_{model_input_type}.csv")

        if os.path.exists(f"runtime_metrics_{model_input_type}.csv"):
            os.remove(f"runtime_metrics_{model_input_type}.csv")

        if os.path.exists(f"linear_test_metrics_{model_input_type}.csv"):
            os.remove(f"linear_test_metrics_{model_input_type}.csv")

        if os.path.exists(f"linear_train_metrics_{model_input_type}.csv"):
            os.remove(f"linear_train_metrics_{model_input_type}.csv")

        if os.path.exists(f"lstm_test_metrics_{model_input_type}.csv"):
            os.remove(f"lstm_test_metrics_{model_input_type}.csv")

        if os.path.exists(f"lstm_train_metrics_{model_input_type}.csv"):
            os.remove(f"lstm_train_metrics_{model_input_type}.csv")

        # Now, initialize empty DataFrames
        performance_df = pd.DataFrame()
        weights_df = pd.DataFrame()
        total_iterations = 0

        for completeness in COMPLETENESS_THRESHOLDS:
            east_timeseries = load_and_preprocess_data(completeness)
            sequence_lengths = list(
                range(
                    int(len(east_timeseries) / 4),
                    int(len(east_timeseries)) + int(len(east_timeseries) / 4),
                    int(len(east_timeseries) / 4),
                )
            )
            print("New Completeness")
            print()
            for sequence_length in sequence_lengths:
                print("     New Sequence Length")
                print()
                east_timeseries = east_timeseries[:sequence_length]

                for horizon in HORIZONS:
                    print("         New Horizon")
                    print()
                    start_time_overall = timeit.default_timer()
                    start_time_linear = timeit.default_timer()

                    total_iterations += 1
                    print(
                        f"Iteration: {total_iterations} / {(len(HORIZONS) * len(COMPLETENESS_THRESHOLDS) * (len(WINDOW_SIZES) + 1)* len(sequence_lengths))}"
                    )
                    # Linear Model Evaluation
                    weights, linear_metrics = evaluate_linear_model(
                        data=east_timeseries,
                        completeness=completeness,
                        sequence_length=sequence_length,
                        horizon=horizon,
                        input_feature_indices=INPUT_INDICES,
                        target_feature_index=TARGET_INDEX,
                        epochs=100,
                        stride=STRIDE,
                        total_iterations=total_iterations,
                    )

                    weights_entry = {
                        "Completeness": completeness,
                        "Sequence Length": sequence_length,
                        "Horizon": horizon,
                        "WindowSize": 1,
                        "Weights": weights,
                        "Sequence": sequence_length,
                        "Test Number": total_iterations,
                    }
                    weights_df = weights_df._append(weights_entry, ignore_index=True)
                    weights_df.to_csv(
                        f"weights_{model_input_type}.csv",
                        mode="w",
                        header=not os.path.exists(f"weights_{model_input_type}.csv"),
                        index=False,
                    )

                    performance_df = performance_df._append(
                        linear_metrics, ignore_index=True
                    )
                    print()
                    print("Linear Model Metrics:")
                    print(pd.DataFrame([linear_metrics], columns=linear_metrics.keys()))
                    print()
                    print()
                    print("Performance DF:")
                    print(performance_df)

                    performance_df.to_csv(
                        f"performance_metrics_{model_input_type}.csv",
                        mode="w",
                        header=not os.path.exists(
                            f"performance_metrics_{model_input_type}.csv"
                        ),
                        index=False,
                    )

                    end_time_linear = timeit.default_timer()
                    elapsed_time = end_time_linear - start_time_linear
                    print()
                    print(f"Elapsed Time: {elapsed_time}")
                    with open(
                        f"runtime_metrics_{model_input_type}.csv", "a", encoding="utf8"
                    ) as f:
                        f.write(
                            f"Completeness_{completeness}_Sequence{sequence_length}_Horizon_{horizon}_WindowSize_{1}_Linear,{elapsed_time}\n"
                        )

                    # LSTM Model Evaluation
                    for window_size in WINDOW_SIZES:
                        print("             New Window Size")
                        total_iterations += 1
                        print(
                            f"Iteration: {total_iterations} / {(len(HORIZONS) * len(COMPLETENESS_THRESHOLDS) * (len(WINDOW_SIZES) + 1) * len(sequence_lengths))}"
                        )
                        start_time_lstm = timeit.default_timer()

                        lstm_metrics = evaluate_lstm_model(
                            data=east_timeseries,
                            completeness=completeness,
                            sequence_length=sequence_length,
                            horizon=horizon,
                            window_size=window_size,
                            input_feature_indices=INPUT_INDICES,
                            target_feature_index=TARGET_INDEX,
                            epochs=50,
                            total_iterations=total_iterations,
                        )

                        performance_df = performance_df._append(
                            lstm_metrics, ignore_index=True
                        )

                        print()
                        print("Linear Model Metrics:")
                        print(
                            pd.DataFrame(
                                [linear_metrics], columns=linear_metrics.keys()
                            )
                        )
                        print()
                        print()
                        print("Performance DF:")
                        print(performance_df)

                        performance_df.to_csv(
                            f"performance_metrics_{model_input_type}.csv",
                            mode="w",
                            header=not os.path.exists(
                                f"performance_metrics_{model_input_type}.csv"
                            ),
                            index=False,
                        )
                        end_time_lstm = timeit.default_timer()
                        elapsed_time = end_time_lstm - start_time_lstm

                        print()
                        print(f"Elapsed Time: {elapsed_time}")

                        with open(
                            f"runtime_metrics_{model_input_type}.csv",
                            "a",
                            encoding="utf8",
                        ) as f:
                            f.write(
                                f"Completeness_{completeness}_Sequence{sequence_length}_Horizon_{horizon}_WindowSize_{window_size}_LSTM,{elapsed_time}\n"
                            )
                    end_time_overall = timeit.default_timer()
                    elapsed_time = end_time_overall - start_time_overall

                    with open(
                        f"runtime_metrics_{model_input_type}.csv", "a", encoding="utf8"
                    ) as f:
                        f.write(
                            f"Completeness_{completeness}_Horizon_{horizon}_Overall,{elapsed_time}\n"
                        )

            total_runtime_end = timeit.default_timer()
            print()
            print()
            print()
            print(f"Total Runtime: {total_runtime_end - total_runtime_start}")

    except KeyboardInterrupt:
        print("Keyboard Interrupted")
        total_runtime_end = timeit.default_timer()
        print(f"Total Runtime: {(total_runtime_end - total_runtime_start):.2f}s")


if __name__ == "__main__":
    main()
