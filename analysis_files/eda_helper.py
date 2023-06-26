from typing import Tuple
import seaborn as sns
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler


def get_custom_palette():
    """
    Generate and return a predefined seaborn color palette.

    This function creates a color palette with predefined colors, sets it as the current
    seaborn palette and then returns the created palette. The returned palette is a list
    of RGB tuples, where each tuple represents a color.

    Returns:
        list: A list of RGB tuples defining the color palette.
    """
    colors = [
        "#4060AF",
        "#FF5416",
        "#FDC82F",
        "#00B2A9",
        "#E7E6E6",
        "#93509E",
        "#00A9E0",
        "#CF0071",
    ]
    sns.set_palette(colors)
    return sns.color_palette(colors)


def get_custom_colormap():
    """
    Return a custom matplotlib colormap.

    This function creates a custom colormap from a predefined set of colors.

    Returns:
        LinearSegmentedColormap: A colormap object that can be used in plotting functions.
    """
    colors = ["#22335C", "#00B2A9", "#FDC82F"]
    return LinearSegmentedColormap.from_list("custom_colormap", colors)


def aggregate_and_preprocess_daily_data(
    df: pd.DataFrame, print_transform_steps: str = False
) -> Tuple[np.ndarray, pd.Index, pd.Index]:
    """
    Aggregate data on a daily basis, extract statistical features, remove NaN values,
    and standardize the data.

    This function first groups the dataframe by date and aggregates the values. It then
    resamples the data on a daily basis, computing the mean and standard deviation for
    each day. It drops any days that have NaN values, likely as a result of there being no
    data for those days in the original dataframe. Finally, it standardizes the data by
    subtracting the mean and dividing by the standard deviation.

    Args:
        df: The input pandas DataFrame. It's expected to have a datetime column named 'dt'
           and a 'value' column.

    Returns:
        A tuple containing three elements:
        - A 2D numpy array that's been aggregated on a daily basis, had statistical
          features extracted, and NaN values removed, and has been standardized.
        - The index from the DataFrame after removing NaN values.
        - The column names from the DataFrame after removing NaN values.

    Prints the number of data points at each step of the processing.
    """
    # aggregate on datetime to combine different trajectories
    agg_df = df.groupby("dt").agg({"value": "sum"})

    # statistical feature extraction and resampling
    resampled_df = pd.DataFrame()
    resampled_df["mean"] = agg_df.resample("d").mean()
    resampled_df["std"] = agg_df.resample("d").std()

    # remove NaN values from dataframe
    removed_df = resampled_df.dropna()

    # scale dataframe
    scaled_data = StandardScaler().fit_transform(removed_df)

    # print transformation steps
    if print_transform_steps is True:
        print(f"Length of original dataset: {len(df)}")
        print(f"Length of aggregated dataset: {len(agg_df)}")
        print(f"Length of resampled dataset: {len(resampled_df)}")
        print(f"NaN values removed: {len(resampled_df) - len(removed_df)}")
        print(f"Final length: {len(scaled_data)}")

    return scaled_data, removed_df.index, removed_df.columns


def sliding_windows_array(data, window_size, horizon=1, stride=1):
    inputs = []
    targets = []
    for i in range(0, len(data) - window_size - horizon + 1, stride):
        input_data = data[i : i + window_size]
        target_data = data[i + window_size : i + window_size + horizon]
        if i == 0:
            print(
                f"Input shape: {input_data.shape} | Target shape: {target_data.shape}"
            )
        inputs.append(input_data)
        targets.append(target_data)
    return np.array(inputs), np.array(targets).reshape(-1, horizon)


def sliding_windows_tensor(data, window_size, horizon=1, stride=1):
    inputs = []
    targets = []
    for i in range(0, len(data) - window_size - horizon + 1, stride):
        input_data = data[i : i + window_size]
        target_data = data[i + window_size : i + window_size + horizon]
        if i == 0:
            print(
                f"Input shape: {input_data.shape} | Target shape: {target_data.shape}"
            )
        inputs.append(input_data)
        targets.append(target_data)
    return torch.tensor(inputs), torch.tensor(targets)


def plot_windows(inputs, predictions, targets, num_plots=5, step=1):
    custom_palette = get_custom_palette()
    num_plots = min(num_plots, len(inputs))
    start_idx = np.random.choice(len(inputs) - num_plots)

    fig, axs = plt.subplots(num_plots, 1, figsize=(6, 2 * num_plots))

    max_y = inputs.min()

    for i in range(num_plots):
        idx = start_idx + step * i
        axs[i].plot(
            range(len(inputs[idx])),
            inputs[idx],
            label="Inputs",
            color=custom_palette[0],
        )
        axs[i].scatter(
            range(len(inputs[idx]), len(inputs[idx]) + len(predictions[idx])),
            predictions[idx],
            label="Predictions",
            color=custom_palette[1],
            marker="x",
            s=60,
        )
        axs[i].scatter(
            range(len(inputs[idx]), len(inputs[idx]) + len(targets[idx])),
            targets[idx],
            label="Targets",
            color=custom_palette[0],
        )
        # Update max_y if the max of the current input or prediction is greater
        max_y = max(max_y, inputs[idx].max(), targets[idx].max())
        axs[i].set_ylim(0, max_y)
        if i == 0:
            axs[i].legend()

    plt.tight_layout()
    plt.show()
