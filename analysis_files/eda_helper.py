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


def get_custom_heatmap():
    """
    Return a custom matplotlib colormap.

    This function creates a custom colormap from a predefined set of colors.

    Returns:
        LinearSegmentedColormap: A colormap object that can be used in plotting functions.
    """
    colors = ["#00A9E0", "#CF0071"]
    return LinearSegmentedColormap.from_list("custom_colormap", colors)


# 1
def remove_directionality(
    df: pd.DataFrame, additional_features: list = None
) -> pd.DataFrame:
    """
    Removes directionality in the data by aggregating (summing) values with the same timestamp.

    This function groups the DataFrame by timestamp ('dt'), sums the 'value' for each group,
    effectively removing any directionality in the data. This operation might be useful for
    data sets where the 'value' depends on a directional parameter and we want to consider
    the total amount regardless of the direction.

    Args:
        df (pd.DataFrame): Input DataFrame with a 'dt' column for datetime and a 'value' column.
        additional_features (list): Additional features to include in the aggregation.

    Returns:
        pd.DataFrame: DataFrame with directionality removed, indexed by 'dt' and with summed 'value'.
    """
    # Define a base dictionary for aggregation
    agg_dict = {"value": "sum", "date": "first"}

    # If there are additional features, add them to the dictionary
    if additional_features:
        for feature in additional_features:
            agg_dict[feature] = "first"  # you can modify this according to your needs

    df = df.groupby("dt").agg(agg_dict)
    return df


# 2
def select_daily_completeness_threshold(
    df: pd.DataFrame, completeness: float, directionality=1, show_prints: bool = True
) -> pd.DataFrame:
    """
    Select data from a sensor dataframe based on completeness threshold.

    Args:
    df (pandas.DataFrame): The dataframe.
    completeness (float): The completeness threshold (ranging from 0 to 1).
    directionality (int, optional): The directionality factor (1 or -1). Default is 1.
    show_prints (bool, optional): Whether to show print statements. Default is True.

    Returns:
    df_selected (pandas.DataFrame): The selected dataframe based on the completeness threshold.
    df__selected (pandas.DataFrame): The selected  dataframe based on the completeness threshold.
    """
    hours = 24  # Number of hours in a day
    periods = 4  # Number of periods in an hour (15 minutes)

    threshold = completeness * directionality * hours * periods

    # Group by the new 'date' column and count the number of entries for each date
    date_counts = df.groupby("date").size()

    # Find the dates that have at least the threshold number of entries
    valid_dates = date_counts[date_counts >= threshold].index

    # Select only the rows that have a date in valid_dates
    df_selected = df[df["date"].isin(valid_dates)]

    if show_prints:
        print(f"Removing incomplete days...")
        print(f"Initial number of records: {len(df)}")
        print(
            f"Number of records in days @ {completeness * 100:.0f}% completeness: {len(df_selected)}"
        )
        print(
            f"Proportion of records removed: {(1 - len(df_selected)/len(df))*100:.2f}%"
        )

    return df_selected


# 3.1
def check_completeness_daily(
    df: pd.DataFrame,
    df_dayofyear: int,
    df_year: int,
    day_number: int,
    days: int,
    year: int,
    completeness: float,
) -> pd.DataFrame:
    """
    Check if data completeness for each day in a sequence meets the threshold.

    Args:
    df (DataFrame): The DataFrame containing the data
    df_dayofyear (Series): A Series representing the day of the year for each data point
    df_year (Series): A Series representing the year for each data point
    day_number (int): The starting day of the sequence
    days (int): The number of days in the sequence
    year (int): The year of the sequence
    completeness (float): The completeness threshold

    Returns:
    bool: True if the completeness meets the threshold for each day, False otherwise
    """
    for i in range(days):
        ts_data = df[(df_dayofyear == day_number + i) & (df_year == year)]
        if len(ts_data) < completeness * 24 * 4:
            return False
    return True


# 3.2
def find_longest_sequence(
    df: pd.DataFrame, completeness: float, show_prints: bool = True
) -> pd.DataFrame:
    """
    Find the longest sequence of consecutive days where data completeness meets the threshold.

    Args:
    df (DataFrame): The DataFrame containing the data
    completeness (float): The completeness threshold
    show_prints (bool, optional): Whether to show print statements. Default is True.

    Returns:
    DataFrame: A DataFrame containing the longest sequence
    """
    df_dayofyear = df.index.to_series().dt.dayofyear
    df_year = df.index.to_series().dt.year
    unique_years = df_year.unique()

    days = 0  # Initialize days counter
    max_day_sequence_start = None  # Initialize starting day of max sequence
    max_sequence_year = None  # Initialize year of max sequence

    while True:
        for year in unique_years:
            min_day_number = df_dayofyear[df_year == year].min()
            max_day_number = df_dayofyear[df_year == year].max()
            # Check each possible sequence starting from each day of the year
            for day_number in range(min_day_number, max_day_number - days + 1):
                if check_completeness_daily(
                    df, df_dayofyear, df_year, day_number, days, year, completeness
                ):
                    days += 1  # Increase the days counter
                    max_day_sequence_start = (
                        day_number  # Update starting day of max sequence
                    )
                    max_sequence_year = year  # Update year of max sequence
                    break
            else:
                continue
            break
        else:
            if days > 0:
                sequence_df = df[
                    (df_dayofyear >= max_day_sequence_start)
                    & (df_dayofyear < max_day_sequence_start + days)
                    & (df_year == max_sequence_year)
                ]
                if show_prints:
                    print(f"Maximum consecutive days: {days - 1}")
                    print(
                        f"Starting from day number {max_day_sequence_start} in {max_sequence_year}"
                    )
                return sequence_df
            else:
                if show_prints:
                    print("No consecutive days found.")
                return None  # If no sequence was found, return None


# 4
def add_term_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds term-related features 'newcastle_term' and 'northumbria_term' to the input DataFrame based on date ranges.

    Args:
        df (pandas.DataFrame): The DataFrame to which the new features will be added. It should have a datetime index.

    Returns:
        pandas.DataFrame: The input DataFrame with additional columns 'newcastle_term' and 'northumbria_term'.
                          These columns represent binary indicators for the terms of each university.

    Note:
        The function assumes that the input DataFrame `df` has a datetime index. Additionally, the term start and end
        dates for Newcastle and Northumbria universities are hardcoded in the function, so any changes to the term
        dates should be made directly in the function code.
    """
    # Make a copy of the input DataFrame to avoid modifying it
    df = df.copy()
    # Define the date range for the series
    start = min(df.index.min(), df.index.min())
    end = max(df.index.max(), df.index.max())
    date_range = pd.date_range(start=start, end=end, freq="15T")

    # Define the start and end dates for each term
    newcastle_term_dates_2122 = [
        ("2021-09-20", "2021-12-17"),
        ("2022-01-10", "2022-03-25"),
        ("2022-04-25", "2022-06-17"),
    ]
    newcastle_term_dates_2223 = [
        ("2022-09-19", "2022-12-16"),
        ("2023-01-09", "2023-03-24"),
        ("2023-04-24", "2023-06-16"),
    ]
    northumbria_term_dates_2122 = [
        ("2021-09-20", "2021-12-17"),
        ("2022-01-10", "2022-04-01"),
        ("2022-04-25", "2022-05-27"),
    ]
    northumbria_term_dates_2223 = [
        ("2022-09-19", "2022-12-16"),
        ("2023-01-09", "2023-03-24"),
        ("2023-04-17", "2023-06-02"),
    ]

    # Create binary series for each term of each university
    newcastle_2122 = [
        date_range.to_series().between(start, end).astype(int)
        for start, end in newcastle_term_dates_2122
    ]
    newcastle_2223 = [
        date_range.to_series().between(start, end).astype(int)
        for start, end in newcastle_term_dates_2223
    ]
    northumbria_2122 = [
        date_range.to_series().between(start, end).astype(int)
        for start, end in northumbria_term_dates_2122
    ]
    northumbria_2223 = [
        date_range.to_series().between(start, end).astype(int)
        for start, end in northumbria_term_dates_2223
    ]

    # Combine the binary series for each university into a single series
    newcastle = pd.concat(newcastle_2122 + newcastle_2223, axis=1).max(axis=1)
    northumbria = pd.concat(northumbria_2122 + northumbria_2223, axis=1).max(axis=1)

    # Add the new features to the input DataFrame df
    df["newcastle_term"] = newcastle.astype(bool)
    df["northumbria_term"] = northumbria.astype(bool)

    return df


# 5
def add_periodicity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates frequency features for time series data and adds these features to the input DataFrame.
    The function generates sine and cosine features based on daily, half-day, quarter-yearly,
    and yearly periods to capture potential cyclical patterns.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with a DatetimeIndex containing the timestamps for which frequency features are to be created.

    Returns
    -------
    df : pandas DataFrame
        The input DataFrame, with the following new columns:
        - 'sin_day': Sine of the time of day, assuming a period of 24 hours.
        - 'cos_day': Cosine of the time of day, assuming a period of 24 hours.
        - 'sin_half_day': Sine of the time of day, assuming a period of 12 hours.
        - 'cos_half_day': Cosine of the time of day, assuming a period of 12 hours.
        - 'sin_quarter': Sine of the day of the year, assuming a period of about 91.25 days.
        - 'cos_quarter': Cosine of the day of the year, assuming a period of about 91.25 days.
        - 'sin_year': Sine of the day of the year, assuming a period of 365 days.
        - 'cos_year': Cosine of the day of the year, assuming a period of 365 days.
    """
    # Make a copy of the input DataFrame to avoid modifying it
    df = df.copy()
    dt_index = df.index
    df["sin_day"] = np.sin(2 * np.pi * dt_index.hour / 24)
    df["cos_day"] = np.cos(2 * np.pi * dt_index.hour / 24)

    df["sin_half_day"] = np.sin(2 * np.pi * dt_index.hour / 12)
    df["cos_half_day"] = np.cos(2 * np.pi * dt_index.hour / 12)

    df["sin_quarter"] = np.sin(2 * np.pi * dt_index.dayofyear / 91.25)
    df["cos_quarter"] = np.cos(2 * np.pi * dt_index.dayofyear / 91.25)

    df["sin_year"] = np.sin(2 * np.pi * dt_index.dayofyear / 365)
    df["cos_year"] = np.cos(2 * np.pi * dt_index.dayofyear / 365)
    return df


# 6
def scale_data(df: pd.DataFrame) -> np.ndarray:
    """
    Scales the 'value' column of the input DataFrame using StandardScaler from sklearn.preprocessing.

    This function standardizes the feature by removing the mean and scaling to unit variance.
    The standard score of a sample x is (x - u) / s where u is the mean of the training samples,
    and s is the standard deviation of the training samples.

    Args:
    df : pd.DataFrame
        DataFrame with the data to be scaled.

    Returns:
    scaled_df : numpy.ndarray
        The scaled version of the input data as a numpy array.
    """

    df = df.copy()
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    scaler = StandardScaler()
    df["value"] = scaler.fit_transform(df["value"].values.reshape(-1, 1))

    scaled_df = df.to_numpy().astype(np.float64)

    return scaled_df


# 7
def resample_frequency(df: pd.DataFrame, frequency: str = None) -> pd.DataFrame:
    """
    Resamples a DataFrame using a given frequency, and calculates mean and standard deviation for each resampled period.

    This function resamples a DataFrame on the provided frequency (resample_frequency) and computes the mean and
    standard deviation for each period. It's primarily used to resample time-series data to a lower frequency (downsample),
    while summarizing the resampled data with its mean and standard deviation.

    If no resample frequency is provided, the function will simply return the original DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame expected to have a DateTimeIndex.
        resample_frequency (str, optional): The frequency for resampling the data.
                                             This string can be any valid frequency alias in pandas.
                                             Default is None.

    Returns:
        pd.DataFrame: A DataFrame with the DateTimeIndex resampled to the provided frequency,
                      and the 'mean' and 'std' columns representing the mean and standard deviation
                      of the 'value' in each period respectively.
    """
    resampled_df = pd.DataFrame()
    resampled_df["mean"] = df["value"].resample(frequency).mean()
    resampled_df["std"] = df["value"].resample(frequency).std()

    return resampled_df


def preprocess_data(
    df: pd.DataFrame,
    completeness_threshold: float,
    frequency: str = None,
    additional_features: list = None,
    show_prints=True,
    remove_dir=True,  # 1
    daily_completeness=True,  # 2
    consecutive_days=True,  # 3
    term_dates=True,  # 4
    periodicity=True,  # 5
    scale=True,  # 6
    resample=False,  # 7
) -> dict:
    """
    Function to perform several preprocessing steps on the data.

    Args:
        df: Input DataFrame. Should have a datetime index.
        remove_directionality: If True, removes the directionality from the data.
        resample_frequency: String specifying the frequency for resampling the data. If None, no resampling is performed.
        completeness_threshold: Float representing the completeness threshold. If None, no completeness threshold is applied.
        consecutive_days: Int, if provided, only the data with these many consecutive days are kept.
        scale: If True, scales the data using StandardScaler.
        term_dates: DataFrame with term dates.
        add_periodicity: If True, adds periodicity to the data.

    Returns:
        dict: The preprocessed data, df.index, and df.columns as a dictionary.
    """

    if remove_dir:
        df = remove_directionality(df, additional_features)

    if daily_completeness:
        df = select_daily_completeness_threshold(
            df, completeness_threshold, show_prints=show_prints
        )

    if consecutive_days:
        df = find_longest_sequence(df, completeness_threshold, show_prints=show_prints)

    if term_dates:
        df = add_term_dates(df)

    if periodicity:
        df = add_periodicity_features(df)

    if scale:
        scaled_data = scale_data(df)

    if resample:
        df = resample_frequency(df, frequency)
        return {"data": df, "index": df.index, "columns": df.columns}
    else:
        df.drop(columns=["date"], inplace=True)
        return {"data": scaled_data, "index": df.index, "columns": df.columns}


def sliding_windows(
    data,
    window_size,
    input_feature_indices,
    target_feature_index,
    horizon=1,
    stride=1,
    shapes=False,
):
    """
    Generate sliding windows from the given data for sequence learning.

    Parameters:
    data (array-like): The time-series data to generate windows from.
    window_size (int): The size of the sliding window to use.
    input_feature_indices (list of ints): Indices of the features to be used as input.
    target_feature_index (int): Index of the feature to be predicted.
    horizon (int, optional): The prediction horizon. Defaults to 1.
    stride (int, optional): The stride to take between windows. Defaults to 1.

    Returns:
    tuple: A tuple containing the inputs and targets as torch tensors.

    """
    inputs = []
    targets = []
    for i in range(0, len(data) - window_size - horizon + 1, stride):
        input_data = data[
            i : i + window_size, input_feature_indices
        ]  # selects only the features indicated by input_feature_indices
        target_data = data[
            i + window_size + horizon - 1, target_feature_index
        ]  # selects the feature indicated by target_feature_index, horizon steps ahead
        if i == 0 and shapes:
            print(
                f"Input shape: {input_data.shape} | Target shape: {target_data.shape}"
            )
        inputs.append(input_data)
        targets.append(target_data)

    # Convert lists of numpy arrays to numpy arrays
    inputs = np.array(inputs)
    targets = np.array(targets)

    return torch.tensor(inputs), torch.tensor(targets)


def plot_windows(inputs, predictions, targets, horizon, num_plots=5, step=1, title=""):
    """
    Plot a random selection of the given inputs, predictions, and targets.

    Parameters:
    inputs (array-like): The input data.
    predictions (array-like): The predicted values.
    targets (array-like): The actual target values.
    horizon (int): The prediction horizon.
    num_plots (int, optional): The number of plots to make. Defaults to 5.
    step (int, optional): The step to take between plots. Defaults to 1.

    Returns:
    None
    """
    custom_palette = get_custom_palette()
    num_plots = min(num_plots, len(inputs))
    start_idx = np.random.choice(len(inputs) - num_plots)

    # Get the global minimum and maximum y-values for the entire inputs dataset
    min_y = -1
    max_y = 4

    fig, axs = plt.subplots(num_plots, 1, figsize=(6, 2 * num_plots))
    for i in range(num_plots):
        idx = start_idx + step * i
        axs[i].plot(
            range(len(inputs[idx])),
            inputs[idx],
            label="Inputs",
            color=custom_palette[0],
        )
        axs[i].scatter(
            range(
                len(inputs[idx]) + horizon,
                len(inputs[idx]) + horizon + len(predictions[idx]),
            ),
            predictions[idx],
            label="Predictions",
            color=custom_palette[1],
            marker="x",
            s=60,
        )
        axs[i].scatter(
            range(
                len(inputs[idx]) + horizon,
                len(inputs[idx]) + horizon + len(targets[idx]),
            ),
            targets[idx],
            label="Targets",
            color=custom_palette[0],
        )
        axs[i].set_ylim(min_y, max_y)
        if i == 0:
            axs[i].legend(loc="upper left")
            axs[i].set_xlabel("Step (15 minutes)")
            axs[i].set_ylabel("Scaled value")

        # Add grid to the plots
        axs[i].grid(True, which="both", linestyle="--", linewidth=0.5)

    fig.suptitle(title, fontsize=10, y=0.93)
