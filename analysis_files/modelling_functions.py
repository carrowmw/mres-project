"""
All the functions to run the analysis loops for both univariate and multivariate models.
"""

# Load the required packages
import os
import typing
import timeit
import pandas as pd
import numpy as np
from math import sqrt

# Import PyTorch packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import scikit-learn functions
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import custom helper functions
from eda_helper import (
    sliding_windows,
    preprocess_data,
    get_custom_palette,
    get_custom_colormap,
)

# Constants
COMPLETENESS_THRESHOLD = 1
WINDOW_SIZE = 12
HORIZON = 3
STRIDE = 1
INPUT_INDICES = 0
TARGET_INDEX = 0
BATCH_SIZE = 8
EPOCHS = 5
ERROR_STD = 4
DATA_PATH = r"C:\#code\#python\#current\mres-project\data\saville_row_east_west"
OUTPUT_TABLES_PATH = r"../output/tables/4/"
OUTPUT_FIGURES_PATH = r"../output/figures/4/"

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get font settings
os.system(r"C:\#code\#python\#current\mres-project\analysis_files\mpl_config.py")

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(OUTPUT_TABLES_PATH, exist_ok=True)
os.makedirs(OUTPUT_FIGURES_PATH, exist_ok=True)

custom_palette = get_custom_palette()
custom_colormap = get_custom_colormap()


def load_data():
    """
    Loads the east and west dataframes from pickled files.

    Returns:
        tuple: A tuple containing two dataframes:
            - east_df (pd.DataFrame): Dataframe for the east data.
            - west_df (pd.DataFrame): Dataframe for the west data.

    Note:
        The paths to the pickled files are assumed to be constructed using
        a global DATA_PATH variable and the respective filenames.
    """
    east_df = pd.read_pickle(os.path.join(DATA_PATH, "east_df.pkl"))
    west_df = pd.read_pickle(os.path.join(DATA_PATH, "west_df.pkl"))
    return east_df, west_df


def load_and_preprocess_data(completeness):
    """
    Load and preprocess the data.
    :param completeness: completeness threshold
    :return: preprocessed timeseries data
    """
    east_df, _ = load_data()
    east_dict = preprocess_data(east_df, completeness)
    return east_dict["data"]


def prepare_dataloaders(
    data: np.ndarray,
    window_size: int,
    input_feature_indices: list,
    target_feature_index: int,
    horizon: int,
    stride: int,
    batch_size: int,
    shuffle=False,
    num_workers=0,
) -> typing.Tuple[Dataset, Dataset, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares training and test dataloaders using sliding windows on the given time-series data.

    Parameters:
    - data (np.ndarray): Time-series data.
    - window_size (int): Size of each sliding window.
    - input_feature_indices (list of ints): Indices of features to be considered as input.
    - target_feature_index (int): The index of the feature that needs to be predicted.
    - horizon (int): Steps ahead for the prediction.
    - stride (int): Steps between the start of each window.
    - batch_size (int): Number of samples per batch to load.
    - shuffle (bool, optional): Whether to shuffle the data samples. Defaults to False.
    - num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 0.

    Returns:
    - tuple: Contains train DataLoader, test DataLoader, test inputs, test targets, train inputs, and train targets.
    """
    inputs, targets = sliding_windows(
        data=data,
        window_size=window_size,
        input_feature_indices=input_feature_indices,
        target_feature_index=target_feature_index,
        horizon=horizon,
        stride=stride,
    )

    # Split data into train and test sets
    train_size = int(0.8 * len(inputs))
    train_inputs, test_inputs = inputs[:train_size], inputs[train_size:]
    train_targets, test_targets = targets[:train_size], targets[train_size:]

    # Create custom PyTorch Dataset and DataLoader objects
    train_dataset = TimeSeriesDataset(train_inputs, train_targets)
    test_dataset = TimeSeriesDataset(test_inputs, test_targets)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return (
        train_dataloader,
        test_dataloader,
        test_inputs,
        test_targets,
        train_inputs,
        train_targets,
    )


def compute_performance_metrics(predictions, targets):
    """
    Compute common performance metrics for regression.

    Parameters:
        - predictions (torch.Tensor): The predicted values.
        - targets (torch.Tensor): The true target values.

    Returns:
        dict: A dictionary containing the following metrics:
            - MAE (float): Mean Absolute Error.
            - MSE (float): Mean Squared Error.
            - rmse (float): Root Mean Squared Error.
            - R^2 (float): R squared or Coefficient of Determination.

    Note:
        The function assumes the predictions and targets are torch tensors.
        They are then flattened and detached before computation.
    """
    predictions_flat_np = predictions.flatten()
    targets_flat_np = targets.flatten()

    return {
        "MAE": mean_absolute_error(targets_flat_np, predictions_flat_np),
        "MSE": mean_squared_error(targets_flat_np, predictions_flat_np),
        "rmse": np.sqrt(mean_absolute_error(targets_flat_np, predictions_flat_np)),
        "R^2": r2_score(targets_flat_np, predictions_flat_np),
    }


def save_model_state(
    model,
    model_name,
    folder_name,
):
    """
    Save the state of a model to disk.

    Args:
        model (torch.nn.Module): The model to save.
        model_name (str): Name for the saved model file (without extension).
        path (str): Directory where the model should be saved.
    """
    path = "C:\\#code\\#python\\#current\\mres-project\\analysis_files\\"
    path = os.path.join(path, folder_name)
    os.makedirs(path, exist_ok=True)  # Ensure the directory exists
    model_path = os.path.join(path, f"{model_name}.pt")
    torch.save(model.state_dict(), model_path)


def create_anomaly_column(timeseries_array: np.ndarray) -> np.ndarray:
    """
    Add a new boolean column to a given time series array indicating anomalies.

    The new column will have a True value if the corresponding value in the first
    column of the original array is greater than 0.9, otherwise, it will have a False value.

    Parameters:
    - timeseries_array (np.ndarray): A 2D numpy array where the first column represents
      the time series values.

    Returns:
    - np.ndarray: The original numpy array concatenated with the new boolean column indicating
    anomalies.

    Example:
    >>> arr = np.array([[0.8], [0.92], [0.85], [0.95]])
    >>> create_anomaly_column(arr)
    array([[0.8 , 0.  ],
           [0.92, 1.  ],
           [0.85, 0.  ],
           [0.95, 1.  ]])
    """
    # Create a new boolean column where True indicates that the value in the first column is greater than 0.9
    anomaly = timeseries_array[:, 0] > 0.9
    # Reshape the new column to be two-dimensional so it can be concatenated with the original array
    anomaly = anomaly.reshape(-1, 1)
    # Concatenate the new column to your array along axis 1
    timeseries_array = np.concatenate((timeseries_array, anomaly), axis=1)
    return timeseries_array


def save_outputs_to_csv(performance_df, weights_df, runtime_dict):
    """
    Save the metrics and weights dataframes to CSV.
    :param performance_df: dataframe containing performance metrics
    :param weights_df: dataframe containing model weights
    :param runtime_dict: dictionary containing runtimes
    """
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


def compute_metrics(model, dataloader, criterion):
    """
    Compute MSE and rmse metrics for a given model and dataloader.

    Args:
    - model (torch.nn.Module): PyTorch model.
    - dataloader (torch.utils.data.DataLoader): DataLoader object.
    - criterion (torch.nn.Module): Loss function.

    Returns:
    - tuple: (Mean Squared Error, Root Mean Squared Error).
    """
    model.eval()
    test_mse, test_rmse = 0, 0

    for X, y in dataloader:
        with torch.inference_mode():
            X = X.float().to(device)
            y = y.unsqueeze(-1).unsqueeze(-1).float().to(device)
            predictions = model(X)
            loss_mse = criterion(predictions.unsqueeze(dim=1), y)

            test_mse += loss_mse.item()
            test_rmse += np.sqrt(loss_mse.item())

    test_mse /= len(dataloader)
    test_rmse /= len(dataloader)

    return test_mse, test_rmse


def evaluate_anomalies(predictions, targets):
    errors = np.abs(predictions - targets)

    # Set anomaly threshold
    error_deviations = 6
    anomaly_threshold = np.round(errors.mean() + error_deviations * errors.std(), 1)
    anomalies = errors > anomaly_threshold

    # Count total anomalies
    total_anomalies = np.sum(anomalies)

    anomaly_percentage = (total_anomalies / len(targets)) * 100
    mean_error = errors.mean()

    return anomaly_percentage, anomaly_threshold, mean_error


class TimeSeriesDataset(Dataset):
    """
    A custom Dataset for time series data.

    Args:
        sequences (torch.Tensor): Input sequences for the dataset.
                                  Shape: [num_samples, sequence_length, feature_dim].
        targets (torch.Tensor): Corresponding targets for the input sequences.
                                Shape: [num_samples, feature_dim].
    """

    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, index):
        """
        Fetches the sequence and target at a particular index.

        Args:
            index (int): The index to retrieve the data from.

        Returns:
            tuple: A tuple containing:
                - sequence (torch.Tensor): Input sequence of shape [sequence_length, feature_dim].
                - target (torch.Tensor): Corresponding target of shape [feature_dim].
        """
        return self.sequences[index], self.targets[index]


class LinearModel(nn.Module):
    """
    A simple linear regression model suitable for time series forecasting.

    Parameters:
    - input_size (int): Number of input features.

    Attributes:
    - linear (nn.Linear): A linear layer that transforms input features into a single output.

    Methods:
    - forward(x: torch.Tensor) -> torch.Tensor: Implements the forward propagation of the model.

    Example:
    --------
    >>> model = LinearModel(input_size=10)
    >>> input_data = torch.randn(32, 10)  # Batch of 32, each with 10 features
    >>> output = model(input_data)

    Notes:
    ------
    - The forward method can process both 2D (batch_size, num_features) and
      3D (batch_size, sequence_len, num_features) input tensors. If the input is 3D,
      it gets reshaped to 2D.
    """

    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        # If x is 3D (batch_size, sequence_len, num_features), we might need to reshape it
        x = x.reshape(x.size(0), -1)
        return self.linear(x)


class LSTMModel(nn.Module):
    """
    LSTM-based model designed for time series forecasting. Suitable for both univariate and multivariate time series.

    Parameters:
    - feature_dim (int): Number of expected features in the input `x`.
    - hidden_size (int, optional): Number of features in the hidden state. Default: 50.
    - output_dim (int, optional): Number of features in the output. Default: 1.
    - num_layers (int, optional): Number of recurrent layers. Default: 1.

    Attributes:
    - lstm (nn.LSTM): LSTM layer.
    - linear (nn.Linear): Linear layer to produce the final output.

    Methods:
    - forward(x: torch.Tensor) -> torch.Tensor: Implements the forward propagation of the model.

    Example:
    --------
    >>> model = LSTMModel(feature_dim=10)
    >>> input_data = torch.randn(32, 7, 10)  # Batch of 32, sequence length of 7, each with 10 features
    >>> output = model(input_data)
    """

    def __init__(self, feature_dim, hidden_size=50, output_dim=1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        """
        Forward propagation method for the LSTM model.

        Args:
        - x (torch.Tensor): Input tensor with sequences. Expected shape: [batch_size, sequence_length, feature_dim].

        Returns:
        - torch.Tensor: Output tensor with predictions. Shape: [batch_size, output_dim].
        """
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x[:, -1, :]  # Selecting the last output of the sequence


def train_and_evaluate_lstm_model(
    lstm_model, train_dataloader, test_dataloader, epochs, lr=0.1
):
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    train_metrics = []
    test_metrics = []

    # Training loop
    for epoch in range(epochs):
        train_mse, train_rmse = 0, 0
        lstm_model.train()

        # Training phase
        for X, y in train_dataloader:
            # Prepare data and move to device
            X = X.float().to(device)
            y = y.unsqueeze(-1).unsqueeze(-1).float().to(device)

            # Model prediction and loss calculation
            train_preds = lstm_model(X)
            loss_MSE = criterion(train_preds.unsqueeze(dim=-1), y)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss_MSE.backward()
            optimizer.step()

            # Accumulate loss for metrics
            train_mse += loss_MSE.item()
            train_rmse += sqrt(loss_MSE.item())

        # Calculate average loss
        train_mse /= len(train_dataloader)
        train_rmse /= len(train_dataloader)
        train_metrics.append([epoch, train_mse, train_rmse])

        # Evaluation Phase
        test_MSE, test_rmse = 0, 0
        lstm_model.eval()

        with torch.inference_mode():
            for X, y in test_dataloader:
                # Prepare data and move to device
                X = X.float().to(device)
                y = y.unsqueeze(-1).unsqueeze(-1).float().to(device)

                # Model prediction and loss calculation
                test_preds = lstm_model(X)
                loss_MSE = criterion(test_preds.unsqueeze(dim=-1), y)

                # Accumulate loss for metrics
                test_MSE += loss_MSE.item()
                test_rmse += sqrt(loss_MSE.item())

        # Calculate average loss
        test_MSE /= len(test_dataloader)
        test_rmse /= len(test_dataloader)
        test_metrics.append([epoch, test_MSE, test_rmse])

        # Print progress every epochs/5 epochs
        if epoch % (epochs / 5) == ((epochs / 5) - 1):
            print(
                f"Epoch: {epoch+1} | Train MSE: {train_mse:.5f} | Train rmse: {train_rmse:.5f} | Test MSE: {test_MSE:.5f} | Test rmse: {test_rmse:.5f}"
            )

        # Adjust learning rate
        scheduler.step()

    return train_metrics, test_metrics


def train_and_evaluate_linear_model(
    linear_model, train_dataloader, test_dataloader, epochs, lr=0.1
):
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(linear_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    linear_weights = []
    train_metrics = []
    test_metrics = []

    # Training loop
    for epoch in range(epochs):
        train_mse, train_rmse = 0, 0
        linear_model.train()

        # Training phase
        for X, y in train_dataloader:
            # Prepare data and move to device
            X = X.float().to(device)
            y = y.unsqueeze(-1).unsqueeze(-1).float().to(device)

            # Model prediction and loss calculation
            train_preds = linear_model(X)
            loss_MSE = criterion(train_preds.unsqueeze(dim=1), y)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss_MSE.backward()
            optimizer.step()

            # Accumulate loss for metrics
            train_mse += loss_MSE.item()
            train_rmse += sqrt(loss_MSE.item())

        # Calculate average loss
        train_mse /= len(train_dataloader)
        train_rmse /= len(train_dataloader)
        train_metrics.append([epoch, train_mse, train_rmse])

        test_MSE, test_rmse = 0, 0
        linear_model.eval()

        # Evaluation Phase
        with torch.inference_mode():
            for X, y in test_dataloader:
                # Prepare data and move to device
                X = X.float().to(device)
                y = y.unsqueeze(-1).unsqueeze(-1).float().to(device)

                # Model prediction and loss calculation
                test_preds = linear_model(X)
                loss_MSE = criterion(test_preds.unsqueeze(dim=1), y)

                # Accumulate loss for metrics
                test_MSE += loss_MSE.item()
                test_rmse += sqrt(loss_MSE.item())

        test_MSE /= len(test_dataloader)
        test_rmse /= len(test_dataloader)
        test_metrics.append([epoch, test_MSE, test_rmse])

        # Saving the weights of the linear lstm_model
        linear_weights.append(
            pd.DataFrame(linear_model.linear.weight.clone().detach().cpu().numpy())
        )

        if epoch % (epochs / 5) == ((epochs / 5) - 1):
            print(
                f"Epoch: {epoch+1} | Train MSE: {train_mse:.5f} | Train rmse: {train_rmse:.5f} | Test MSE: {test_MSE:.5f} | Test rmse: {test_rmse:.5f}"
            )

        # Adjust learning rate
        scheduler.step()

    return linear_weights, train_metrics, test_metrics


def train_and_evaluate_model(
    model, train_dataloader, test_dataloader, model_type="Linear", epochs=1
):
    """
    Train and evaluate a given model.
    :param model: PyTorch model (Linear or LSTM)
    :param train_dataloader: training dataloader
    :param test_dataloader: test dataloader
    :param model_type: type of model ("Linear" or "LSTM")
    :return: predictions, metrics
    """
    if model_type == "Linear":
        return train_and_evaluate_linear_model(
            model, train_dataloader, test_dataloader, epochs=epochs
        )
    elif model_type == "LSTM":
        return train_and_evaluate_lstm_model(
            model, train_dataloader, test_dataloader, epochs=epochs
        )


def evaluate_linear_model(
    data,
    completeness,
    input_feature_indices,
    target_feature_index,
    sequence_length,
    horizon,
    epochs,
    stride,
    total_iterations,
):
    """
    Evaluate the performance of the Linear Model.

    Args:
    - east_timeseries (pandas.DataFrame): Processed timeseries data.
    - completeness (float): The completeness threshold.
    - horizon (int): The forecasting horizon.

    Returns:
    - tuple: (Linear Model Weights, Performance Metrics).
    """
    test_metrics_df = pd.DataFrame()
    train_metrics_df = pd.DataFrame()

    print(
        f"Test Number: {total_iterations} |  Completeness: {completeness} | Sequence Length {sequence_length} | Horizon: {horizon} | Window Size: {1}"
    )

    (
        train_dataloader,
        test_dataloader,
        test_inputs,
        test_targets,
        train_inputs,
        train_targets,
    ) = prepare_dataloaders(
        data=data[:sequence_length],
        window_size=1,
        input_feature_indices=input_feature_indices,
        target_feature_index=target_feature_index,
        horizon=horizon,
        stride=stride,
        batch_size=1,
        shuffle=False,
    )

    # Train and evaluate linear model
    linear_model = LinearModel(input_size=len(input_feature_indices)).to(device)
    print("\nTraining Linear Model...")
    start_time = timeit.default_timer()
    linear_weights, train_metrics, test_metrics = train_and_evaluate_linear_model(
        linear_model, train_dataloader, test_dataloader, epochs
    )

    end_time = timeit.default_timer()

    train_metrics_entry = {
        "Completeness": completeness,
        "Sequence Length": sequence_length,
        "Horizon": horizon,
        "Window Size": 1,
        "Train Metrics": train_metrics,
        "Test Metrics": test_metrics,
        "Linear Weights": linear_weights,
        "Test Number": total_iterations,
    }

    train_metrics_df = train_metrics_df._append(train_metrics_entry, ignore_index=True)

    # Compute performance metrics
    test_mse, test_rmse = compute_metrics(linear_model, test_dataloader, nn.MSELoss())
    print()
    print("Evaluating Linear Model...")
    print(f"Test MSE: {test_mse:.4f} | Test rmse: {test_rmse:.4f}")
    print()

    test_metrics_entry = {
        "Completeness": completeness,
        "Sequence Length": sequence_length,
        "Horizon": horizon,
        "Window Size": 1,
        "Test MSE": test_mse,
        "Test rmse": test_rmse,
        "Test Number": total_iterations,
    }

    test_metrics_df = test_metrics_df._append(test_metrics_entry, ignore_index=True)

    all_predictions = []
    for X, _ in test_dataloader:
        batch_predictions = linear_model(X.squeeze(0).float().to(device))
        all_predictions.append(batch_predictions.detach().cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0).flatten()
    test_targets_numpy = test_targets.detach().cpu().numpy().flatten()

    # Construct metrics dictionary
    metrics = compute_performance_metrics(all_predictions, test_targets_numpy)
    metrics.update(
        {
            "Model": "Linear",
            "Training Time (s)": end_time - start_time,
            "Test Number": total_iterations,
            "Completeness": completeness,
            "Sequence Length": sequence_length,
            "Horizon": horizon,
            "WindowSize": 1,
        }
    )

    folder_name = (
        "multivariate_model_states"
        if len(input_feature_indices) > 1
        else "univariate_model_states"
    )
    # Save model state
    save_model_state(
        linear_model,
        f"Linear_TestNumber{total_iterations}_Completeness{completeness}_SequenceLength{sequence_length}_Horizon{horizon}_WindowSize1",
        folder_name=folder_name,
    )

    test_metrics_df.to_csv(
        "linear_test_metrics.csv",
        index=False,
        mode="a",
        header=not os.path.exists("linear_test_metrics.csv"),
    )
    train_metrics_df.to_csv(
        "linear_train_metrics.csv",
        index=False,
        mode="a",
        header=not os.path.exists("linear_train_metrics.csv"),
    )

    return linear_model.linear.weight.detach().numpy(), metrics


def evaluate_lstm_model(
    data,
    completeness,
    input_feature_indices,
    target_feature_index,
    sequence_length,
    horizon,
    window_size,
    epochs,
    total_iterations,
):
    """
    Evaluate the performance of the LSTM Model.

    Args:
    - east_timeseries (pandas.DataFrame): Processed timeseries data.
    - completeness (float): The completeness threshold.
    - horizon (int): The forecasting horizon.
    - window_size (int): The window size for LSTM.

    Returns:
    - dict: Performance Metrics.
    """

    test_metrics_df = pd.DataFrame()
    train_metrics_df = pd.DataFrame()

    print(
        f"Test Number: {total_iterations} | Completeness: {completeness} | Sequence Length {sequence_length} | Horizon: {horizon} | Window Size: {window_size} "
    )

    (
        train_dataloader,
        test_dataloader,
        test_inputs,
        test_targets,
        train_inputs,
        train_targets,
    ) = prepare_dataloaders(
        data[:sequence_length],
        window_size,
        input_feature_indices=input_feature_indices,
        target_feature_index=target_feature_index,
        horizon=horizon,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Train and evaluate LSTM model
    lstm_model = LSTMModel(feature_dim=len(input_feature_indices)).to(device)
    print("\nTraining LSTM Model...")
    start_time = timeit.default_timer()
    train_metrics, test_metrics = train_and_evaluate_lstm_model(
        lstm_model, train_dataloader, test_dataloader, epochs=epochs
    )
    end_time = timeit.default_timer()

    train_metrics_entry = {
        "Completeness": completeness,
        "Sequence Length": sequence_length,
        "Horizon": horizon,
        "Window Size": 1,
        "Train Metrics": train_metrics,
        "Test Metrics": test_metrics,
        "Test Number": total_iterations,
    }

    train_metrics_df = train_metrics_df._append(train_metrics_entry, ignore_index=True)

    # Compute performance metrics
    test_mse, test_rmse = compute_metrics(lstm_model, test_dataloader, nn.MSELoss())
    print()
    print("Evaluating LSTM Model...")
    print(f"Test MSE: {test_mse:.4f} | Test rmse: {test_rmse:.4f}")
    print()

    test_metrics_entry = {
        "Completeness": completeness,
        "Sequence Length": sequence_length,
        "Horizon": horizon,
        "Window Size": 1,
        "Test MSE": test_mse,
        "Test rmse": test_rmse,
        "Test Number": total_iterations,
    }

    test_metrics_df = test_metrics_df._append(test_metrics_entry, ignore_index=True)

    all_lstm_predictions = []

    for X, _ in test_dataloader:
        batch_predictions = lstm_model(X.float().to(device))
        all_lstm_predictions.append(batch_predictions.detach().cpu().numpy())
    all_lstm_predictions_flat = np.concatenate(all_lstm_predictions).ravel()
    test_targets_numpy = test_targets.detach().cpu().numpy().flatten()

    # Construct metrics dictionary
    metrics = compute_performance_metrics(all_lstm_predictions_flat, test_targets_numpy)
    metrics.update(
        {
            "Model": "LSTM",
            "Training Time (s)": end_time - start_time,
            "Test Number": total_iterations,
            "Completeness": completeness,
            "Sequence Length": sequence_length,
            "Horizon": horizon,
            "WindowSize": window_size,
        }
    )
    folder_name = (
        "multivariate_model_states"
        if len(input_feature_indices) > 1
        else "univariate_model_states"
    )
    # Save model state
    save_model_state(
        lstm_model,
        f"LSTM_TestNumber{total_iterations}_Completeness{completeness}_SequenceLength{sequence_length}_Horizon{horizon}_WindowSize{window_size}",
        folder_name=folder_name,
    )

    test_metrics_df.to_csv(
        "lstm_test_metrics.csv",
        index=False,
        mode="a",
        header=not os.path.exists("lstm_test_metrics.csv"),
    )
    train_metrics_df.to_csv(
        "lstm_train_metrics.csv",
        index=False,
        mode="a",
        header=not os.path.exists("lstm_train_metrics.csv"),
    )

    return metrics
