import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
INPUT_INDEX = 0
TARGET_INDEX = 0
BATCH_SIZE = 8
EPOCHS = 5
ERROR_STD = 4
DATA_PATH = "../data/saville_row_east_west/"
OUTPUT_TABLES_PATH = "../output/tables/4/"
OUTPUT_FIGURES_PATH = "../output/figures/4/"

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set fonts and style for matplotlib
FONT_NAME = "Derailed"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = FONT_NAME
plt.rcParams["axes.unicode_minus"] = False

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
            - RMSE (float): Root Mean Squared Error.
            - R^2 (float): R squared or Coefficient of Determination.

    Note:
        The function assumes the predictions and targets are torch tensors.
        They are then flattened and detached before computation.
    """
    predictions_flat_np = predictions.flatten().detach().numpy()
    targets_flat_np = targets.flatten().detach().numpy()

    return {
        "MAE": mean_absolute_error(targets_flat_np, predictions_flat_np),
        "MSE": mean_squared_error(targets_flat_np, predictions_flat_np),
        "RMSE": np.sqrt(mean_absolute_error(targets_flat_np, predictions_flat_np)),
        "R^2": r2_score(targets_flat_np, predictions_flat_np),
    }


class BaselineModel(nn.Module):
    """
    A simple baseline model that returns the last element from each input sequence.
    This model acts as a naive predictor, taking the most recent data point as its prediction.
    """

    def forward(self, x):
        """
        Forward pass of the BaselineModel.

        Args:
            x (torch.Tensor): Input tensor with sequences.
                              Expected shape: [batch_size, sequence_length, feature_dim].

        Returns:
            torch.Tensor: Predictions tensor containing the last element from each sequence.
                          Shape: [batch_size, feature_dim].
        """
        return x[:, -1]  # Get the last element from each input sequence


class LSTMModel(nn.Module):
    """
    LSTM-based model for time series prediction.

    Args:
        input_size (int, optional): The number of expected features in the input `x`. Default: 1.
        hidden_size (int, optional): The number of features in the hidden state. Default: 50.
        output_size (int, optional): Number of features in the output. Default: 1.
        num_layers (int, optional): Number of recurrent layers. Default: 1.
    """

    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the LSTMModel.

        Args:
            x (torch.Tensor): Input tensor with sequences.
                              Expected shape: [batch_size, sequence_length, feature_dim].

        Returns:
            torch.Tensor: Predictions tensor, taking the last output from the LSTM sequence.
                          Shape: [batch_size, feature_dim].
        """
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x[:, -1, :]  # Selecting the last output of the sequence


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


def train_lstm_model(model, train_dataloader, test_dataloader):
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    train_metrics = []
    test_metrics = []

    # Training loop
    for epoch in range(EPOCHS):
        train_MSE, train_RMSE = 0, 0
        model.train()

        # Training phase
        for X, y in train_dataloader:
            # Prepare data and move to device
            X = X.unsqueeze(-1).float().to(device)
            y = y.unsqueeze(-1).unsqueeze(-1).float().to(device)

            # Model prediction and loss calculation
            train_preds = model(X)
            loss_MSE = criterion(train_preds.unsqueeze(dim=-1), y)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss_MSE.backward()
            optimizer.step()

            # Accumulate loss for metrics
            train_MSE += loss_MSE.item()
            train_RMSE += sqrt(loss_MSE.item())

        # Calculate average loss
        train_MSE /= len(train_dataloader)
        train_RMSE /= len(train_dataloader)
        train_metrics.append([epoch, train_MSE, train_RMSE])

        test_MSE, test_RMSE = 0, 0
        model.eval()

        # Evaluation phase
        with torch.no_grad():
            for X, y in test_dataloader:
                # Prepare data and move to device
                X = X.unsqueeze(-1).float().to(device)
                y = y.unsqueeze(-1).unsqueeze(-1).float().to(device)

                # Model prediction and loss calculation
                test_preds = model(X)
                loss_MSE = criterion(test_preds.unsqueeze(dim=-1), y)

                # Accumulate loss for metrics
                test_MSE += loss_MSE.item()
                test_RMSE += sqrt(loss_MSE.item())

        # Calculate average loss
        test_MSE /= len(test_dataloader)
        test_RMSE /= len(test_dataloader)
        test_metrics.append([epoch, test_MSE, test_RMSE])

        # Print progress every 5 epochs
        if epoch % 5 == 4:
            print(
                f"Epoch: {epoch+1} | Train MSE: {train_MSE:.5f} | Train RMSE: {train_RMSE:.5f} | Test MSE: {test_MSE:.5f} | Test RMSE: {test_RMSE:.5f}"
            )

    return train_metrics, test_metrics


def main():
    # Load and preprocess data
    east_df, _ = load_data()
    east_dict = preprocess_data(east_df, COMPLETENESS_THRESHOLD)
    east_timeseries = east_dict["data"]

    # Extract input and target sequences
    inputs, targets = sliding_windows(
        east_timeseries, WINDOW_SIZE, INPUT_INDEX, TARGET_INDEX, HORIZON, STRIDE
    )

    # Baseline Model Evaluation
    baseline_model = BaselineModel()
    baseline_predictions = baseline_model(inputs.squeeze())
    metrics = compute_performance_metrics(baseline_predictions, targets)
    metrics["Model"] = "Baseline"
    performance_df = pd.DataFrame([metrics])

    # Setup LSTM Model
    model = LSTMModel().to(device)
    train_size = int(0.8 * len(inputs))
    train_inputs, test_inputs = inputs[:train_size], inputs[train_size:]
    train_targets, test_targets = targets[:train_size], targets[train_size:]
    train_dataset = TimeSeriesDataset(train_inputs, train_targets)
    test_dataset = TimeSeriesDataset(test_inputs, test_targets)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train LSTM model and get metrics
    train_metrics, test_metrics = train_lstm_model(
        model, train_dataloader, test_dataloader
    )
    train_metrics_df = pd.DataFrame(train_metrics, columns=["Epoch", "MSE", "RMSE"])
    test_metrics_df = pd.DataFrame(test_metrics, columns=["Epoch", "MSE", "RMSE"])
    print("LSTM Train Metrics")
    print(train_metrics_df.iloc[::5].round(3))
    print()
    print("LSTM Test Metrics")
    print(test_metrics_df.round(3))

    # Prediction and Error Calculation
    model.eval()
    with torch.no_grad():
        predictions = model(inputs.to(device).unsqueeze(dim=-1).float())
    errors = (predictions.squeeze(dim=-1) - targets.squeeze(dim=-1)).abs().numpy()

    LSTM_predictions = model(inputs.to(device).unsqueeze(dim=-1).float())
    metrics = compute_performance_metrics(LSTM_predictions, targets)
    metrics["Model"] = "LSTM"
    metrics_df = pd.DataFrame([metrics])  # Convert metrics to a DataFrame
    performance_df = pd.concat([performance_df, metrics_df], ignore_index=True)
    print(performance_df.round(3))

    # Calculate anomaly threshold based on error statistics
    threshold = errors.mean() + ERROR_STD * errors.std()

    # Detect anomalies based on error threshold
    anomalies = errors > threshold
    total_anomalies = np.sum(anomalies)
    time = np.arange(len(inputs))
    anomaly_times = time[anomalies]
    anomaly_values = targets[anomalies]

    # Focus on a specific time range for visualization
    anomalies_in_range = (anomaly_times > 300) & (anomaly_times < 500)
    anomaly_times_in_range = anomaly_times[anomalies_in_range]
    anomalies_values_in_range = anomaly_values[anomalies_in_range]

    # Visualize predictions, targets, and anomalies
    plt.figure(figsize=(10, 5))
    plt.plot(
        time[300:500],
        targets[300:500],
        label="Targets",
        linewidth=1,
        color=custom_palette[0],
    )
    plt.plot(
        time[300:500],
        predictions.numpy()[300:500],
        linewidth=1,
        label="Predicted data",
        color=custom_palette[1],
    )
    plt.scatter(
        anomaly_times_in_range,
        anomalies_values_in_range,
        label="Anomalies",
        color=custom_palette[7],
        marker=7,
        s=30,
        zorder=3,
    )
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.ylabel("Scaled value")
    plt.xlabel("Timestep")
    plt.title(
        f"Total anomalies: {total_anomalies} | Anomaly percentage: {(total_anomalies / len(targets))*100:.3f}% | Anomaly threshold: {ERROR_STD}$\sigma$"
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
