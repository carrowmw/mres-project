import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from modelling_functions import (
    LSTMModel,
    LinearModel,
    prepare_dataloaders,
    load_and_preprocess_data,
)


from eda_helper import get_custom_palette, get_custom_heatmap, get_custom_colormap

custom_palette = get_custom_palette()
custom_heatmap = get_custom_heatmap()
custom_colormap = get_custom_colormap()

cwd = os.getcwd()
column_names = [
    "MAE",
    "MSE",
    "RMSE",
    "R^2",
    "Model",
    "Training Time (s)",
    "Test Number",
    "Completeness",
    "Sequence Length",
    "WindowSize",
    "Horizon",
]
performance = pd.read_csv(
    cwd + "\\performance_metrics_mm.csv", header=None, names=column_names
)
runtime = pd.read_csv(cwd + "\\runtime_metrics_mm.csv")
weights = pd.read_csv(cwd + "\\weights_mm.csv")

selected_columns = [
    "MAE",
    "MSE",
    "RMSE",
    "R^2",
    "Training Time (s)",
    "Test Number",
    "Completeness",
    "Sequence Length",
    "WindowSize",
    "Horizon",
]

performance_lstm = performance[performance["Model"] == "LSTM"][selected_columns]
performance_linear = performance[performance["Model"] == "Linear"][selected_columns]


def process_data(dataframe):
    """
    Process the given dataframe based on the provided steps.
    Args:
    - dataframe (pd.DataFrame): The dataframe to process.
    Returns:
    - pd.DataFrame: The processed dataframe.
    """
    dataframe = dataframe.sort_values(by="R^2").reset_index(drop=True)
    grouped_dataframe = dataframe.groupby(
        ["Completeness", "Horizon", "WindowSize"]
    ).mean()
    grouped_dataframe.reset_index(inplace=True)
    sorted_dataframe = grouped_dataframe.sort_values(
        by="Training Time (s)"
    ).reset_index(drop=True)

    return sorted_dataframe


# Process the dataframes
sorted_lstm_performance = process_data(performance_lstm)
sorted_linear_performance = process_data(performance_linear)

# Multiplying index value by length of window_sizes to get the correct index for viewing
sorted_linear_performance.index = sorted_lstm_performance.iloc[::6].index

window_values = np.sort(sorted_lstm_performance["WindowSize"].unique())
horizon_values = np.sort(sorted_lstm_performance["Horizon"].unique())
completeness_values = np.sort(sorted_lstm_performance["Completeness"].unique())

# Additional operations for linear_data
sorted_linear_performance = sorted_linear_performance.sort_values(
    ["Completeness", "Horizon"], ascending=[False, True]
).reset_index(drop=True)

def extract_values_from_filename(filename):
    """
    Extracts the Completeness, Sequence Length, Horizon, and Window Size values
    from the given filename using regular expressions.

    Args:
    - filename (str): The filename to extract the values from.

    Returns:
    - dict: A dictionary containing the extracted values. Returns None for values not found.
    """
    # Regular expressions for each value
    regex_patterns = {
        "Completeness": r"Completeness([\d.]+)",
        "SequenceLength": r"SequenceLength(\d+)",
        "Horizon": r"Horizon(\d+)",
        "WindowSize": r"WindowSize(\d+)",
    }

    extracted_values = {}
    for key, pattern in regex_patterns.items():
        match = re.search(pattern, filename)
        if match:
            # Convert to float if it has a decimal point, else convert to int
            value = (
                float(match.group(1)) if "." in match.group(1) else int(match.group(1))
            )
            extracted_values[key] = value
        else:
            extracted_values[key] = None

    return extracted_values


# Specify the folder path
folder_path = r"C:\#code\#python\#current\mres-project\analysis_files\anomaly_graphs"


def setup():
    custom_palette = get_custom_palette()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_folder_path = r"C:\#code\#python\#current\mres-project\analysis_files\saved_models"
    graph_save_path = r"C:\#code\#python\#current\mres-project\analysis_files\anomaly_graphs"
    os.makedirs(graph_save_path, exist_ok=True)
    return custom_palette, device, model_folder_path, graph_save_path

def process_file(model_file, model_folder_path):
    if not model_file.endswith(".pt"):
        return None
    values = extract_values_from_filename(model_file)
    state_path = os.path.join(model_folder_path, model_file)
    return values, state_path

def load_data(values, east_timeseries=None):
    if east_timeseries is None or completeness_value != values["Completeness"]:
        east_timeseries = load_and_preprocess_data(values["Completeness"])
    data = east_timeseries
    return data

def get_dataloaders(data, values):
    return prepare_dataloaders(
        data=data,
        window_size=values["WindowSize"],
        input_feature_indices=[0],
        target_feature_index=0,
        horizon=values["Horizon"],
        stride=1,
        batch_size=1,
        shuffle=False,
    )

def load_model(state_path, model_type, device):
    if model_type == "Linear":
        model = LinearModel(input_size=1)
    else:
        model = LSTMModel(feature_dim=1)
    model.load_state_dict(torch.load(state_path))
    model.to(device)
    return model

def predict(model, test_dataloader, device):
    all_predictions = []
    for inputs, _ in test_dataloader:
        with torch.inference_mode():
            batch_predictions = model(inputs.float().to(device))
            all_predictions.append(batch_predictions.detach().cpu().numpy())
    return np.concatenate(all_predictions).ravel()

def detect_anomalies(all_predictions, targets_numpy):
    errors = np.abs(all_predictions - targets_numpy)
    totals = []
    percentages = []
    standard_deviations = list(np.arange(start=2, stop=10.1, step=0.1))

        standard_deviations = list(np.arange(start=2, stop=11, step=1))
        threshold_i = errors.mean() + i * errors.std()
        anomalies_i = errors > threshold_i
        total_anomalies_i = np.sum(anomalies_i)
        anomaly_percentage_i = (total_anomalies_i / len(targets)) * 100
        totals.append(total_anomalies_i)
        percentages.append(anomaly_percentage_i)

    return errors, totals, percentages, threshold_i

def plot_graph1(horizon_value, window_size_value, completeness_value, standard_deviations, percentages, graph_save_path, model_file, custom_palette):
    # ... [plotting code for first graph]
    plt.close()

def plot_graph2(errors, threshold_i, targets_numpy, all_predictions, graph_save_path, model_file, custom_palette):
    # ... [plotting code for second graph]
    plt.close()

def main():
    custom_palette, device, model_folder_path, graph_save_path = setup()

    east_timeseries = None

    for model_file in os.listdir(model_folder_path):
        values, state_path = process_file(model_file, model_folder_path)
        if values is None:
            continue
        data = load_data(values, east_timeseries)
        train_dataLoader, test_dataloader, test_targets = get_dataloaders(data, values)

        if model_file.startswith("Linear"):
            model = load_model(state_path, "Linear", device)
        else:
            model = load_model(state_path, "LSTM", device)

        all_predictions = predict(model, test_dataloader, device)
        targets_numpy = test_targets.detach().cpu().numpy().flatten()
        errors, totals, percentages, threshold_i = detect_anomalies(all_predictions, targets_numpy)

        plot_graph1(values["Horizon"], values["WindowSize"], values["Completeness"], standard_deviations, percentages, graph_save_path, model_file, custom_palette)
        plot_graph2(errors, threshold_i, targets_numpy, all_predictions, graph_save_path, model_file, custom_palette)

if __name__ == "__main__":
    main()
