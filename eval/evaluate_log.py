import os
import re
import pandas as pd
import numpy as np

def extract_metrics(log_path):
    """
    A function to read the log file and extract metrics
    """
    # Initialize empty lists to store data
    epochs = []
    steps = []
    train_geh = []
    train_mae = []
    valid_geh = []
    valid_mae = []

    # Regular expressions to capture the data from the log
    epoch_step_pattern = re.compile(r'Epoch (\d+), Step (\d+)')
    train_pattern = re.compile(r'Train Loss: GEH: ([\d.]+|nan), MAE: ([\d.]+|nan)')
    valid_pattern = re.compile(r'Valid Loss: GEH: ([\d.]+|nan), MAE: ([\d.]+|nan)')

    # Reading the log file
    with open(log_path, 'r') as f:
        for line in f:
            train_match = train_pattern.search(line)
            valid_match = valid_pattern.search(line)
            
            if train_match:
                train_geh.append(float(train_match.group(1)) if train_match.group(1) != 'nan' else np.nan)
                train_mae.append(float(train_match.group(2)) if train_match.group(2) != 'nan' else np.nan)

                epoch_step_match = epoch_step_pattern.search(line)
                epoch = int(epoch_step_match.group(1))
                step = int(epoch_step_match.group(2))
                epochs.append(epoch)
                steps.append(step)
            
            if valid_match:
                valid_geh.append(float(valid_match.group(1)) if valid_match.group(1) != 'nan' else np.nan)
                valid_mae.append(float(valid_match.group(2)) if valid_match.group(2) != 'nan' else np.nan)

    # Create a DataFrame
    df = pd.DataFrame({
        'epoch': epochs,
        'step': steps,
        'train_geh': train_geh,
        'train_mae': train_mae,
        'valid_geh': valid_geh,
        'valid_mae': valid_mae
    })

    return df