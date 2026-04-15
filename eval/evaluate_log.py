"""Utilities for parsing Mukara training logs into tabular metrics."""

from pathlib import Path
import re
import numpy as np
import pandas as pd


EPOCH_STEP_PATTERN = re.compile(r"Epoch (\d+), Step (\d+)")
TRAIN_PATTERN = re.compile(r"Train Loss: GEH: ([\\d.]+|nan), MAE: ([\\d.]+|nan)")
VALID_PATTERN = re.compile(r"Valid Loss: GEH: ([\\d.]+|nan), MAE: ([\\d.]+|nan)")


def _to_float(value: str) -> float:
    """Convert metric strings to floats, preserving NaN values."""
    return float(value) if value != "nan" else np.nan


def extract_metrics(log_path: str | Path) -> pd.DataFrame:
    """Read a training log and return epoch/step-level train/validation metrics."""
    epochs = []
    steps = []
    train_geh = []
    train_mae = []
    valid_geh = []
    valid_mae = []

    with Path(log_path).open("r", encoding="utf-8") as file:
        for line in file:
            train_match = TRAIN_PATTERN.search(line)
            valid_match = VALID_PATTERN.search(line)

            if train_match:
                train_geh.append(_to_float(train_match.group(1)))
                train_mae.append(_to_float(train_match.group(2)))

                epoch_step_match = EPOCH_STEP_PATTERN.search(line)
                if epoch_step_match:
                    epochs.append(int(epoch_step_match.group(1)))
                    steps.append(int(epoch_step_match.group(2)))

            if valid_match:
                valid_geh.append(_to_float(valid_match.group(1)))
                valid_mae.append(_to_float(valid_match.group(2)))

    return pd.DataFrame(
        {
            "epoch": epochs,
            "step": steps,
            "train_geh": train_geh,
            "train_mae": train_mae,
            "valid_geh": valid_geh,
            "valid_mae": valid_mae,
        }
    )
