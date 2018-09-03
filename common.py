'''
Implementing all accuracy for the model
'''
import tensorflow as tf
import pandas as pd
from pandas import Series
import numpy as np


def find_trend(comparison_dataset, base_dataset):
    j = 0
    trend = list()
    while j < len(base_dataset):
        # Check whether the index is out of range or not
        # If yes then exit the loop
        base_dataset = np.reshape(base_dataset, len(base_dataset))
        base_dataset = pd.Series(base_dataset)
        if j + 1 == len(base_dataset.values):
            break

        if comparison_dataset[j] < comparison_dataset[j + 1]:
            trend.append("up")
        else:
            trend.append("down")
        j += 1

    return trend


def net_pnl(predictions, original, predicted_trends):
    i = 0
    PNL = list()
    while i < len(predictions):

        # Otherwise a key error would happen
        original = pd.Series(original)
        if i + 1 == len(original.values):
            break

        if predicted_trends[i] == "up":
            # For upward trend
            diff = original.values[i + 1] - original.values[i]
            PNL.append(round(diff, 2))
        else:
            # For downward trend
            diff = original.values[i] - original.values[i + 1]
            PNL.append(round(diff, 2))
        i += 1

    return PNL


def accuracy(predicted_trend, actual_trend):
    i = 0
    correct_pred = 0
    incorrect_pred = 0
    while i < len(predicted_trend):
        if predicted_trend[i] == actual_trend[i]:
            correct_pred += 1
        else:
            incorrect_pred += 1

        i += 1

    return correct_pred, incorrect_pred