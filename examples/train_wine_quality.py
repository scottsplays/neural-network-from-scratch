"""
Train and evaluate on the UCI Wine Quality (red) dataset.

Pipeline:
  1. Load 11 physicochemical features + quality score
  2. Normalize features per column, transpose to samples-as-rows
  3. 70/30 train/test split (seeded for reproducibility)
  4. Train a 2-hidden-layer network and report MMSE on both splits

Usage:
    python examples/train_wine_quality.py
"""

import os
import random
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mathematics
from vector import Vector
from matrix import Matrix, MMSE
from NeuralNetwork import NeuralNetwork


def load_wine_data(csv_path):
    """Build input/output matrices from the CSV file."""
    df = pd.read_csv(csv_path)
    feature_columns = list(df.columns[:11])

    # Stack features as rows, normalize each feature, then transpose so
    # each row is one wine sample (required by the network).
    inputs = Matrix()
    for column_name in feature_columns:
        inputs.append(Vector(list(df[column_name])))
    inputs.normalize()
    inputs.transpose()

    outputs = Matrix([Vector(list(df["quality"]))])
    outputs.transpose()

    return inputs, outputs


def split_data(inputs, outputs, train_ratio=0.7, seed=42):
    """Random train/test split at the sample (row) level."""
    random.seed(seed)
    train_x, train_y = Matrix([]), Matrix([])
    test_x, test_y = Matrix([]), Matrix([])

    for i in range(inputs.get_n()):
        if random.random() < train_ratio:
            train_x.append(inputs.get_row(i))
            train_y.append(outputs.get_row(i))
        else:
            test_x.append(inputs.get_row(i))
            test_y.append(outputs.get_row(i))

    return train_x, train_y, test_x, test_y


def evaluate(model, inputs, outputs):
    """Run a forward pass without weight updates and return batch MMSE."""
    from layer import LayerTensor, matrix_to_layer_matrix

    input_batch = matrix_to_layer_matrix(inputs.get_copy().normalize())
    model.layers = LayerTensor([input_batch])
    model.pre_activations = LayerTensor()
    predictions = model.forward_pass(input_batch)
    return MMSE(predictions, outputs.get_copy())


def main():
    csv_path = os.path.join(ROOT, "data", "winequality-red.csv")
    inputs, outputs = load_wine_data(csv_path)
    train_x, train_y, test_x, test_y = split_data(inputs, outputs)

    print(f"Training samples: {train_x.get_n()}")
    print(f"Test samples:     {test_x.get_n()}")

    model = NeuralNetwork(
        train_x,
        train_y,
        function=mathematics.linear,
        learning_rate=0.00001,  # smaller rate for the full 1,500-sample batch
    )
    model.add_layer(16, mathematics.tanh)
    model.add_layer(16, mathematics.tanh)

    plot_path = os.path.join(ROOT, "assets", "wine_loss_curve.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    model.train(epochs=1000, log_every=100, plot_path=plot_path, show_plot=False)

    train_loss = evaluate(model, train_x, train_y)
    test_loss = evaluate(model, test_x, test_y)

    print(f"\nTrain MMSE: {train_loss:.6f}")
    print(f"Test MMSE:  {test_loss:.6f}")
    print(f"Loss curve saved to: {plot_path}")


if __name__ == "__main__":
    main()
