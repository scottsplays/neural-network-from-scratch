"""
Quick-start demo: train on randomly generated regression data.

This script is the fastest way to verify the framework works end-to-end.
Expected final MMSE after 1000 epochs: ~0.07

Usage:
    python examples/train_synthetic.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mathematics
from matrix import Matrix
from NeuralNetwork import NeuralNetwork


def main():
    # Matrix(n, m) builds n samples with m random features in [-1, 1].
    inputs = Matrix(15, 4)
    outputs = Matrix(15, 2)

    model = NeuralNetwork(
        inputs,
        outputs,
        function=mathematics.linear,
        learning_rate=0.0001,  # conservative rate for stable convergence
    )
    model.add_layer(32, mathematics.RELU)
    model.add_layer(32, mathematics.RELU)

    plot_path = os.path.join(ROOT, "assets", "synthetic_loss_curve.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    losses = model.train(
        epochs=1000,
        log_every=100,
        plot_path=plot_path,
        show_plot=False,
    )

    print(f"\nFinal loss: {losses[-1]:.6f}")
    print(f"Loss curve saved to: {plot_path}")


if __name__ == "__main__":
    main()
