# Neural Network Framework from Scratch

A feedforward neural network built from first principles in Python using custom `Matrix`, `Vector`, `Layer`, and `LayerConnection` classes. No PyTorch, TensorFlow, or scikit-learn is used for the core model — only plain Python, math, and linear algebra.

**Author:** [Scott Kennedy](https://github.com/scottsplays)  
**Resume project:** Object-oriented neural network framework for ML/AI engineering portfolio work

---

## Why this project exists

This repository demonstrates that I can implement the full ML training loop myself:

- Forward propagation through configurable hidden layers
- Activation functions (ReLU, sigmoid, tanh, linear) and their derivatives
- Mean squared error loss
- Backpropagation and gradient-based weight updates
- Feature normalization and training loss visualization

It is a learning and portfolio project, not a production framework.

---

## Architecture

```
Input Matrix (samples × features)
        ↓
  [Hidden Layer + activation]  × N
        ↓
  [Output Layer + linear activation]
        ↓
  Predictions → MMSE loss → backprop → weight update
```

### Core components

| Module | Responsibility |
|--------|----------------|
| `vector.py` | Vector math, normalization, element-wise ops |
| `matrix.py` | Matrix math, multiplication, MMSE loss |
| `neuron.py` | Single neuron value + activation |
| `layer.py` | Layers of neurons, batch forward pass |
| `LayerConnection.py` | Weights, activations, gradient update |
| `NeuralNetwork.py` | Network construction, training loop |
| `mathematics.py` | Activations and derivatives |

---

## Quick start

### 1. Clone or copy this folder

This directory (`github_repo/`) is a presentation-ready copy meant to be published as its own GitHub repository. Your local development files can stay separate.

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 3. Run the examples

**Synthetic regression (fast demo):**

```bash
python examples/train_synthetic.py
```

**UCI Wine Quality dataset:**

```bash
python examples/train_wine_quality.py
```

Loss curves are saved to `assets/`.

---

## Example results

### Synthetic data (`15 samples`, `4 features`, `2 outputs`)

- Architecture: `4 → 32 (ReLU) → 32 (ReLU) → 2 (linear)`
- Learning rate: `0.0001`
- Final training MMSE after 1000 epochs: **~0.07**

### Wine Quality (`~1,120 train / ~480 test samples`, `11 features`, `1 output`)

- Architecture: `11 → 16 (tanh) → 16 (tanh) → 1 (linear)`
- Learning rate: `0.00001`
- Features normalized per column; 70/30 train-test split
- After 200 epochs: train MMSE **~0.26**, test MMSE **~0.25**
- Run `train_wine_quality.py` with 1000 epochs for further convergence

---

## Project structure

```
.
├── NeuralNetwork.py
├── LayerConnection.py
├── layer.py
├── matrix.py
├── vector.py
├── neuron.py
├── mathematics.py
├── tensor.py
├── examples/
│   ├── train_synthetic.py
│   └── train_wine_quality.py
├── data/
│   └── winequality-red.csv
├── assets/
│   ├── synthetic_loss_curve.png
│   └── wine_loss_curve.png
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Design notes

### Built without high-level ML libraries

The network uses custom matrix operations and manual backpropagation. External dependencies are limited to:

- `pandas` for loading the wine dataset
- `matplotlib` for plotting training loss

### Normalization

Input features are min-max normalized **per feature** before training. For the wine dataset, build feature columns first, normalize, then transpose so each row is one sample.

### Batch training

The presentation version uses full-batch gradient descent: all training samples contribute to each weight update step.

---

## Possible extensions

- Mini-batch or stochastic gradient descent
- Separate train/validation tracking in `NeuralNetwork.train()`
- Additional metrics (MAE, R²)
- Save/load model weights
- Comparison benchmark against PyTorch on the same dataset

---

## License

MIT License. See `LICENSE`.
