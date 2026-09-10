# Neural Network Framework from Scratch

A neural-network learning and portfolio project built from first principles in Python. The repository contains custom `Vector`, `Matrix`, `Layer`, `Weight`, and model code for forward propagation, backpropagation, gradient-based learning, plus an interactive Streamlit interface called **NeuralLab** for inspecting the math as it runs.

**Author:** [Scott Kennedy](https://github.com/scottsplays)

## Why this project exists

The goal is to understand and demonstrate neural-network mechanics rather than hide them behind a high-level ML framework. Core computations are implemented with custom Python data structures and manual linear algebra/backpropagation.

The repository also retains earlier framework experiments (`NeuralNetwork.py`, `LayerConnection.py`, `matrix.py`, `vector.py`, and related examples) as part of the project's development history.

---

## NeuralLab — interactive Streamlit interface

NeuralLab wraps the newer from-scratch classification stack in a browser interface without replacing its neural-network math with PyTorch, TensorFlow, or Keras.

### Features

- Train the custom network on built-in Iris or Wine classification datasets
- Configure hidden-layer size, epochs, learning rate, split size, and random seed
- Plot cross-entropy training loss from `Model.train()`
- Display train/test accuracy and a confusion matrix
- Visualize the network architecture
- Inspect learned weight matrices as heatmaps and numeric tables
- Select a test sample and inspect its complete forward pass
- View cached `x`, biased input `xb`, weight matrix `W`, pre-activation `z`, and activation `y`
- Inspect final softmax probabilities
- Inspect backpropagation values `dy`, activation derivative, `dz`, `dW`, and `dx`
- Run backprop inspection on a deep copy so visualization does not mutate the trained model

### Run NeuralLab

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

Then choose an experiment in the sidebar and click **Train from scratch**.

---

## NeuralLab architecture

The current interactive classifier uses three dense connections:

```text
Input
  ↓
fc1 + ReLU
  ↓
fc2 + ReLU
  ↓
fc3 + Linear
  ↓
Softmax
  ↓
Class probabilities
```

The hidden width is configurable in the Streamlit UI.

For each dense layer, bias is represented by appending a constant `1` to the input vector:

```text
xb = [x; 1]
z  = xb @ W
y  = f(z)
```

The corresponding `Weight` stores these forward-pass values so the Math Inspector can display the exact numerical values produced by the framework.

Backpropagation follows:

```text
dz = f'(z) * dy
dW = xb.T @ dz
dx = (dz @ W.T) with the bias component removed
W  = W - learning_rate * dW
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for additional implementation notes.

---

## Current NeuralLab modules

| Module | Responsibility |
|---|---|
| `tensor.py` | Custom `Vector` and `Matrix` types and linear-algebra operations |
| `layer.py` | `Layer`, `Weight`, forward caches, and weight updates |
| `mathematics.py` | Activations, derivatives, and loss functions |
| `model.py` | Three-layer classifier, softmax, training loop, and online SGD |
| `visualization_helpers.py` | Safe display conversion plus architecture/activation figures |
| `backprop_inspector.py` | Inspection-only backpropagation snapshots |
| `app.py` | Streamlit experiment and visualization interface |
| `visualize.py` | Basic standalone plotting helper |

The Streamlit/scikit-learn code is used for UI, datasets, preprocessing, splitting, metrics, and plotting. The neural-network forward/backward computations remain in the custom framework.

---

## Earlier framework experiments

The repository also includes the earlier implementation and examples built around:

- `NeuralNetwork.py`
- `LayerConnection.py`
- `matrix.py`
- `vector.py`
- `neuron.py`
- `examples/`
- `data/`
- `assets/`

These files document the evolution of the project and include previous regression experiments such as synthetic data and Wine Quality training.

---

## Project structure

```text
.
├── app.py
├── model.py
├── tensor.py
├── layer.py
├── mathematics.py
├── visualization_helpers.py
├── backprop_inspector.py
├── visualize.py
├── ARCHITECTURE.md
│
├── NeuralNetwork.py          # earlier framework
├── LayerConnection.py        # earlier framework
├── matrix.py                 # earlier framework
├── vector.py                 # earlier framework
├── neuron.py                 # earlier framework
├── examples/
├── data/
├── assets/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Possible next extensions

- Validation-loss tracking during training
- Save/load trained weights
- Mini-batch training
- Additional activation and optimizer choices in NeuralLab
- Side-by-side PyTorch numerical/performance comparison
- More datasets and user-uploaded tabular data
- Interactive neuron/connection highlighting in the architecture view

---

## License

MIT License. See `LICENSE`.
