# Neural Network Framework from Scratch

A neural-network learning and portfolio project built from first principles in Python. The repository contains custom `Vector`, `Matrix`, `Layer`, `Weight`, and model code for forward propagation, backpropagation, gradient-based learning, plus an interactive Streamlit interface called **NeuralLab** for inspecting the math as it runs.

This neural-network framework is also the foundation for a larger long-term goal: **designing a physics-specialized Large Language Model from scratch in the same first-principles manner**. Rather than beginning with a high-level deep-learning framework, the project is intended to grow step by step from custom linear algebra and neural-network primitives toward embeddings, attention, Transformers, language modeling, and eventually a model specialized for mathematical and physics-oriented reasoning.

**Author:** [Scott Kennedy](https://github.com/scottsplays)

## Why this project exists

The immediate goal is to understand and demonstrate neural-network mechanics rather than hide them behind a high-level ML framework. Core computations are implemented with custom Python data structures and manual linear algebra/backpropagation.

The broader goal is to use that understanding as the base of a progressively more capable AI architecture. NeuralLab provides a way to inspect the mechanics at the neural-network level today; future stages will apply the same philosophy to the components required for a small language model and, ultimately, a **physics-specialized LLM built from scratch**.

The repository also retains earlier framework experiments (`NeuralNetwork.py`, `LayerConnection.py`, `matrix.py`, `vector.py`, and related examples) as part of the project's development history.

---

## Long-term vision: Physics-Specialized LLM from Scratch

The final ambition of this project is not simply to create a tabular classifier. The classifier and NeuralLab interface are an experimental foundation for understanding and implementing the machinery that larger neural architectures depend on.

The intended progression is:

```text
Custom Vector / Matrix mathematics
              ↓
Dense neural networks + backpropagation
              ↓
Embeddings
              ↓
Self-attention
              ↓
Multi-head attention
              ↓
Transformer blocks
              ↓
Tokenization + language modeling
              ↓
Small from-scratch Transformer language model
              ↓
Physics / mathematics specialization
              ↓
Physics-Specialized LLM
```

The same principle should remain consistent throughout the project: **implement the important mechanisms directly, understand the mathematics behind them, visualize what the model is doing, and only use higher-level libraries where they are supporting the experiment rather than replacing the architecture being studied.**

A future physics-focused model could be trained and evaluated on mathematical and physics-oriented text and problems, with experiments designed to measure how well the architecture learns relationships between equations, concepts, and natural-language explanations. The exact LLM architecture, dataset, scale, and training strategy are future work; the current repository should therefore be viewed as the first stage of that larger research/engineering project rather than a completed LLM implementation.

---

## NeuralLab — interactive Streamlit interface

NeuralLab wraps the newer from-scratch classification stack in a browser interface without replacing its neural-network math with PyTorch, TensorFlow, or Keras.

### Features

- Train the custom network on built-in Iris or Wine classification datasets
- **Upload your own CSV classification dataset**
- Choose the target column and numeric input features interactively
- Automatically map text or numeric class labels to output neurons
- Validate missing values, non-finite inputs, class counts, and train/test split feasibility before training
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

### Custom CSV format

Choose **Upload CSV** in the dataset selector. NeuralLab will preview the file and let you select:

1. A target column for classification
2. One or more numeric feature columns

For example:

```csv
height,weight,wingspan,species
14.2,2.8,22.1,Robin
11.7,1.9,18.4,Finch
16.0,3.2,24.5,Robin
12.1,2.0,19.0,Finch
```

The current CSV workflow is intentionally focused on **classification**. Input features must be numeric, while the target labels may be text or numeric. NeuralLab standardizes the selected features using statistics fitted only on the training split, then dynamically creates a model with the required number of input and output neurons.

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

The input width is determined by the selected dataset/features, the hidden width is configurable in the Streamlit UI, and the output width is determined by the number of target classes.

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
| `app.py` | Streamlit datasets, CSV upload, training, metrics, and visualization interface |
| `visualize.py` | Basic standalone plotting helper |

The Streamlit/scikit-learn code is used for UI, built-in datasets, preprocessing, splitting, metrics, and plotting. The neural-network forward/backward computations remain in the custom framework.

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

## Roadmap

### Neural-network / NeuralLab improvements

- Validation-loss tracking during training
- Save/load trained weights
- Mini-batch training
- Additional activation and optimizer choices in NeuralLab
- Side-by-side PyTorch numerical/performance comparison
- Categorical feature preprocessing for uploaded CSVs
- Regression mode for custom datasets
- Interactive neuron/connection highlighting in the architecture view

### Toward the Physics-Specialized LLM

- Implement an embedding layer using the custom tensor/matrix abstractions
- Implement scaled dot-product self-attention
- Extend self-attention into multi-head attention
- Build Transformer feed-forward, normalization, and residual components
- Assemble Transformer blocks from the custom components
- Add tokenization and sequence-processing infrastructure
- Train a small language model before attempting domain specialization
- Build a physics/mathematics corpus and evaluation set
- Experiment with physics-focused training and specialization
- Compare the from-scratch implementation with an equivalent framework-based Transformer to study correctness, speed, and scaling tradeoffs
- Extend NeuralLab-style visualization to embeddings, attention maps, Transformer activations, and language-model predictions

---

## License

MIT License. See `LICENSE`.
