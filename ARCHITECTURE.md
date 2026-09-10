# Architecture & Math Notes

This document explains the core building blocks of the framework and the mathematics used during training. Implementation lives in `tensor.py`, `layer.py`, `model.py`, and `mathematics.py`.

## Vector

A **Vector** is a 1-D list of numbers with length `n`.

- Element-wise `+`, `-`, `*`
- Dot product and matrix multiply helpers (`@`)
- Utilities: `copy`, `bias` / `unbias`, `apply`, `softmax`, `argmax`

Appending a constant `1` to an activation vector lets the next weight matrix absorb a bias term as an extra row, so `z = xb W` includes bias without a separate bias vector.

## Matrix

A **Matrix** is a list of row `Vector`s with shape `(n, m)`. A dense layer with bias stores `W` with shape `(d_in + 1, d_out)` and multiplies using the row-vector convention `z = xb W`.

## Layer

A **Layer** wraps a `Vector` of activations for one forward step. It holds activations, applies element-wise activations, and runs `forward(weight)` using bias augmentation, matrix multiplication, and activation.

For each layer:

`xb = [x; 1]`

`z = xb W`

`y = f(z)`

Intermediate values `x`, `z`, and `y` are cached on the corresponding `Weight` for backpropagation.

## Weight

A **Weight** owns one dense connection: matrix `W`, activation `f`, and bias flag.

Forward cache:
- `x`: input activations
- `xb`: input with bias
- `z`: pre-activation
- `y`: post-activation

Backward:

`dz = f'(z) ⊙ dy`

`dW = xb^T dz`

`dx = (dz W^T)_drop_bias`

`W = W - learning_rate * dW`

## Model

`Model` stacks three dense layers:

`input -> ReLU hidden -> ReLU hidden -> linear logits -> softmax`

Cross-entropy is used for classification, and training uses stochastic / online SGD with one sample update at a time.

## Streamlit interface

The Streamlit layer does not replace the framework math. It adds dataset loading/preprocessing, training controls, metrics, and inspection views for the values produced by the custom classes.
