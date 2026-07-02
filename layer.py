"""
Layer abstractions for single samples, batches, and training history.

Layer       — one sample (a vector of neurons)
LayerMatrix — a batch of samples processed in parallel
LayerTensor — ordered list of LayerMatrix snapshots during training
"""

from vector import Vector
from matrix import Matrix, product, col_as_vector
from neuron import Neuron


class Layer:
    """A single input or hidden representation (one sample)."""

    def __init__(self, neurons=None):
        if neurons is None:
            neurons = []
        if type(neurons) == int:
            self.neurons = [Neuron() for _ in range(neurons)]
            self.n = neurons
        else:
            self.neurons = neurons
            self.n = len(neurons)

    def get_copy(self):
        return Layer(self.neurons)

    def apply(self, function):
        for neuron in self.neurons:
            neuron.apply(function)

    def as_vector(self):
        return Vector(self.get_values())

    def bias(self):
        self.neurons.append(Neuron(1))
        return self

    def get_values(self):
        return [neuron.get_value() for neuron in self.neurons]

    def next_layer(self, connection, pre_activations=None):
        weights = connection.get_weights()
        activation = connection.get_activation()

        # Affine step: [features, 1] @ weights
        values = self.as_vector().get_copy()
        values.bias()
        prod = product(values, weights)
        predictions = col_as_vector(prod)

        next_layer = vector_to_layer(predictions)

        # Store pre-activation values for the backward pass.
        if pre_activations is not None:
            pre_activations.append(next_layer)

        next_layer.apply(activation)
        return next_layer


class LayerMatrix:
    """Batch of layers — one layer (sample) per row in the input matrix."""

    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []

    def get_copy(self):
        return LayerMatrix(self.layers)

    def apply(self, function):
        for layer in self.layers:
            layer.apply(function)

    def bias(self):
        for layer in self.layers:
            layer.bias()
        return self

    def as_matrix(self):
        return Matrix([layer.as_vector() for layer in self.layers])

    def next_layer(self, connection, pre_activations=None):
        weights = connection.get_weights()
        activation = connection.get_activation()

        values = self.as_matrix().get_copy()
        values.bias()

        # Batch forward pass: (samples x features+bias) @ (features+bias x outputs)
        predictions = product(values, weights)
        next_layer = matrix_to_layer_matrix(predictions)

        if pre_activations is not None:
            pre_activations.append(next_layer)

        next_layer.apply(activation)
        return next_layer


class LayerTensor:
    """Container for layer snapshots collected during forward propagation."""

    def __init__(self, matrices=None):
        self.matrices = matrices if matrices is not None else []

    def append(self, matrix):
        self.matrices.append(matrix)

    def get_matrix(self, i):
        return self.matrices[i]

    def reverse(self):
        self.matrices = self.matrices[::-1]
        return self


def vector_to_layer(vector):
    return Layer([Neuron(value) for value in vector.as_list()])


def matrix_to_layer_matrix(matrix):
    return LayerMatrix([vector_to_layer(row) for row in matrix.as_lists()])
