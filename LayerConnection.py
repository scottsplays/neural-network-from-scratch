"""
Stores the weight matrix and activation for one network layer transition.

Each LayerConnection owns:
  - a weight matrix mapping (previous layer + bias) -> current layer
  - an activation function applied after the affine transform
  - the learning rate used during gradient descent
"""

from mathematics import derivative
from matrix import product


class LayerConnection:

    def __init__(self, weights, activation, learning_rate=0.001):
        self.weights = weights
        self.activation = activation
        self.learning_rate = learning_rate

        self.n = self.weights.get_n()
        self.m = self.weights.get_m()

    def get_weights(self):
        return self.weights

    def get_n(self):
        return self.n

    def get_m(self):
        return self.m

    def get_activation(self):
        return self.activation

    def update(self, grad_layer, previous_layer, pre_activation):
        """
        Apply one backpropagation step for this connection.

        Args:
            grad_layer: upstream gradient (samples x outputs)
            previous_layer: activations feeding into this layer
            pre_activation: layer values before the activation function

        Returns:
            grad_layer propagated to the previous layer
        """
        weights = self.weights.get_copy()

        # Chain rule: multiply upstream gradient by activation derivative.
        pre_act = pre_activation.get_copy()
        pre_act.apply(derivative(self.activation))
        grad_layer.element_multiplication(pre_act)

        # Build input matrix with bias column, then transpose so rows align
        # with weight indices for the batched outer-product gradient.
        layer = previous_layer.get_copy()
        layer.bias()
        layer.transpose()

        # Full-batch gradient: sum of per-sample outer products.
        gradient = product(layer, grad_layer)
        gradient.scalar(self.learning_rate)
        self.weights.subtract(gradient)

        # Propagate gradient to the previous layer (remove bias column last).
        grad_layer = grad_layer.multiply(weights.transpose())
        grad_layer = grad_layer.pop()
        return grad_layer
