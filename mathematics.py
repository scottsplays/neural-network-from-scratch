"""
Activation functions and their derivatives for forward/backward passes.

Each activation has a matching derivative used during backpropagation.
The derivative() helper maps an activation function to its gradient rule.
"""

import math
import random


def squared(x):
    return x ** 2


def sigmoid(x):
    """Maps any real value to the open interval (0, 1)."""
    return 1 / (1 + math.exp(-1 * x))


def dsigmoid(x):
    """Derivative of sigmoid with respect to pre-activation."""
    return sigmoid(x) * (1 - sigmoid(x))


def random_number(a=0, b=1):
    return a + (b - a) * random.random()


def tanh(x):
    """Maps any real value to the open interval (-1, 1)."""
    exp_p = math.exp(x)
    exp_m = 1 / exp_p
    return (exp_p - exp_m) / (exp_p + exp_m)


def dtanh(x):
    return 1 - tanh(x) ** 2


def random_list(n):
    return [random_number() for _ in range(n)]


def RELU(x):
    """Rectified linear unit — common default for hidden layers."""
    return x if x > 0 else 0


def dRELU(x):
    return 1 if x > 0 else 0


def linear(x):
    """Identity activation — typical choice for regression outputs."""
    return x


def one(x):
    """Derivative of the linear activation."""
    return 1


def derivative(function):
    """Return the derivative function that matches a given activation."""
    if function == RELU:
        return dRELU
    if function == sigmoid:
        return dsigmoid
    if function == tanh:
        return dtanh
    if function == linear:
        return one
    return one
