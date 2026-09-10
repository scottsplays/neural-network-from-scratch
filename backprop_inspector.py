"""Inspection-only backpropagation using the equations in Weight.backward."""
from copy import deepcopy
import mathematics
from layer import Layer
from tensor import Matrix
from visualization_helpers import display_values


def backward_snapshot(model, sample, target):
    shadow = deepcopy(model)
    prediction = shadow.forward(sample)
    loss = mathematics.Loss(Layer(target), prediction, loss_type=mathematics.CrossEntropyLoss)
    dy = Layer(loss.grad_layer())
    steps = []
    for index in reversed(range(len(shadow.weights))):
        weight = shadow.weights[index]
        derivative = weight.z.activation(mathematics.derivative(weight.activation_function))
        dz = derivative * dy
        dx = Layer((dz @ weight.weight_matrix.transpose()).unbias())
        dW = weight.xb.transpose() @ Matrix([dz.neurons])
        steps.append(dict(layer=f'fc{index+1}', dy=display_values(dy),
            derivative=display_values(derivative), dz=display_values(dz),
            dx=display_values(dx), dW=display_values(dW), xb=display_values(weight.xb)))
        dy = dx
    return float(loss.value), steps
