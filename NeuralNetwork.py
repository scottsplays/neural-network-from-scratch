"""
Feedforward neural network with manual backpropagation.

This is the top-level API. A network is built by:
  1. Passing input/output matrices to __init__
  2. Optionally calling add_layer() to insert hidden layers
  3. Calling train() to run full-batch gradient descent
"""

import matplotlib.pyplot as plt

import mathematics
from matrix import Matrix, MMSE
from layer import Layer, LayerTensor, matrix_to_layer_matrix
from LayerConnection import LayerConnection


class NeuralNetwork:
    """
    Fully-connected feedforward network implemented without ML frameworks.

    Matrix layout:
      inputs  — (num_samples x num_features)
      outputs — (num_samples x num_outputs)
    """

    def __init__(self, inputs, outputs, function=mathematics.linear, learning_rate=0.001):
        self.inputs = inputs
        self.outputs = outputs
        self.learning_rate = learning_rate

        # n = feature count, size = sample count, m = output dimension
        self.n = self.inputs.get_m()
        self.size = self.inputs.get_n()
        self.m = self.outputs.get_m()

        self.initial_weights = Matrix(self.n + 1, self.m)
        self.final_activation = function

        # Start with a single output connection; hidden layers are added later.
        self.connections = [
            LayerConnection(self.initial_weights, mathematics.linear, learning_rate)
        ]

    def add_layer(self, hidden_size, activation):
        """
        Insert a hidden layer before the output connection.

        Rewires the last connection into a hidden layer and appends a new
        output connection that preserves the original output activation.
        """
        output_weights = Matrix(hidden_size + 1, self.m)
        output_connection = LayerConnection(
            output_weights, self.final_activation, self.learning_rate
        )

        previous = self.connections[-1]
        hidden_weights = Matrix(previous.get_n(), hidden_size)
        hidden_connection = LayerConnection(
            hidden_weights, activation, self.learning_rate
        )

        self.connections.pop()
        self.connections.append(hidden_connection)
        self.connections.append(output_connection)

    def evaluate(self, sample):
        """Run a forward pass for a single Layer (inference, no training state)."""
        layer = Layer(sample)
        for connection in self.connections:
            layer = layer.next_layer(connection)
        return layer

    def forward_pass(self, input_batch):
        """Propagate an entire batch forward, recording intermediate states."""
        layer = input_batch
        for connection in self.connections:
            layer = layer.next_layer(connection, self.pre_activations)
            self.layers.append(layer.get_copy())
        return layer.as_matrix()

    def backward_propagation(self, predictions):
        """
        Reverse through the network and update every connection's weights.

        Requires self.layers and self.pre_activations from the latest
        forward pass.
        """
        self.layers.reverse()
        self.connections.reverse()
        self.pre_activations.reverse()

        # Output-layer error signal.
        grad_layer = predictions.subtract(self.outputs.get_copy())

        for i, connection in enumerate(self.connections):
            previous_activations = self.layers.get_matrix(i + 1)
            pre_activation = self.pre_activations.get_matrix(i)

            grad_layer = connection.update(
                grad_layer,
                previous_activations.as_matrix(),
                pre_activation.as_matrix(),
            )

        # Restore connection order for the next epoch.
        self.connections.reverse()

    def train(self, epochs=1000, log_every=100, plot_path=None, show_plot=False):
        """
        Train with full-batch gradient descent.

        Args:
            epochs: number of weight-update steps
            log_every: print loss every N epochs
            plot_path: optional path to save a loss curve PNG
            show_plot: whether to open an interactive matplotlib window

        Returns:
            list of loss values (one per epoch, including epoch 0)
        """
        inputs = self.inputs.get_copy().normalize()
        input_batch = matrix_to_layer_matrix(inputs)

        epoch_vals = []
        loss_vals = []

        for epoch in range(epochs + 1):
            self.layers = LayerTensor([input_batch])
            self.pre_activations = LayerTensor()

            predictions = self.forward_pass(input_batch)
            loss = MMSE(predictions, self.outputs.get_copy())

            epoch_vals.append(epoch)
            loss_vals.append(loss)

            if epoch % log_every == 0:
                print(f"Epoch {epoch:4d} | Loss = {loss:.6f}")

            if epoch < epochs:
                self.backward_propagation(predictions)

        if plot_path or show_plot:
            plt.figure(figsize=(8, 4))
            plt.plot(epoch_vals, loss_vals)
            plt.xlabel("Epoch")
            plt.ylabel("Mean Squared Error")
            plt.title("Training Loss")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if plot_path:
                plt.savefig(plot_path, dpi=150)
                print(f"Saved loss curve to {plot_path}")

            if show_plot:
                plt.show()
            else:
                plt.close()

        return loss_vals
