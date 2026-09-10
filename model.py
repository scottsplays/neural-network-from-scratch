import mathematics
from mathematics import Loss
from tensor import Vector, Matrix
from layer import Layer, Weight
from visualize import plot

class Model:

    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = Weight(input_size, hidden_size, activation_function=mathematics.relu)
        self.fc2 = Weight(hidden_size, hidden_size, activation_function=mathematics.relu)
        self.fc3 = Weight(hidden_size, output_size, activation_function=mathematics.linear)
        self.weights = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        x = Layer(x)
        out = x.forward(self.fc1)
        out = out.forward(self.fc2)
        out = out.forward(self.fc3)
        out = Layer(out.neurons.softmax())
        return out

    def backward(self, X, y, lr):
        X, y = Layer(X), Layer(y)
        y_pred = self.forward(X)
        loss = Loss(y, y_pred, loss_type=mathematics.CrossEntropyLoss)
        dy = Layer(loss.grad_layer())
        for weight in reversed(self.weights):
            dy = weight.backward(dy, lr)
        return loss.value

    def train(self, X_train, y_train, num_epochs = 100, lr = 0.01, visualize = False, on_epoch = None):
        """Train with existing online SGD; return losses and optionally report epochs."""
        epochs, losses = [], []
        for epoch in range(num_epochs):
            net_loss = 0
            for i, X in enumerate(X_train):
                net_loss += self.backward(X, y_train[i], lr)
            avg_loss = net_loss / len(X_train)
            epochs.append(epoch)
            losses.append(avg_loss)
            if on_epoch is not None:
                on_epoch(epoch + 1, float(avg_loss))
            if((epoch + 1) % 10 == 0):
                print(f"Epoch {epoch + 1}: Loss = {avg_loss}")
        if(visualize):
            plot(epochs, losses, "Epochs", "Losses", "Training Loss over Time")
        return losses


if __name__ == "__main__":
    data = Matrix(50, 2)
    targets = Matrix(50, 2)
    model = Model(2, 16, 2)
