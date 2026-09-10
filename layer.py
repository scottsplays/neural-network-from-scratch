import mathematics
from tensor import Vector, Matrix

class Layer:

    def __init__(self, inp = None):
        if isinstance(inp, Layer):
            self.neurons = inp.neurons
            return
        if isinstance(inp, Matrix):
            inp = inp.flatten()
        if isinstance(inp, float):
            inp = [inp]
        self.neurons = inp if isinstance(inp, Vector) else Vector(inp)

    def __str__(self):
        return str(self.neurons)

    def __add__(self, l):
        return Layer(self.neurons + l.neurons)

    def __sub__(self, l):
        return Layer(self.neurons - l.neurons)

    def __mul__(self, l):
        mult = l.neurons if isinstance(l, Layer) else l
        return Layer(self.neurons * mult)

    def __rmul__(self, l):
        return self * l

    def __matmul__(self, M):
        mult = Matrix([M.neurons]) if isinstance(M, Layer) else M
        return self.neurons @ mult

    def as_list(self):
        return self.neurons.vals

    def to_list(self):
        """Return detached activation values using the Vector conversion API."""
        return self.neurons.to_list()

    def copy(self):
        return Layer(self.neurons.copy())

    def activation(self, function):
        temp = []
        for neuron in self.neurons.vals:
            temp.append(function(neuron))
        return Layer(temp)

    @property
    def shape(self):
        return self.neurons.shape

    def bias(self):
        v = self.neurons.copy()
        v.append(1)
        return Layer(v)

    def forward(self, weight):
        W = weight.weight_matrix
        activation = weight.activation_function
        v = self.neurons
        x = v.copy()
        if(weight.bias):
            v = v.bias()
        prod = v @ W
        prod = Layer(prod)
        z = prod.copy()
        y = prod.activation(activation)
        weight.save_layers(x, z, y)
        return y

class Weight:

    def __init__(self, n, m, activation_function = mathematics.linear, bias = True):
        if bias:
            n += 1
        self.weight_matrix = Matrix(n, m)
        self.activation_function = activation_function
        self.bias = bias

    def save_layers(self, x, z, y):
        self.x = x
        self.xb = x.bias()
        self.z = z
        self.y = y

    def backward(self, dy, lr):
        d_activation = mathematics.derivative(self.activation_function)
        dz = self.z.activation(d_activation) * dy
        dx = (dz @ self.weight_matrix.transpose()).unbias()
        dW = self.xb.transpose() @ Matrix([dz.neurons])
        self.weight_matrix = self.weight_matrix - lr * dW
        return Layer(dx)
