import numpy as np

class Variable:

    def __init__(self, value = None):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __add__(self, x):
        return Variable(self.value + x.value)

    def __sub__(self, x):
        return Variable(self.value - x.value)

    def __mul__(self, x):
        return Variable(self.value * x.value)

    def __eq__(self, v):
        self.value = v

class Expression:

    def __init__(self, exp):
        self.str = exp
        self.exp = self.parser(exp)
        self.variables = []

    def parser(self, s):
        s = s.replace(" ", "")
        operations = ['(', ')', '^', '*', '/', '+', '-']
        for c in s:
            if(int(c)):
                print("Yes")

    def __str__(self):
        return

def random():
    return np.random.random()

def random_list(n):
    temp = []
    for _ in range(n):
        temp.append(random())
    return temp

def random_ints(n, a, b):
    temp = []
    for _ in range(n):
        r = random()
        temp.append(int(a + (b - a + 1) * r))
    return temp

def linear(x):
    return x

def one(x):
    return 1

def exp(x):
    return np.exp(x)

def relu(x):
    return  x / 2 + np.abs(x) / 2

def drelu(x):
    return 0 if x < 0 else 1

def derivative(function):
    if(function == linear):
        return one
    if(function == relu):
        return drelu

def MSELoss(y_actual, y_pred):
    sum = 0
    n = len(y_actual)
    for i, val in enumerate(y_actual):
        sum += (val - y_pred[i]) ** 2
    return sum / n

def dMSELoss(y_actual, y_pred):
    temp = []
    n = len(y_actual)
    for i, val in enumerate(y_actual):
        temp.append(2 * (val - y_pred[i]) / n)
    return temp

def CrossEntropyLoss(y_actual, y_pred):
    sum = 0
    for i, val in enumerate(y_actual):
        sum -= val * np.log(max(y_pred[i], 1e-8))
    return sum

def dCrossEntropyLoss(y_actual, y_pred):
    temp = []
    for i, val in enumerate(y_actual):
        temp.append(y_pred[i] - val)
    return temp

def softmax_gradient(y_actual, y_pred):
    return [actual - pred for actual, pred in zip(y_actual, y_pred)]

class Loss:

    def __init__(self, x, y, loss_type = MSELoss):
        self.loss_type = loss_type
        self.x = x.neurons.vals
        self.y = y.neurons.vals

    @property
    def value(self):
        return self.loss_type(self.x, self.y)

    def grad_layer(self):
        if(self.loss_type == MSELoss):
            return dMSELoss(self.x, self.y)
        if(self.loss_type == CrossEntropyLoss):
            return dCrossEntropyLoss(self.x, self.y)
