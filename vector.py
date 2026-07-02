"""
Custom 1-D vector type used as the building block for all matrix math.

Vectors support element-wise operations, dot products, and min-max
normalization. Passing an integer to the constructor creates a vector
of random values in [-1, 1], which is useful for synthetic datasets.
"""

import mathematics


class Vector:

    def __init__(self, elements=None):
        if elements is None:
            elements = []
        if type(elements) == int:
            # Matrix(n, m) calls Vector(m) to build random rows.
            self.elements = [
                mathematics.random_number(-1, 1) for _ in range(elements)
            ]
            self.n = elements
        else:
            self.elements = elements
            self.n = len(elements)

    def add(self, v):
        if self.n != v.get_size():
            raise ValueError("Vector sizes do not match for addition")
        self.elements = [self.elements[i] + v.get_element(i) for i in range(self.n)]
        return Vector(self.elements)

    def subtract(self, v):
        if self.n != v.get_size():
            raise ValueError("Vector sizes do not match for subtraction")
        self.elements = [self.get_element(i) - v.get_element(i) for i in range(self.n)]
        return Vector(self.elements)

    def as_list(self):
        return self.elements

    def dot_product(self, v):
        if self.get_size() != v.get_size():
            raise ValueError("Vector sizes do not match for dot product")
        return sum(self.get_element(i) * v.get_element(i) for i in range(self.n))

    def append(self, x):
        self.elements.append(x)
        self.n += 1

    def pop(self):
        """Remove the last element (used when stripping the bias term)."""
        self.elements = self.elements[:-1]
        self.n -= 1
        return Vector(self.elements)

    def get_element(self, i):
        return self.elements[i]

    def get_size(self):
        return self.n

    def scalar(self, s):
        self.elements = [element * s for element in self.elements]
        return Vector(self.elements)

    def get_copy(self):
        return Vector(self.elements.copy())

    def multiply(self, v):
        """Element-wise (Hadamard) product — used in gradient masking."""
        self.elements = [self.get_element(i) * v.get_element(i) for i in range(self.n)]
        return Vector(self.elements)

    def bias(self):
        """Append a constant 1 for the bias term in affine transforms."""
        self.elements.append(1)

    def apply(self, func):
        self.elements = [func(element) for element in self.elements]
        return Vector(self.elements)

    def normalize(self):
        """Min-max scale this vector to [0, 1]."""
        minimum, maximum = min(self.elements), max(self.elements)
        span = maximum - minimum
        if span == 0:
            self.elements = [0.0] * self.n
            return Vector(self.elements)
        self.elements = [(element - minimum) / span for element in self.elements]
        return Vector(self.elements)

    def randomize(self, a=0, b=1):
        self.elements = [mathematics.random_number(a, b) for _ in range(self.n)]


def zero_vector(n):
    return Vector([0.0] * n)


def MSE(v1, v2):
    """Mean squared error for a single vector pair (one sample)."""
    total = sum(
        (v1.get_element(i) - v2.get_element(i)) ** 2
        for i in range(v1.get_size())
    )
    return total / 2
