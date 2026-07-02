"""
Custom 2-D matrix type built from Vector rows.

Convention used throughout this project:
  - rows (n) = number of samples
  - columns (m) = number of features or outputs per sample

Matrix(n, m) with integer arguments creates an n x m matrix of random values.
"""

from vector import Vector, zero_vector, MSE


class Matrix:

    def __init__(self, vectors=None, m=1):
        if vectors is None:
            vectors = []
        if type(vectors) == int:
            n = vectors
            self.rows = [Vector(m) for _ in range(n)]
            self.n = n
            self.m = m
        else:
            self.rows = vectors
            self.n = len(vectors)
            self.m = vectors[0].get_size() if self.n > 0 else 0

    def randomize(self):
        for row in self.rows:
            row.randomize()

    def as_lists(self):
        return self.rows

    def get_n(self):
        return self.n

    def get_m(self):
        return self.m

    def append(self, row):
        self.rows.append(row)
        self.n += 1
        self.m = row.get_size()

    def get_row(self, i):
        return self.rows[i]

    def get_col(self, i):
        return Vector([row.get_element(i) for row in self.rows])

    def get_element(self, i, j):
        return self.get_row(i).get_element(j)

    def transpose(self):
        """Swap row/column interpretation (features-as-rows <-> samples-as-rows)."""
        temp = [self.get_col(i) for i in range(self.m)]
        self.n, self.m = self.m, self.n
        self.rows = temp
        return Matrix(temp)

    def scalar(self, s):
        for row in self.rows:
            row.scalar(s)
        return Matrix(self.rows)

    def add(self, other):
        if other.get_n() != self.n or other.get_m() != self.m:
            raise ValueError("Matrix sizes do not match for addition")
        self.rows = [self.get_row(i).add(other.get_row(i)) for i in range(self.n)]
        return Matrix(self.rows)

    def subtract(self, other):
        if other.get_n() != self.n or other.get_m() != self.m:
            raise ValueError("Matrix sizes do not match for subtraction")
        self.rows = [
            self.get_row(i).subtract(other.get_row(i).get_copy())
            for i in range(self.n)
        ]
        return Matrix(self.rows)

    def multiply(self, other):
        """Standard matrix multiplication."""
        if self.m != other.get_n():
            raise ValueError("Matrix sizes do not match for multiplication")
        temp = []
        for i in range(self.n):
            row = Vector([])
            for j in range(other.get_m()):
                row.append(self.get_row(i).dot_product(other.get_col(j).get_copy()))
            temp.append(row)
        self.m = other.get_m()
        self.rows = temp
        return Matrix(temp)

    def apply(self, func):
        self.rows = [row.get_copy().apply(func) for row in self.rows]
        return Matrix(self.rows)

    def bias(self):
        """Append a bias column of ones to every sample row."""
        for row in self.rows:
            row.append(1)
        self.m += 1
        return Matrix(self.rows)

    def normalize(self):
        """
        Min-max normalize each row independently.

        When rows represent feature columns (before transpose), this scales
        each feature across all samples to [0, 1].
        """
        self.rows = [row.normalize() for row in self.rows]
        return Matrix(self.rows)

    def get_copy(self):
        return Matrix([row.get_copy() for row in self.rows])

    def pop(self):
        """Drop the last column from every row (remove bias during backprop)."""
        for row in self.rows:
            row.pop()
        self.m -= 1
        return Matrix(self.rows)

    def element_multiplication(self, other):
        """Element-wise product — applies activation derivatives to gradients."""
        self.rows = [
            self.get_row(i).multiply(other.get_row(i))
            for i in range(len(self.rows))
        ]
        return Matrix(self.rows)


def MMSE(predictions, targets):
    """
    Mean squared error across an entire batch matrix.

    Returns the average per-element squared error, which is the scalar
    loss printed during training.
    """
    total = 0
    for i in range(predictions.get_n()):
        total += MSE(predictions.get_row(i), targets.get_row(i))
    return total / (predictions.get_n() * predictions.get_m())


def product(a, b):
    """Matrix multiply, promoting a lone Vector to a 1 x n matrix when needed."""
    if type(a) == Vector:
        a = Matrix([a.get_copy()])
    return a.multiply(b)


def col_as_vector(matrix):
    """Extract the first column as a vector (single-output layers)."""
    return matrix.get_col(0)


def zero_matrix(n, m):
    return Matrix([zero_vector(m) for _ in range(n)])
