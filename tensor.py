"""
Lightweight container for an ordered stack of matrices.

Used during training to store forward-pass layer snapshots and
pre-activation values that backpropagation reads in reverse order.
"""

from matrix import Matrix


class Tensor:

    def __init__(self, matrices=None, rows=1, cols=1):
        if matrices is None:
            matrices = []
        if type(matrices) == int:
            count = matrices
            self.matrices = [Matrix(rows, cols) for _ in range(count)]
            self.n = count
        else:
            self.matrices = matrices
            self.n = len(matrices)

    def append(self, matrix):
        self.matrices.append(matrix)
        self.n += 1
        return self

    def get_matrix(self, index):
        return self.matrices[index]

    def reverse(self):
        self.matrices = self.matrices[::-1]
