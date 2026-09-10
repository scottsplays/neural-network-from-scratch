import mathematics

class Vector:

    def __init__(self, inp = None):
        if(isinstance(inp, int)):
            self.vals = mathematics.random_list(inp)
            self.n = inp
        else:
            self.vals = [] if inp is None else inp
            self.n = len(self.vals)

    def __str__(self):
        strings = [str(val) for val in self.vals]
        return " ".join(strings)

    def __getitem__(self, i):
        return self.vals[i]

    def __len__(self):
        return self.n

    def to_list(self):
        return [float(value) for value in self.vals]

    def __add__(self, v):
        if(self.n != v.n):
            raise ValueError(f"Size Mismatch for Vector Addition: {self.shape} + {v.shape}")
        temp = []
        for i in range(self.n):
            temp.append(self[i] + v[i])
        return Vector(temp)

    def __sub__(self, v):
        if(self.n != v.n):
            raise ValueError(f"Size Mismatch for Vector Subtraction: {self.shape} - {v.shape}")
        temp = []
        for i in range(self.n):
            temp.append(self[i] - v[i])
        return Vector(temp)

    def __mul__(self, v):
        size = v.n if isinstance(v, Vector) else self.n
        if(self.n != size):
            raise ValueError(f"Size Mismatch for Vector Multiplication: {self.shape} x {size}")
        temp = []
        for i in range(self.n):
            mult = v[i] if isinstance(v, Vector) else v
            temp.append(self[i] * mult)
        return Vector(temp)

    def __rmul__(self, v):
        return self * v

    def __matmul__(self, v):
        if(isinstance(v, Matrix)):
            self = Matrix([self])
            return self @ v
        if self.n != v.n:
            raise ValueError(f"Size mismatch for Dot Product: {self.shape} * {v.shape}")
        sum = 0
        for i, val in enumerate(self.vals):
            sum += val * v[i]
        return sum

    def transpose(self):
        mat = Matrix([self])
        return mat.transpose()

    def apply(self, function):
        temp = []
        for val in self.vals:
            temp.append(function(val))
        return Vector(temp)

    @property
    def shape(self):
        return (self.n,)

    def append(self, val):
        self.vals.append(val)
        self.n += 1

    def fill(self, num):
        self.vals = [num] * self.n
        return Vector(self.vals)

    def copy(self):
        return Vector(self.vals.copy())

    def softmax(self):
        temp = []
        max_val = max(self.vals)
        exp_vals = [mathematics.exp(val - max_val) for val in self.vals]
        exp_sum = sum(exp_vals)
        for val in exp_vals:
            temp.append(val / exp_sum)
        return Vector(temp)

    def argmax(self):
        max_arg = -1
        max_elem = float("-infinity")
        for i in range(len(self)):
            if(self[i] > max_elem):
                max_arg = i
                max_elem = self[i]
        return max_arg

    def bias(self):
        v = self.copy()
        v.append(1)
        return v

    def unbias(self):
        v = self.copy().vals
        v.pop()
        return Vector(v)


class Matrix:

    def to_list(self):
        return [row.to_list() for row in self.vectors]

    def __init__(self, inp = None, m = 0):
        if(isinstance(inp, int)):
            self.vectors = [Vector(mathematics.random_list(m)) for _ in range(inp)]
            self.n = inp
            self.m = m
        else:
            if inp is None:
                inp = [Vector()]
            self.vectors = inp if isinstance(inp[0], Vector) else [Vector(i) for i in inp]
            self.n = len(inp)
            self.m = self.vectors[0].n

    def __str__(self):
        st = ""
        for vector in self.vectors:
            st += str(vector) + "\n"
        return st

    def __getitem__(self, i):
        if(isinstance(i, tuple)):
            return self.vectors[i[0]][i[1]]
        return self.vectors[i]

    def __len__(self):
        return self.n

    def transpose(self):
        temp = []
        for i in range(self.m):
            temp_row = []
            for j in range(self.n):
                temp_row.append(self[j][i])
            temp.append(Vector(temp_row))
        return Matrix(temp)

    def column(self, j):
        temp = []
        for i in range(self.n):
            temp.append(self[i][j])
        return Vector(temp)

    def __add__(self, M):
        if self.shape != M.shape:
            raise ValueError(f"Size Mismatch for Matrix Addition: {self.shape} + {M.shape}")
        temp = []
        for i in range(self.n):
            temp.append(self[i] + M[i])
        return Matrix(temp)

    def __sub__(self, M):
        if self.shape != M.shape:
            raise ValueError(f"Size Mismatch for Matrix Subtraction: {self.shape} - {M.shape}")
        temp = []
        for i in range(self.n):
            temp.append(self[i] - M[i])
        return Matrix(temp)

    def __mul__(self, M):
        size = M.shape if isinstance(M, Matrix) else self.shape
        if self.shape != size:
            raise ValueError(f"Size Mismatch for Element-Wise Matrix Multiplication: {self.shape} x {size}")
        temp = []
        for i, row in enumerate(self.vectors):
            mult = M[i] if isinstance(M, Matrix) else M
            temp.append(row * mult)
        return Matrix(temp)

    def __rmul__(self, M):
        return self * M

    def __matmul__(self, M):
        if(isinstance(M, Vector)):
            M = Matrix([M])
            M = M.transpose()
            return self @ M
        if self.m != M.n:
            raise ValueError(f"Size Mismatch for Matrix Multiplication: {self.shape} x {M.shape}")
        temp = []
        M_T = M.transpose()
        for i in range(self.n):
            temp_row = []
            for j in range(M.m):
                v1 = self.vectors[i]
                v2 = M_T[j]
                temp_row.append(v1 @ v2)
            temp.append(Vector(temp_row))
        return Matrix(temp)

    @property
    def shape(self):
        return (self.n, self.m)

    def append(self, v):
        if not isinstance(v, Vector):
            v = Vector(v)
        if(self.m != v.n):
            raise ValueError(f"Size Mismatch for Matrix Append: {self.shape} += {v.shape}")
        self.vectors.append(v)
        self.n += 1
        if(self.m == 0):
            self.m = v.n

    def flatten(self):
        temp = []
        for i in range(self.n):
            row = self[i]
            for j in range(self.m):
                element = row[j]
                temp.append(element)
        return Vector(temp)

    def fill(self, num):
        for vector in self.vectors:
            vector.fill(num)
        return Matrix(self.vectors)

    def copy(self):
        return Matrix([row.copy() for row in self.vectors])

    def bias(self):
        M = self.copy()
        temp = []
        for row in M.vectors:
            temp.append(row.bias())
        return Matrix(temp)

    def unbias(self):
        M = self.copy().vectors
        temp = []
        for vector in M:
            temp.append(vector.unbias())
        return Matrix(temp)
