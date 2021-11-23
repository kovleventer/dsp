import numpy as np

class csArray():
    def __init__(self, array):
        self.array = np.array(array, dtype=np.csingle)

    def __add__(self, other):
        if type(other) == csArray:
            other = other.array
        return csArray(self.array + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if type(other) == csArray:
            other = other.array
        return csArray(self.array * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if type(other) == csArray:
            other = other.array
        return csArray(self.array - other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __neg__(self):
        return -1 * self

    def __matmul__(self, other):
        return csArray(self.array @ other)

    def T(self):
        return csArray(self.array.T)

    def conj(self):
        return csArray(self.array.conj())

def csZeros(shape):
    return csArray(np.zeros(shape))

def csExp(arr):
    return csArray(np.exp(arr.array))

def csSum(arr):
    return np.sum(arr.array).astype(np.csingle)

def csDot(arr1, arr2):
    return csArray(np.inner(arr1.array, arr2.array))

