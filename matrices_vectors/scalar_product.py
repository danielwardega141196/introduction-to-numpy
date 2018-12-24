"""
    Iloczyn skalarny
    Mając wektory n - elementowe a i b napisać funkcję obliczającą ich iloczyn skalarny.

    1) za pomocą iteracji po elementach
    2) używając funkcjonalności numpy

"""
import numpy as np

def s1(a, b):
    n = len(a)
    dot_product = 0

    # Mnożenie elementów o tych samych indeksach i sumowanie tych iloczynów
    for idx in range(n):
        dot_product += a[idx] * b[idx]

    return dot_product


def s2(a, b):
    return np.inner(a, b)

# Tests
assert s1([1,1,1,1],[1,1,1,1]) == 4
assert s2([1,1,1,1],[1,1,1,1]) == 4
assert np.isscalar(s2([1,1,1,1],[1,1,1,1]))
assert np.isscalar(s1([1,1,1,1],[1,1,1,1]))
assert s2([7,7,7],[7,7,7]) == 147
assert s1([7,7,7],[7,7,7]) == 147

print("All tests were successful.")