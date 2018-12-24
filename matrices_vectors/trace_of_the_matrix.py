"""
    Należy obliczyć ślad dla macierzy kwadratowej A.
"""

import numpy as np

MESSAGE_REGARDING_DIFFERENT_DIMENSIONS = "A square matrix should have " \
                                         "the same number of rows and columns."

def Tr(A):

    # Liczba wierszy i kolumn w macierzy kwadratowej powinna być taka sama
    n_1 = A.shape[0]
    n_2 = A.shape[1]

    # Jeżeli liczba wierszy w macierzy kwadratowej różni się
    # od liczby kolumn w tej macierzy wyrzuć wyjątek
    if n_1 != n_2:
        raise ValueError(MESSAGE_REGARDING_DIFFERENT_DIMENSIONS)

    # Suma elementów na przekątnej macierzy
    trace = 0
    for i in range(n_1):
        trace += A[i][i]

    return trace


# Tests
assert Tr(np.diag([1,2,3,4])) == 10
try:
    Tr(np.ones((2, 3)))
except (BaseException, Exception) as error:
    if type(error) != ValueError or \
            str(error) != MESSAGE_REGARDING_DIFFERENT_DIMENSIONS:
        raise AssertionError("The matrix should be square matrix!.")

print("All tests were successful.")
