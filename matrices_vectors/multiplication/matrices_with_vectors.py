"""
    Mnożenie macierzy przez wektor:
    Mając wektor n-elementowy x oraz macierz A o wymiarach m×n napisać funkcję obliczającą ich iloczyn:
                                    y = Ax

    1) Za pomocą iteracji po elementach (podwójna pętla)
    2) korzystając z faktu, że każdy element wektora yi jest iloczynem skalarnym i-tego rzędu macierzy A oraz wektora x (pojedyncza pętla)
    3) używając funkcji: np.dot lub np.tensordot (bez pętli)
"""

import numpy as np

MESSAGE_REGARDING_DIFFERENT_DIMENSIONS = "The number of columns in the matrix " \
                                         "should be the same as the length of the vector."


def y1(A, x):
    # liczba kolumn(n) w macierzu A
    n = A.shape[1]

    # Jeżeli liczba kolumn w macierzy A nie jest równa długości wektora x
    # wyrzuć wyjątek
    if n != len(x):
        raise ValueError(MESSAGE_REGARDING_DIFFERENT_DIMENSIONS)

    y = []

    # Przechodzimy przez kolejne wiersze macierzy A
    for row in A:

        # Obliczanie każdego kolejnego elementu wektora y,
        # w którym i-ty element to iloczyn skalarny wektora x i i-tego wiersza macierza A
        scalar_product = 0

        for idx in range(n):
            scalar_product += row[idx] * x[idx]

        y.append(scalar_product)

    return y


def y2(A, x):
    # liczba kolumn(n) w macierzu A
    n = A.shape[1]

    # Jeżeli liczba kolumn w macierzy A nie jest równa długości wektora x
    # wyrzuć wyjątek
    if n != len(x):
        raise ValueError(MESSAGE_REGARDING_DIFFERENT_DIMENSIONS)

    y = []

    # Przechodzimy przez kolejne wiersze macierzy A
    for row in A:
        # Obliczanie każdego kolejnego elementu wektora y ( za pomocą funkcji z pakietu numpy (inner)),
        # w którym i-ty element to iloczyn skalarny wektora x i i-tego wiersza macierza A
        scalar_product = np.inner(row, x)
        y.append(scalar_product)

    return y


def y3(A, x):
    # liczba kolumn(n) w macierzu A
    n = A.shape[1]

    # Jeżeli liczba kolumn w macierzy A nie jest równa długości wektora x
    # wyrzuć wyjątek
    if n != len(x):
        raise ValueError(MESSAGE_REGARDING_DIFFERENT_DIMENSIONS)

    return np.dot(A, x)


# Tests
A = np.arange(9).reshape(3, 3)
x = np.arange(3)
np.testing.assert_almost_equal(y1(A, x), np.array([5, 14, 23]))
np.testing.assert_almost_equal(y2(A, x), np.array([5, 14, 23]))
np.testing.assert_almost_equal(y3(A, x), np.array([5, 14, 23]))

try:
    y1(np.arange(9).reshape(3, 3), np.array([1, 1]))
except (BaseException, Exception) as error:
    if type(error) != ValueError or \
            str(error) != MESSAGE_REGARDING_DIFFERENT_DIMENSIONS:
        raise AssertionError("Second dimension of matrix should "
                             "be the same as dimension of vector!.")

try:
    y2(np.arange(9).reshape(3, 3), np.array([1, 1]))
except (BaseException, Exception) as error:
    if type(error) != ValueError or \
            str(error) != MESSAGE_REGARDING_DIFFERENT_DIMENSIONS:
        raise AssertionError("Second dimension of matrix should "
                             "be the same as dimension of vector!.")

try:
    y3(np.arange(9).reshape(3, 3), np.array([1, 1]))
except (BaseException, Exception) as error:
    if type(error) != ValueError or \
            str(error) != MESSAGE_REGARDING_DIFFERENT_DIMENSIONS:
        raise AssertionError("Second dimension of matrix should "
                             "be the same as dimension of vector!.")

print("All tests were successful.")
