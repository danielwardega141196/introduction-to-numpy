"""
    Mamy dwie macierze Am×n i Bn×k. Należy napisać funkcję obliczającą ich iloczyn.

    1) Za pomocą iteracji po elementach (potrójna pętla).
    2) Korzystając z faktu, że każdy element macierzy  Cij jest
       iloczynem skalarnym  i-tego rzędu macierzy A oraz
       j-tego rzędu macierzy B (podwójna pętla)
    3) Używając funkcji:  np.dot lub np.tensordot (bez pętli)
"""

import numpy as np

MESSAGE_REGARDING_DIFFERENT_DIMENSIONS = 'The number of columns ' \
                                         'in the first matrix ' \
                                         'should be the same as ' \
                                         'the number of rows in the second matrix.'

def C1(A, B):

    # Wymiary m, n, k

    # m - liczka wierszy macierzy A
    # n - liczba kolumn macierzy A, liczba wierszy macierzy B
    # k - liczba kolumn macierzy B

    m, n_1 = A.shape
    n_2, k = B.shape

    # Jeżeli liczba kolumn macierzy A jest różna od liczby wierszy macierzy B
    # wyrzuć wyjątek
    if n_1 != n_2:
        raise ValueError(MESSAGE_REGARDING_DIFFERENT_DIMENSIONS)

    # Macierz C będzie miał wymiary (m,k)
    C = np.zeros((m, k))

    for i in range(m):

        for j in range(k):

            # Element Cij to iloczyn skalarny i-tego wiersza macierzy A i j-tej kolumny macierzy B

            scalar_product = 0

            # Zarówno każdy wiersz macierzy A jak i każda kolumna macierzy B zawiera n elementów
            for l in range(n_1):
                scalar_product += A[i][l] * B.T[j][l]

            C[i][j] = scalar_product

    return C

def C2(A, B):

    # Wymiary m, n, k

    # m - liczka wierszy macierzy A
    # n - liczba kolumn macierzy A, liczba wierszy macierzy B
    # k - liczba kolumn macierzy B

    m, n_1 = A.shape
    n_2, k = B.shape

    # Jeżeli liczba kolumn macierzy A jest różna od liczby wierszy macierzy B
    # wyrzuć wyjątek
    if n_1 != n_2:
        raise ValueError(MESSAGE_REGARDING_DIFFERENT_DIMENSIONS)

    # Macierz C będzie miał wymiary (m,k)
    C = np.zeros((m, k))

    for i in range(m):

        for j in range(k):
            # Element Cij to iloczyn skalarny i-tego wiersza macierzy A i j-tej kolumny macierzy B
            # aby go obliczyć używamy funkcji wbudowanej inner
            scalar_product = np.inner(A[i], B.T[j])
            C[i][j] = scalar_product

    return C


def C3(A, B):

    # n - liczba kolumn macierzy A, liczba wierszy macierzy B
    n_1 = A.shape[1]
    n_2 = B.shape[0]

    # Jeżeli liczba kolumn macierzy A jest różna od liczby wierszy macierzy B
    # wyrzuć wyjątek
    if n_1 != n_2:
        raise ValueError(MESSAGE_REGARDING_DIFFERENT_DIMENSIONS)

    C = np.dot(A, B)
    return C

# Tests
m = np.arange(9).reshape(3,3)
assert np.prod(C1(np.eye(3),m) == m)
assert np.prod(C1(m,m) == C3(m,m))
assert np.prod(C2(m,m) == C3(m,m))

try:
    C1(np.arange(9).reshape(3,3),np.ones((2,3)))
except (BaseException, Exception) as error:
    if type(error) != ValueError or \
            str(error) != MESSAGE_REGARDING_DIFFERENT_DIMENSIONS:
        raise AssertionError("Second dimension of matrix "
                             "should be the same as dimension of vector!.")

try:
    C2(np.arange(9).reshape(3,3),np.ones((2,3)))
except (BaseException, Exception) as error:
    if type(error) != ValueError or \
            str(error) != MESSAGE_REGARDING_DIFFERENT_DIMENSIONS:
        raise AssertionError("Second dimension of matrix "
                             "should be the same as dimension of vector!.")

try:
    C3(np.arange(9).reshape(3, 3), np.ones((2, 3)))
except (BaseException, Exception) as error:
    if type(error) != ValueError or \
            str(error) != MESSAGE_REGARDING_DIFFERENT_DIMENSIONS:
        raise AssertionError("Second dimension of matrix "
                             "should be the same as dimension of vector!.")
    
print("All tests were successful.")
