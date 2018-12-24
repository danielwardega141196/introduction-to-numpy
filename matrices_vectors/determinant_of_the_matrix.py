"""
    Obliczyć wyznacznik macierzy A korzystając z rozwinięcia Laplace'a.
    https://pl.wikipedia.org/wiki/Rozwini%C4%99cie_Laplace%E2%80%99a
"""
import numpy as np

MESSAGE_REGARDING_DIFFERENT_DIMENSIONS = "The matrix should be a square matrix"

def validate_matrix_dimensions(A):

    # Sprawdź czy podana macierz jest macierzą kwadratową,
    # jeżeli nie jest wyrzuć wyjątek
    n_1, n_2 = A.shape

    if n_1 != n_2:
        raise ValueError(MESSAGE_REGARDING_DIFFERENT_DIMENSIONS)

def calculate_the_determinant_of_the_matrix(A):

    # Jeżeli macierz ma wymiary (1,1) oznacza to że zawiera tylko 1 element,
    # zwróć ten element
    m,n = A.shape
    if (m,n) == (1, 1):
        return A[0][0]

    suma = 0
    # Iterujemy po kolejnych kolumnach macierzy A
    for i in range(n):
        # -1 podnosimy do potęgi i, ponieważ w macierzy kwadratowej
        # oznacza ono zarówno indeks wiersza jak i kolumny
        poverty = (-1) ** i

        # Jako kolejne elementy będziemy brać kolejne elementy pierwszego wiersza
        current_element = A[0][i]

        # Nasz bieżący element będzie mnożony przez macierz pozbawioną pierwszego wiersza oraz i-tej kolumny
        temp_A = A[1:, [k for k in range(n) if k != i]]

        # Sumujemy kolejne częsci rozwinięcia Laplace’a
        suma += poverty * current_element * calculate_the_determinant_of_the_matrix(temp_A)

    return suma


def determinant_of_the_matrix(A):

    validate_matrix_dimensions(A=A)
    return calculate_the_determinant_of_the_matrix(A=A)

# Tests
A = np.array([[ 1.,  3.,  0.,  0.],
              [ -3.,  2.,  3.,  2.],
              [ 1.,  2.,  2.,  2.],
              [ 1.,  0.,  0.,  3.]])

np.testing.assert_equal(round(determinant_of_the_matrix(A), 2),
                        round(np.linalg.det(A), 2))

try:
    determinant_of_the_matrix(np.ones((2,3)))

except (BaseException, Exception) as error:
    if type(error) != ValueError or \
            str(error) != MESSAGE_REGARDING_DIFFERENT_DIMENSIONS:
        raise AssertionError(str(error))

print("All tests were successful.")
