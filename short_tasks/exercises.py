import numpy as np
import inspect

# 1) Stwórz wektor zer o rozmiarze 10

v = np.zeros((10,), dtype=int)

assert v.sum() == 0
assert v.shape == (10,)

# ---------------------------------------------------
# 2) Napisz funkcję zwracającą ilość danych jaką zajmuje tablica numpy

def memof(a):
    return a.nbytes

assert memof(np.zeros((12,10),dtype=np.int32))==480

# ---------------------------------------------------
# 3) Napisz funkcję tworzącą wektor zer o rozmiarze n z jedynką na piątym miejscu

def make_v(n):

    # Tworzenie wektora z 10 zerami
    v = np.zeros((n,), dtype=int)

    # Ustawienie jedynki na piątym miejscu
    v[4] = 1
    return v

assert make_v(14)[4] == 1
np.testing.assert_array_equal( make_v(14)[:4] , 0)
np.testing.assert_array_equal( make_v(14)[5:] , 0)

# ---------------------------------------------------
# 4) Utwórz wektor z wartościami od 10 do 49 włącznie

def make10_49():
    v = np.array(range(10, 50))
    return v

assert make10_49()[7] == 17

# ---------------------------------------------------
# 5) Odwóć kolejność elementów wektora

def reverse(v):
    v = v[::-1]
    return v

np.testing.assert_equal(reverse(np.array([1,2,3])), np.array([3,2,1]))

# ---------------------------------------------------
# 6) Stwórz macierz jednostkową o rozmiarze n x n

def identity_matrix(n):
    v = np.eye(n)
    return v

np.testing.assert_equal(identity_matrix(2), np.array([[1, 0],  [0, 1]]))

# ---------------------------------------------------
# 7) Utwórz macierz n×n z wartościami od 1 do n**2

# Tak by każdy rząd zawierał wartości rosnące o 1
def n2_col(n):
    # Inicjalizacja talicy z wartościami od 1 do n**2,
    # a następnie zmienienie jej kształtu na macierz n x n
    v = np.arange(1, n ** 2 + 1).reshape(n, n)
    return v

np.testing.assert_equal( n2_col(2), np.array([[1, 2],  [3, 4]]) )

# Tak by każda kolumna zawierała wartości rosnące o 1
def n2_row(n):
    # Zamiana kolumn z wierszami tak aby wartości nie roszły w
    # wierszach tylko w kolumnach
    v = n2_col(n).transpose()
    return v

np.testing.assert_equal( n2_row(2), np.array([[1, 3],  [2, 4]]) )

# ---------------------------------------------------
# 8) Utwórz macierz n×m z losowymi wartościami

# Z przedziału [0,1)
def rand1(n, m):
    v = np.random.uniform(0,1,size=(n,m))
    return v

assert np.min(rand1(n=10, m=11)) >= 0
assert np.max(rand1(n=10, m=10)) < 1
assert np.shape(rand1(n=10, m=15)) == (10, 15)

# Z przedziału [a,b)
def rand2(n, m, a, b):
    v = np.random.uniform(a, b, size=(n, m))
    return v

assert np.min(rand2(n=10, m=12, a=3, b=7)) >= 3
assert np.max(rand2(n=10, m=12, a=3, b=7)) < 7
assert np.shape(rand2(n=10, m=12, a=3, b=7)) == (10, 12)

# ---------------------------------------------------
# 9) Znajdź wskaźniki dla których wartości wektora są równe zero

def is_zero(x):
    return np.where(x == 0)[0]

x = np.array([1,2,0,1,0,11])
np.testing.assert_equal(is_zero(x), np.array([2,4]))


# ---------------------------------------------------
# 10) Oblicz dla zadanego wektora jego wartość najmniejszą, największą oraz średnią.

def mystats(x):
    return np.min(x), np.max(x), np.average(x)

x = np.array([1,2,0,1,0,11])
assert  mystats(x) == (0, 11, 2.5)

# ---------------------------------------------------
# 11) Stwórz dwuwymiarową tablicę z zerami w środku i jedynkami na zewnątrz.
def zeros_padded(n):

    # Stworzenie macierzy n x n wypełnionej zerami
    matrix = np.zeros((n, n))

    # Zastępienie krańcowych wartości jedynkami
    matrix = np.pad(matrix[1:-1, 1:-1], 1, 'constant', constant_values=1)

    return matrix

padded_array = np.array([[ 1.,  1.,  1.,  1.],
                         [ 1.,  0.,  0.,  1.],
                         [ 1.,  0.,  0.,  1.],
                         [ 1.,  1.,  1.,  1.]])

np.testing.assert_equal(zeros_padded(n=4),
                        padded_array)

# ---------------------------------------------------
# 12) Używając `np.pad` dodaj do tablicy otoczenie z wartością 3

def pad3(x):

    # Do całej macierzy dodajemy obramowanie o szerokości 1 elementu
    # Obramowanie będzie wypełnione 3
    matrix = np.pad(x, 1, 'constant', constant_values=3)
    return matrix

x = np.ones((2,3))
x_pad3 = np.array([[ 3.,  3.,  3.,  3.,  3.],
                   [ 3.,  1.,  1.,  1.,  3.],
                   [ 3.,  1.,  1.,  1.,  3.],
                   [ 3.,  3.,  3.,  3.,  3.]])
np.testing.assert_equal(pad3(x), x_pad3)

# ---------------------------------------------------
# 13) Dla danej tablicy zastąp maksymalne wartości zerami.

def maxto0(x):
    # Wyznaczenie maksimum w danym macierzu
    maximium = np.max(x)

    # Zamienienie maksymalnych wartości na zera
    x[np.where(x == maximium)] = 0

    return x

x_expected = np.array([[1, 0, 1, 2],
                       [0, 2, 1, 0],
                       [2, 0, 2, 1]])

x = np.array([[1, 3, 1, 2],
              [3, 2, 1, 3],
              [2, 0, 2, 1]])

np.testing.assert_equal(maxto0(x),x_expected)

# ---------------------------------------------------
# 14) Niech będzie dana tablica k parametrów zmierzonych w n pomiarach:
#       xij:  j-ty parametr w i-tym pomiarze

# Stwórz funkcję nie zawierającą pętli, która obliczy:
# - średnią po pomiarach dla wszyskich zmiennych
# - odchylenie od wartości średniej dla każdej zmiennej we wszystkich pomiarach
# - odchylenie średniokwadratowe dla każdej zmiennej

def data_stats(x):

    # Liczba wierszy w danym macierzu
    n = x.shape[0]

    # Suma wszystkich pomiarów (suma danej kolumny) podzielona przez n pomiarów (liczba wierszy)
    x_avg = 1 / n * np.sum(x, axis=0)

    # Od każdego wiersza w macierzy odejmujemy średnie wartości poszczególnych kolumn
    x_delta = x - x_avg

    # Odchylenie średniokwadratowe dla każdej zmiennej (dla każdej kolumny)
    x_sigma = 1 / n * np.sum((x - x_avg) ** 2, axis=0)

    return x_avg, x_delta, x_sigma


example_matrix = np.array([[3, 4, 2, 3],
                           [3, 4, 3, 4],
                           [4, 3, 2, 2],
                           [3, 2, 1, 2],
                           [3, 2, 3, 1]])

average, delta, sigma = data_stats(example_matrix)
expected_average = np.array([ 3.2, 3. ,  2.2,  2.4])
expected_delta = np.array([[-0.2,  1. , -0.2,  0.6],
                           [-0.2,  1. ,  0.8,  1.6],
                           [ 0.8,  0. , -0.2, -0.4],
                           [-0.2, -1. , -1.2, -0.4],
                           [-0.2, -1. ,  0.8, -1.4]])
expected_sigma = np.array([0.16, 0.8 , 0.56, 1.04])

np.testing.assert_allclose(average,
                           expected_average)

np.testing.assert_allclose(delta,
                           expected_delta)

np.testing.assert_allclose(sigma,
                           expected_sigma)

blacklist = [".mean",".average","for","while","std"]
assert all([ not keyword  in inspect.getsource(data_stats) for keyword in blacklist])

print("All tests were successful.")