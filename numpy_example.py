import math
import datetime
import numpy as np

a = np.arange(15)
print(a)

a = np.arange(15).reshape(3, 5)
print(a)

zero_arr = np.zeros(10)
print(zero_arr)

zero_matrix = np.zeros([5, 5])
print(f"zero_matrix = {zero_matrix}")

one_arr = np.ones(5)
print(one_arr)

normalize_zero = np.linalg.norm(zero_arr)
print(f"normalize_zero = {normalize_zero}")

normalize_one = np.linalg.norm(one_arr)
print(f"normalize_one = {normalize_one}")

two_arr = np.array([2, 2, 2, 2, 2])

three_arr = np.add(one_arr, two_arr)
print(f"three_arr = {three_arr}")

four_arr = np.array([10, 1, 2, 3, 4])
print(f"four_arr * 2 = {four_arr * 2}")

dot_product = np.dot(four_arr, two_arr)
print(f"dot_product = {dot_product}")

vector_1 = np.array([1, 2])
vector_2 = np.array([3, 2])

matrix_1 = np.outer(vector_1, vector_2)
print(f"matrix_1 = {matrix_1}")

vector_3 = np.array([1, 2])
matrix_4 = np.array([[3, 2],
                     [6, 4]])
vector_4 = np.matmul(vector_3, matrix_4)
print(f"vector_4 = {vector_4}")


filter_arr = np.array([1, 1, 2, 3, 4])
filtered_arr = filter_arr[np.where(filter_arr == 1)]
print(f"filter_arr = {filtered_arr}")


arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

print(f"arr[0][1] = {arr[0][1]}")

print(f"arr * 2 = {arr * 2}")

a = np.array([[1, 2, 3]])

#print(a)

#b = np.append(a, [4, 5, 6], axis=1)
b = np.concatenate((a, [[4, 5, 6]]))

b = np.concatenate((b, [[7, 8, 9]]))

print(b)

print(datetime.datetime.now())

value = math.pow(math.e, 2)
print(f"pow e = {value}")