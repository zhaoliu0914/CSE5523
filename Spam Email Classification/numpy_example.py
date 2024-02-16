import datetime
import numpy as np

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

print(f"arr[0][1] = {arr[0][1]}")

a = np.array([[1, 2, 3]])

#print(a)

#b = np.append(a, [4, 5, 6], axis=1)
b = np.concatenate((a, [[4, 5, 6]]))

b = np.concatenate((b, [[7, 8, 9]]))

print(b)

print(datetime.datetime.now())