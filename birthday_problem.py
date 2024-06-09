import math
import numpy as np
from matplotlib import pyplot as plt

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

if __name__ == '__main__':
    y_values = []
    # x_values = np.linspace(0, 50, num=50, dtype=int)
    # x_values = np.arange(15)
    x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

    for x in x_values:
        y = (math.factorial(365) / math.factorial(365 - x)) / (365 ** x)
        y_values.append(y)

    plt.scatter(x_values, y_values, color="red")
    #plt.plot(x_values, y_values, color="red")
    plt.show()
