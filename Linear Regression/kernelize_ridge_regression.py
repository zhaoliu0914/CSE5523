import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


lambda_regression = 1


def design_matrix_multiplication(design_matrix):
    """
    transpose(design_matrix) * design_matrix
    """
    design_matrix_transpose = np.transpose(design_matrix)
    result = np.matmul(design_matrix_transpose, design_matrix)
    return result


def polynomial_kernel_1(u, v):
    """
    Polynomial Kernel: k(u, v) = (<u, v> + 1)^2
    """
    return (np.dot(u, v) + 1)**2


def polynomial_kernel_2(u, v):
    """
    Polynomial Kernel: k(u, v) = (<u, v> + 1)^3
    """
    return (np.dot(u, v) + 1)**3


def polynomial_kernel_3(u, v):
    """
    Polynomial Kernel: k(u, v) = (<u, v> + 1)^4
    """
    return (np.dot(u, v) + 1)**4


def gaussian_kernel(u, v):
    """
    Gaussian Kernel: k(u, v) = e^((-1/2) * (||u - v||^2)/variance) where variance=1
    """
    return math.exp(-0.5 * np.linalg.norm(u - v))


def get_y_matrix(dataset):
    shape = dataset.shape
    row_size = shape[0]
    column_size = shape[1]

    matrix = np.zeros(row_size)

    for index in range(row_size):
        matrix[index] = dataset[index][column_size - 1]

    return matrix


if __name__ == '__main__':
    train_dataset_temp = pd.read_csv("steel_composition_train.csv") #, header=None
    #train_label_temp = pd.read_csv("digits_training_labels.csv", header=None)

    train_dataset = train_dataset_temp.to_numpy()
    #train_label = normalize_label_to_numpy(train_label_temp)

    # Remove "id" column
    train_dataset = np.delete(train_dataset, 0, axis=1)

    y_matrix = get_y_matrix(train_dataset)

    # Remove "Strength" column
    train_dataset = np.delete(train_dataset, train_dataset.shape[1] - 1, axis=1)

    shape = train_dataset.shape
    row_size = shape[0]
    column_size = shape[1]

    #print(f"train_dataset[0] = {train_dataset[0]}")
    #print(f"sum(train_dataset[0]) = {np.sum(train_dataset[0])}")

    train_dataset = (train_dataset - np.min(train_dataset)) / (np.max(train_dataset) - np.min(train_dataset))

    #for row in range(row_size):
    #    vector_sum = np.sum(train_dataset[row])
    #    for column in range(column_size):
    #        train_dataset[row][column] = train_dataset[row][column] / vector_sum


    #print(train_dataset[0])
    #print(f"row_size = {row_size}")
    #print(f"column_size = {column_size}")

    identity_matrix = np.identity(row_size)

    sum_residuals = 0

    k = np.zeros((row_size, row_size))

    for i in range(row_size):
        for j in range(row_size):
            k[i, j] = polynomial_kernel_1(train_dataset[i], train_dataset[j])
            #k[i, j] = polynomial_kernel_2(train_dataset[i][i], train_dataset[j][j])
            #k[i, j] = polynomial_kernel_3(train_dataset[i][i], train_dataset[j][j])
            #k[i, j] = gaussian_kernel(train_dataset[i][i], train_dataset[j][j])

    w = np.linalg.inv(k + lambda_regression * identity_matrix)
    w = np.matmul(np.transpose(y_matrix), w)

    #print(f"w = {w}")
    #k_x = np.zeros(row_size)
    #for i in range(row_size):
    #    k_x[i] = polynomial_kernel_1(train_dataset[i], train_dataset[0])
    #print(f"k_x = {k_x}")

    # Calculate Root Mean Square Error(RMSE)
    for index in range(row_size):
        #k_x = np.matmul(train_dataset, train_dataset[index])
        k_x = np.zeros(row_size)
        for i in range(row_size):
            k_x[i] = polynomial_kernel_1(train_dataset[i], train_dataset[index])

        y_head = np.dot(w, k_x)

        print(f"y_head = {y_head}")

        residual = y_matrix[index] - y_head
        residual_square = residual**2
        sum_residuals += residual_square

    print(f"sum_residuals = {sum_residuals}")
    print(f"row_size = {row_size}")
    RMSE = math.sqrt(sum_residuals / row_size)

    print(f"RMSE = {RMSE}")



