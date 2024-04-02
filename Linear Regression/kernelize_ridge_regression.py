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


def polynomial_kernel_1():
    """
    Polynomial Kernel: k(u, v) = (<u, v> + 1)^2
    """
    print("")


def polynomial_kernel_2():
    """
    Polynomial Kernel: k(u, v) = (<u, v> + 1)^3
    """
    print("")


def polynomial_kernel_3():
    """
    Polynomial Kernel: k(u, v) = (<u, v> + 1)^4
    """
    print("")


def gaussian_kernel():
    """
    Gaussian Kernel: k(u, v) = e^((-1/2) * (||u - v||^2)/variance) where variance=1
    """
    print("")


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

    #print(train_dataset[0])
    #print(f"row_size = {row_size}")
    #print(f"column_size = {column_size}")

    identity_matrix = np.identity(column_size)

    w = design_matrix_multiplication(train_dataset)
    w = np.add(w, lambda_regression * identity_matrix)
    w = np.linalg.inv(w)
    w = np.matmul(w, np.transpose(train_dataset))
    w = np.matmul(w, y_matrix)

    w_transpose = np.transpose(w)
    sum_residuals = 0

    # Calculate Root Mean Square Error(RMSE)
    for index in range(row_size):
        y_head = np.matmul(w_transpose, train_dataset[index])
        residual = y_matrix[index] - y_head
        residual_square = math.pow(residual, 2)

        sum_residuals += residual_square

    RMSE = math.sqrt(sum_residuals / (row_size - 2))

    print(f"RMSE = {RMSE}")



