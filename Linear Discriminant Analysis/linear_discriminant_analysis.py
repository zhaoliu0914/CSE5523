import math
import numpy as np
import pandas as pd


def normalize_label_to_numpy(train_label_temp):
    size = train_label_temp.shape[0]
    train_label = np.zeros(size)

    # Normalize training labels to [-1, 1]
    index = 0
    while index < size:
        label = train_label_temp.iloc[index, 0]
        if label == 4:
            train_label[index] = -1
        else:
            train_label[index] = 1

        index += 1

    return train_label


def calculate_lambda(train_label):
    row_size = train_label.shape[0]
    arr_of_1 = train_label[np.where(train_label == 1)]
    lambda_head = len(arr_of_1) / row_size

    return lambda_head


def calculate_mu(category, train_dataset, train_label):
    row_size = train_dataset.shape[0]
    column_size = train_dataset.shape[1]

    numerator = np.zeros(column_size)
    denominator = 1

    # if category == 1, then mu1
    # if category == 0, then mu0
    if category == 1:
        for index in range(row_size):
            if train_label[index] == 1:
                numerator = np.add(numerator, train_dataset[index])

        arr_of_1 = train_label[np.where(train_label == 1)]
        denominator = len(arr_of_1)

    else:
        for index in range(row_size):
            if train_label[index] == -1:
                numerator = np.add(numerator, train_dataset[index])

        arr_of_0 = train_label[np.where(train_label == -1)]
        denominator = len(arr_of_0)

    #print(f"numerator = {numerator}")
    #print(f"denominator = {denominator}")
    mu = numerator / denominator

    return mu


def calculate_sigma(mu_1, mu_0, train_dataset):
    row_size = train_dataset.shape[0]
    column_size = train_dataset.shape[1]

    sigma = np.zeros([column_size, column_size])

    for index in range(row_size):
        x_mu_vector = np.subtract(train_dataset[index], mu_1)
        x_mu_matrix = np.outer(x_mu_vector, x_mu_vector)

        sigma = np.add(sigma, x_mu_matrix)

    for index in range(row_size):
        x_mu_vector = np.subtract(train_dataset[index], mu_0)
        x_mu_matrix = np.outer(x_mu_vector, x_mu_vector)

        sigma = np.add(sigma, x_mu_matrix)

    return sigma


if __name__ == '__main__':
    train_dataset_temp = pd.read_csv("digits_training_data.csv", header=None)
    train_label_temp = pd.read_csv("digits_training_labels.csv", header=None)

    test_dataset_temp = pd.read_csv("digits_test_data.csv", header=None)
    test_label_temp = pd.read_csv("digits_test_labels.csv", header=None)

    train_dataset = train_dataset_temp.to_numpy()
    train_label = normalize_label_to_numpy(train_label_temp)
    test_dataset = test_dataset_temp.to_numpy()
    test_label = normalize_label_to_numpy(test_label_temp)

    #shape = train_dataset.shape
    #train_row_size = shape[0]
    #train_column_size = shape[1]

    lambda_head = calculate_lambda(train_label)

    mu_1 = calculate_mu(1, train_dataset, train_label)
    mu_0 = calculate_mu(0, train_dataset, train_label)

    sigma = calculate_sigma(mu_1, mu_0, train_dataset)
    #sigma_inverse = np.linalg.inv(sigma)

    test_shape = test_dataset.shape
    test_row_size = test_shape[0]
    test_column_size = test_shape[1]

    number_correct = 0
    for index in range(test_row_size):
        probability_1 = 1
        probability_0 = 1

        real_value = test_label[index]
        predict_value = 1

        x_mu_1 = np.subtract(test_dataset[index], mu_1)
        vector_temp = np.matmul(x_mu_1, sigma)
        exponential_value = np.dot(vector_temp, x_mu_1)
        exponential_value = -0.5 * exponential_value

        probability_1 = lambda_head * math.pow(math.e, exponential_value)

        x_mu_0 = np.subtract(test_dataset[index], mu_0)
        vector_temp = np.matmul(x_mu_0, sigma)
        exponential_value = np.dot(vector_temp, x_mu_0)
        exponential_value = -0.5 * exponential_value

        probability_0 = (1 - lambda_head) * math.pow(math.e, exponential_value)

        if probability_1 >= probability_0:
            predict_value = 1
        else:
            predict_value = -1

        if predict_value == real_value:
            number_correct += 1

    print(f"test accuracy = {number_correct / test_row_size}")