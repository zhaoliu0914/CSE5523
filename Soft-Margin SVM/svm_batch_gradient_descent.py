import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

eta = 0.001
slack_cost_c = 3
number_iterations = [2, 4, 6, 8, 10, 12, 14, 16, 18 ,20, 22, 24, 26, 28, 30]
training_accuracies = []


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

    #print(train_label)
    return train_label


def derivative_E_on_w(w, b, train_dataset, train_label):
    row_size = train_dataset.shape[0]
    column_size = train_dataset.shape[1]

    summation = np.zeros(column_size)

    for i in range(row_size):
        x_i = train_dataset[i]
        y_i = train_label[i]

        # Checking max(0, 1 - yi(W * xi + b))
        # if 0 >= 1 - yi(W * xi + b), then Derivative(0) on w = 0
        # if 0 < 1 - yi(W * xi + b), then Derivative(1 - yi(Wxi + b)) on w = -yi*xi
        temp = np.add(np.dot(w, x_i), b)
        temp = 1 - (y_i * temp)

        if 0 < temp:
            summation = np.add(summation, -1 * np.dot(x_i, y_i))

    summation = slack_cost_c * summation
    w_gradient = np.add(w, summation)

    return w_gradient


def derivative_E_on_b(w, b, train_dataset, train_label):
    row_size = train_dataset.shape[0]

    b_gradient = 0

    for i in range(row_size):
        x_i = train_dataset[i]
        y_i = train_label[i]

        # Checking max(0, 1 - yi(W * xi + b))
        # if 0 >= 1 - yi(W * xi + b), then Derivative(0) on b = 0
        # if 0 < 1 - yi(W * xi + b), then Derivative(1 - yi(Wxi + b)) on b = -yi
        temp = np.add(np.dot(w, x_i), b)
        temp = 1 - (y_i * temp)

        if 0 < temp:
            b_gradient = b_gradient + (-1 * y_i)

    b_gradient = slack_cost_c * b_gradient

    return b_gradient


if __name__ == '__main__':
    train_dataset_temp = pd.read_csv("digits_training_data.csv", header=None)
    train_label_temp = pd.read_csv("digits_training_labels.csv", header=None)

    train_dataset = train_dataset_temp.to_numpy()
    train_label = normalize_label_to_numpy(train_label_temp)

    shape = train_dataset.shape
    row_size = shape[0]
    column_size = shape[1]

    for number_iteration in number_iterations:
        w = np.zeros(column_size)
        b = 0

        for i in range(number_iteration):
            alpha = eta / (1 + i * eta)

            w_gradient = derivative_E_on_w(w, b, train_dataset, train_label)
            w = np.add(w, -1 * alpha * w_gradient)

            b_gradient = derivative_E_on_b(w, b, train_dataset, train_label)
            b = b + (-1 * alpha * b_gradient)

        # Verify Training Accuracy
        number_correct = 0
        for i in range(row_size):
            predict_value = -1
            real_value = train_label[i]

            w_x_dot_product = np.dot(w, train_dataset[i])
            result_value = w_x_dot_product + b

            if result_value < 0:
                predict_value = -1
            else:
                predict_value = 1

            if predict_value == real_value:
                number_correct += 1

        print(f"Iteration = {number_iteration}, Training Accuracy = {number_correct / row_size}")
        training_accuracies.append(number_correct / row_size)

    plt.plot(number_iterations, training_accuracies)
    plt.show()