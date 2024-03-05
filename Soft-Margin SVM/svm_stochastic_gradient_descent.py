import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

eta = 0.001
slack_cost_c = 3
number_iterations = [2, 4, 6, 8, 10]
training_accuracys = []


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


def derivative_E_on_w(w, train_dataset, train_label, index):
    row_size = train_dataset.shape[0]

    y_i = train_label[index]
    y_i_negative = -1 * y_i

    x_i = train_dataset[index]

    summation = y_i_negative * x_i

    # TODO: Max(0, -yx)

    summation = slack_cost_c * summation

    w = w * (1/row_size)
    w_gradient = np.add(w, summation)

    return w_gradient


def derivative_E_on_b(train_dataset, train_label, index):
    #row_size = train_dataset.shape[0]

    b_gradient = 0

    y_i = train_label[index]
    y_i_negative = -1 * y_i

    if y_i_negative > 0:
        b_gradient = slack_cost_c * y_i_negative

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

            permutation = random.randrange(0, row_size)

            for i in range(permutation):
                w_gradient = derivative_E_on_w(w, train_dataset, train_label, i)
                w = np.add(w, -1 * alpha * w_gradient)

                b_gradient = derivative_E_on_b(train_dataset, train_label, i)
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

        training_accuracys.append(number_correct / row_size)
        print(f"Training Accuracy = {number_correct / row_size}")

    plt.plot(number_iterations, training_accuracys)
    plt.show()