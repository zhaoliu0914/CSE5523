import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize_label_to_numpy(train_label_temp):
    size = train_label_temp.shape[0]
    train_label = np.zeros(size)

    # Normalize training labels to [0, 1]
    index = 0
    while index < size:
        label = train_label_temp.iloc[index, 0]
        if label == 4:
            train_label[index] = 0
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
            if train_label[index] == 0:
                numerator = np.add(numerator, train_dataset[index])

        arr_of_0 = train_label[np.where(train_label == 0)]
        denominator = len(arr_of_0)

    #print(f"numerator = {numerator}")
    #print(f"denominator = {denominator}")
    mu = numerator / denominator

    return mu


def calculate_sigma(mu_1, mu_0, train_dataset, train_label):
    row_size = train_dataset.shape[0]
    column_size = train_dataset.shape[1]

    sigma = np.zeros([column_size, column_size])

    for index in range(row_size):
        # For y = 1
        if train_label[index] == 1:
            x_mu_vector = np.subtract(train_dataset[index], mu_1)
        else:
            # For y = 0
            x_mu_vector = np.subtract(train_dataset[index], mu_0)

        x_mu_matrix = np.outer(x_mu_vector, x_mu_vector)
        sigma = np.add(sigma, x_mu_matrix)

    #for index in range(row_size):
    #    if train_label[index] == 0:
    #        x_mu_vector = np.subtract(train_dataset[index], mu_0)
    #        x_mu_matrix = np.outer(x_mu_vector, x_mu_vector)
    #        sigma = np.add(sigma, x_mu_matrix)

    sigma = (1 / row_size) * sigma

    for i in range(column_size):
        diagonal_value = sigma[i][i]
        sigma[i][i] = diagonal_value + 0.000000001

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

    sigma = calculate_sigma(mu_1, mu_0, train_dataset, train_label)
    sigma_inverse = np.linalg.inv(sigma)

    train_shape = train_dataset.shape
    train_row_size = train_shape[0]
    train_column_size = train_shape[1]

    test_shape = test_dataset.shape
    test_row_size = test_shape[0]
    test_column_size = test_shape[1]

    # Calculate Train Accuracy
    number_correct_train = 0
    for index in range(train_row_size):
        probability_1 = 1
        probability_0 = 1

        real_value = train_label[index]
        predict_value = 1

        x_mu_1 = np.subtract(train_dataset[index], mu_1)
        vector_temp = np.matmul(x_mu_1, sigma_inverse)
        exponential_value = np.dot(vector_temp, x_mu_1)
        exponential_value = -0.5 * exponential_value

        probability_1 = lambda_head * math.pow(math.e, exponential_value)

        x_mu_0 = np.subtract(train_dataset[index], mu_0)
        vector_temp = np.matmul(x_mu_0, sigma_inverse)
        exponential_value = np.dot(vector_temp, x_mu_0)
        exponential_value = -0.5 * exponential_value

        probability_0 = (1 - lambda_head) * math.pow(math.e, exponential_value)

        if probability_1 > probability_0:
            predict_value = 1
        else:
            predict_value = 0

        if predict_value == real_value:
            number_correct_train += 1

    # Calculate Test Accuracy
    number_correct_test = 0
    misclassified_indices = []
    misclassified_predict_values = []
    for index in range(test_row_size):
        probability_1 = 1
        probability_0 = 1

        real_value = test_label[index]
        predict_value = 1

        x_mu_1 = np.subtract(test_dataset[index], mu_1)
        vector_temp = np.matmul(x_mu_1, sigma_inverse)
        exponential_value = np.dot(vector_temp, x_mu_1)
        exponential_value = -0.5 * exponential_value

        probability_1 = lambda_head * math.pow(math.e, exponential_value)

        x_mu_0 = np.subtract(test_dataset[index], mu_0)
        vector_temp = np.matmul(x_mu_0, sigma_inverse)
        exponential_value = np.dot(vector_temp, x_mu_0)
        exponential_value = -0.5 * exponential_value

        probability_0 = (1 - lambda_head) * math.pow(math.e, exponential_value)

        if probability_1 > probability_0:
            predict_value = 1
        else:
            predict_value = 0

        if predict_value == real_value:
            number_correct_test += 1
        else:
            misclassified_indices.append(index)
            if predict_value == 0:
                misclassified_predict_values.append(4)
            else:
                misclassified_predict_values.append(9)

    print(f"Number of correct prediction for Train dataset = {number_correct_train}")
    print(f"Total number of Train dataset = {train_row_size}")
    print(f"Train Accuracy = {number_correct_train / train_row_size}")

    print(f"Number of correct prediction for Test dataset = {number_correct_test}")
    print(f"Total number of Test dataset = {test_row_size}")
    print(f"Test Accuracy = {number_correct_test / test_row_size}")

    # Reshape the images to the correct size (26x26 since the training data has 676 features per image)
    image_size = int(np.sqrt(train_column_size))

    # Select up to 5 misclassified images to show
    max_images_to_show = 5
    images_to_show = min(len(misclassified_indices), max_images_to_show)
    print(f"images_to_show = {images_to_show}")

    # Plot the misclassified images
    fig, axes = plt.subplots(1, images_to_show, figsize=(10, 3))
    for i in range(images_to_show):
        ax = axes[i] if images_to_show > 1 else axes
        misclassified_index = misclassified_indices[i]
        ax.imshow(test_dataset[misclassified_index].reshape(image_size, image_size), cmap='gray')
        ax.set_title(f'Pred: {misclassified_predict_values[i]}')
        ax.axis('off')
    plt.show()