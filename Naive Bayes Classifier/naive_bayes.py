import datetime
import numpy as np
import pandas as pd

SPAM = 1
NON_SPAM = 0

"""
For every new test data, we wish to calculate whether it's a spam(Y=1) or it's a non-spam(Y=0),
so the formular is P(Y|X) = [P(X|Y)P(Y)]/P(X)

Since we want to predict whether Y=1 or Y=0,
we only need to compare P(Y=1|X) = [P(X|Y=1)P(Y=1)]/P(X) and P(Y=0|X) = [P(X|Y=0)P(Y=0)]/P(X)

Since [P(X|Y=1)P(Y=1)]/P(X) and [P(X|Y=0)P(Y=0)]/P(X) have same P(X) and it does not affect the final answer,
we could simply eliminate the P(X) from both equations.

For the Naive Bayes, we treat every X in P(X|Y) is conditional independent to each other,
we could module P(X|Y) = Sigma from 1 to d dimension P(X[d]|Y).
For example, P(X|Y) = P(X[1]|Y) * P(X[2]|Y) * P(X[3]|Y) * .......... * P(X[57]|Y)
As the result, P(Y=1|X) = P(X[1]|Y=1) * P(X[2]|Y=1) * P(X[3]|Y=1) * .......... * P(X[57]|Y=1) * P(Y=1)
               P(Y=0|X) = P(X[1]|Y=0) * P(X[2]|Y=0) * P(X[3]|Y=0) * .......... * P(X[57]|Y=0) * P(Y=0)

We just simply split each column data to 2 groups,
one group is all the values of each column less than and equal to median
another group is all the values of each column greater than median

Since each column only has 2 status less equal and greater than median, we could apply Bernoulli Distribution to columns.
For example, P(X[1]|Y=1) will become P(X[1] <= median|Y=1) or P(X[1] > median|Y=1)

If P(Y=1|X) is greater than P(Y=1|X), we could decide this data/row is a spam(Y=1), and vice versa
"""

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    print(f"Start Time: {start_time}")

    train = pd.read_csv("spambase.train", header=None)
    test = pd.read_csv("spambase.test", header=None)

    train = pd.concat([train, test], ignore_index=True, sort=False)

    number_correct = 0
    number_wrong = 0

    # number of rows in Test dataset
    test_rows = test.shape[0]
    # number of columns in Test dataset
    test_columns = test.shape[1]

    # number of rows in Training dataset
    train_rows = train.shape[0]
    # number of columns in Training dataset
    train_columns = train.shape[1]

    # Training dataset with Y = 1
    train_y_1 = train[train[57].isin([1])]
    number_train_y_1 = train_y_1.shape[0]

    # Training dataset with Y = 0
    train_y_0 = train[train[57].isin([0])]
    number_train_y_0 = train_y_0.shape[0]

    # Probability(Y = 1)
    probability_y_1 = number_train_y_1 / train_rows
    # Probability(Y = 0)
    probability_y_0 = number_train_y_0 / train_rows

    # Calculate the median of each column, and store them into medians[]
    medians = []
    for column in range(train_columns - 1):
        medians.append(train[column].median())
        # print(f"column = {column}, median = {train[column].median()}")


    # We just simply split each column data to 2 groups,
    # one group is all the values of each column less than and equal to median
    # another group is all the values of each column greater than median
    # Since each column only has 2 status less equal and greater than median, we could apply Bernoulli Distribution to columns.
    # For example, P(X[1]|Y=1) will become P(X[1] <= median|Y=1) or P(X[1] > median|Y=1)
    # Based on the value of median,
    # calculate the posterior probability of P(X[d]<=median|Y=1) and P(X[d]<=median|Y=0) for each column
    # where X[d]<=median is strict less than and equal to median "<=", and X[d], d indicates each column.
    # For P(X[d]>median|Y=1) = 1 - P(X[d]<=median|Y=1)
    # store all the posterior to condition_probability,
    # condition_probability[0] = theta_0 while condition_probability[1] = theta_1
    condition_probability = np.array([[0, 0]])
    for column in range(train_columns - 1):
        median = medians[column]

        # Calculate the posterior probability of P(X[d]<=median|Y=1) and call it theta_1
        number_match_1 = train_y_1[train_y_1[column] <= median].shape[0]
        theta_1 = number_match_1 / number_train_y_1

        # Calculate the posterior probability of P(X[d]<=median|Y=0) and call it theta_0
        number_match_0 = train_y_0[train_y_0[column] <= median].shape[0]
        theta_0 = number_match_0 / number_train_y_0

        # condition_probability[0] = theta_0 while condition_probability[1] = theta_1
        column_condition_proba = np.array([[theta_0, theta_1]])

        # Concatenate all posterior conditional probability of each column, and there are total 57 columns,
        # so condition_probability has 57 dimensions/rows
        if column == 0:
            condition_probability = column_condition_proba
        else:
            condition_probability = np.concatenate((condition_probability, column_condition_proba))

    # Predict process
    # For every new test data, we wish to calculate whether it's a spam(Y=1) or it's a non-spam(Y=0)
    # So the formular is P(Y|X) = [P(X|Y)P(Y)]/P(X)
    # Since we want to predict whether Y=1 or Y=0,
    # we only need to compare P(Y=1|X) = [P(X|Y=1)P(Y=1)]/P(X) and P(Y=0|X) = [P(X|Y=0)P(Y=0)]/P(X)
    # Since [P(X|Y=1)P(Y=1)]/P(X) and [P(X|Y=0)P(Y=0)]/P(X) have same P(X) and it does not affect the final answer,
    # we could simply eliminate the P(X) from both equations.
    # For the Naive Bayes, we treat every X in P(X|Y) is conditional independent to each other,
    # we could module P(X|Y) = Sigma from 1 to d dimension P(X[d]|Y).
    # For example, P(X|Y) = P(X[1]|Y) * P(X[2]|Y) * P(X[3]|Y) * .......... * P(X[57]|Y)
    # As the result, P(Y=1|X) = P(X[1]|Y=1) * P(X[2]|Y=1) * P(X[3]|Y=1) * .......... * P(X[57]|Y=1) * P(Y=1)
    #                P(Y=0|X) = P(X[1]|Y=0) * P(X[2]|Y=0) * P(X[3]|Y=0) * .......... * P(X[57]|Y=0) * P(Y=0)
    # If P(Y=1|X) is greater than P(Y=1|X), we could decide this data/row is a spam(Y=1), and vice versa
    for row in range(test_rows):
        predict_value = 0
        real_value = test.iloc[row, test_columns - 1]

        # For spam (Y=1) which indicates label = 1
        probability_spam = 1
        # For Non-spam (Y=0) which indicates label = 0
        probability_non_spam = 1

        for column in range(test_columns - 1):

            new_value = test.iloc[row, column]

            median = medians[column]

            """
            number_match_1 = train_y_1[train_y_1[column] <= median].shape[0]
            theta_1 = number_match_1 / number_train_y_1

            number_match_0 = train_y_0[train_y_0[column] <= median].shape[0]
            theta_0 = number_match_0 / number_train_y_0
            """

            # get posterior conditional probability for each column
            # P(X[d]|Y=1), like P(X[1]|Y=1)
            theta_0 = condition_probability[column][0]
            theta_1 = condition_probability[column][1]

            # For P(X[d]>median|Y=1) = 1 - P(X[d]<=median|Y=1)
            # For P(X[d]>median|Y=0) = 1 - P(X[d]<=median|Y=0)
            if new_value > median:
                theta_1 = 1 - theta_1
                theta_0 = 1 - theta_0

            # Multiply all posterior conditional probabilities together
            probability_spam = probability_spam * theta_1
            probability_non_spam = probability_non_spam * theta_0

        # Multiply all posterior conditional probabilities and P(Y=1)
        probability_spam = probability_spam * probability_y_1
        probability_non_spam = probability_non_spam * probability_y_0

        # Determine whether current data is spam(1) or non-spam(0), based on the probability of P(Y=1|X) and P(Y=0|X)
        if probability_spam >= probability_non_spam:
            predict_value = 1
        else:
            predict_value = 0

        if predict_value == real_value:
            number_correct = number_correct + 1
        else:
            number_wrong = number_wrong + 1

    end_time = datetime.datetime.now()
    print(f"End Time: {end_time}")
    print(f"Total spend time: {end_time - start_time}")

    print(f"number_correct = {number_correct}")
    print(f"number_wrong = {number_wrong}")

    print(f"Test Error: {number_wrong / test_rows}")



