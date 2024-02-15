import pandas

SPAM = 1
NON_SPAM = 0


if __name__ == '__main__':
    train = pandas.read_csv("spambase.train", header=None)
    test = pandas.read_csv("spambase.test", header=None)

    number_correct = 0
    number_wrong = 0

    # number of rows
    test_rows = test.shape[0]
    # number of columns
    test_columns = test.shape[1]

    # Calculate P(Y=1)
    number_spam = 0
    train_rows = train.shape[0]
    train_columns = train.shape[1]
    for row in range(train_rows):
        train_label = test.iloc[row, test_columns - 1]
        if train_label == 1:
            number_spam = number_spam + 1

    probability_y_1 = number_spam / train_rows

    # Calculate P(Y=1)
    probability_y_0 = 1 - probability_y_1

    train_y_1 = train[train[57].isin([1])]
    train_y_0 = train[train[57].isin([0])]

    for row in range(test_rows):
        predict_value = 0
        real_value = test.iloc[row, test_columns - 1]

        probability_spam = 1
        probability_non_spam = 1

        # For spam (Y=1) which indicates label = 1
        for column in range(test_columns-1):
            given_value = test.iloc[row, column]
            number_match = train_y_1[train_y_1[column].isin([given_value])].shape[0]
            probability_x = number_match / train_rows

            probability_spam = probability_spam * probability_x
        probability_spam = probability_spam * probability_y_1

        # For Non-spam (Y=0) which indicates label = 0
        for column in range(test_columns-1):
            given_value = test.iloc[row, column]
            number_match = train_y_0[train_y_0[column].isin([given_value])].shape[0]
            probability_x = number_match / train_rows

            probability_non_spam = probability_non_spam * probability_x
        probability_non_spam = probability_non_spam * probability_y_0

        if probability_spam > probability_non_spam:
            predict_value = 1
        else:
            predict_value = 0

        if predict_value == real_value:
            #print(f"Correct: real_value = {real_value}, predict_value = {predict_value}")
            number_correct = number_correct + 1
        else:
            #print(f"Wrong: real_value = {real_value}, predict_value = {predict_value}")
            number_wrong = number_wrong + 1

    print(f"Test Error: {number_wrong / test_rows}")