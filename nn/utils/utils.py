# nn/utils/utils.py

import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x):
    # x_exp = np.exp(x - np.min(x, axis=1, keepdims=True))
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / x_sum


def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)


# Define the cross-entropy loss and its derivative
def cross_entropy_loss(y_true, y_pred):
    # m = y_true.shape[0]
    # epsilon = 1e-15
    # y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # loss = -np.sum(y_true * np.log(y_pred)) / m

    # loss = ((y_true - y_pred) ** 2).mean() / 2
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    return loss


def cross_entropy_loss_derivative(y_true, y_pred, z):
    # return (y_pred - y_true) * (softmax(z) - y_true)
    return y_pred - y_true
    # return y_true * (1 - y_pred)


def confusion_matrix(predicted_labels, true_labels):
    num_classes = predicted_labels.shape[1]
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(predicted_labels.shape[0]):
        predicted_class = np.argmax(predicted_labels[i])
        true_class = np.argmax(true_labels[i])
        matrix[true_class, predicted_class] += 1

    return matrix


def one_hot_encode(y):
    # Flatten the input array
    y = y.flatten()

    # Get the number of unique classes
    n_classes = np.max(y) + 1

    # Initialize the one-hot encoded array
    one_hot = np.zeros((len(y), n_classes))

    # Set the appropriate indices to 1
    one_hot[np.arange(len(y)), y] = 1

    return one_hot


def classification_summary(confusion_matrix_):
    num_classes = confusion_matrix_.shape[0]

    # Compute true positive, false positive, true negative, and false negative
    tp = np.diag(confusion_matrix_)
    fp = np.sum(confusion_matrix_, axis=0) - tp
    fn = np.sum(confusion_matrix_, axis=1) - tp
    tn = np.sum(confusion_matrix_) - (tp + fp + fn)

    # Compute precision, recall, and F1 score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    # Compute accuracy
    accuracy = np.sum(tp) / np.sum(confusion_matrix_)

    # Create a summary table
    summary = "{:<13}{:<13}{:<13}{:<13}\n".format(
        "Class", "Precision", "Recall", "F1 Score"
    )
    for i in range(num_classes):
        summary += "{:<13}{:<13.2f}{:<13.2f}{:<13.2f}\n".format(
            i, precision[i], recall[i], f1_score[i]
        )

    summary += f"\nAccuracy: {accuracy:.2f}"

    return summary


# Define a custom formatting function
# Define a custom formatting function
def custom_float_format(x):
    if isinstance(x, np.ndarray):
        return np.vectorize(custom_float_format)(x)
    else:
        if abs(x) < 1e-6:
            return "{:0.6e}".format(x)
        else:
            return "{:0.6f}".format(x)
