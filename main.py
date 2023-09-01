import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nn.models import Sequential
from nn.layers import InputLayer, Dense
from nn.utils import confusion_matrix, one_hot_encode, classification_summary


# Smoothing function (moving average)
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


# Define the window size for smoothing
window_size = 20  # You can adjust this value as needed

# Create a sample model
model = Sequential()

model.add(InputLayer(input_shape=(1, 14), name="Input_Layer"))
model.add(Dense(100, activation="relu", name="Hidden_Layer_1"))
model.add(Dense(40, activation="relu", name="Hidden_Layer_2"))
model.add(Dense(4, activation="softmax", name="Output_Layer"))

# Read the data
train_x = pd.read_csv("demo_data/x_train.csv")
train_y = pd.read_csv("demo_data/y_train.csv")
test_x = pd.read_csv("demo_data/x_test.csv")
test_y = pd.read_csv("demo_data/y_test.csv")

# model.load_model("Task_1/a/w.csv", "Task_1/a/b.csv")

# Training the model using the train method

# model.train(
#     train_x.values,
#     one_hot_encode(train_y.values),
#     num_epochs=2000,
#     print_every=20,
#     learning_rate=1,
#     print_details=True,
# )

for lr in [1, 0.1, 0.001]:
# for lr in [1]:
    train_loss, test_loss, training_accuracy, testing_accuracy, lr_ = model.validate(
        train_x,
        one_hot_encode(train_y.values),
        test_x,
        one_hot_encode(test_y.values),
        print_every=20,
        num_epochs=5000,
        learning_rate=1,
        print_details=True,
        redu_lr=True,
        redu_lr_factor=0.6,
    )

    # Apply moving average to your data
    train_loss = moving_average(train_loss, window_size)
    test_loss = moving_average(test_loss, window_size)

    # Apply moving average to your data
    training_accuracy = moving_average(training_accuracy, window_size)
    testing_accuracy = moving_average(testing_accuracy, window_size)

    # Plot the loss
    plt.plot(train_loss, label="Train Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.title(f"Learning Rate: {lr}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot the accuracy
    plt.plot(training_accuracy, label="Train Accuracy")
    plt.plot(testing_accuracy, label="Test Accuracy")
    plt.title(f"Learning Rate: {lr}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # plot the learning rate over time
    plt.plot(lr_, label="Learning Rate")
    plt.title(f"Learning Rate: {lr}")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.show()

# model.train(input_data, labels, num_epochs=1, print_every=100, learning_rate=0.3)

# Use the predict method to get predictions
predictions = model.predict(test_x.values, use_saved_weights=True)


confusion_mat = confusion_matrix(predictions, one_hot_encode(test_y.values))
print("\n", classification_summary(confusion_mat))

# # # Plot the confusion matrix
# fig, ax = plt.subplots()
# im = ax.imshow(confusion_mat, cmap="Blues")
# im.set_clim(0, 1)
# ax.grid(False)
# ax.xaxis.set(ticks=(0, 1, 2, 3), ticklabels=("0", "1", "2", "3"))
# ax.yaxis.set(ticks=(0, 1, 2, 3), ticklabels=("0", "1", "2", "3"))
# ax.set_ylim(3.5, -0.5)
# for i in range(4):
#     for j in range(4):
#         ax.text(j, i, confusion_mat[i, j], ha="center", va="center", color="white")
# plt.show()
