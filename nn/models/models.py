# custom_nn_package/models/models.py
import csv
import os

import numpy as np

from nn.layers.layers import InputLayer
from nn.utils import cross_entropy_loss, cross_entropy_loss_derivative


# Parent Model class
class Model:
    def add(self, layer):
        """
        Add a layer to the neural network model.

        Args:
        layer (Layer): An instance of a Layer subclass to be added to the model.

        Notes:
        - This method allows you to add layers to the neural network model sequentially.
        - The order in which layers are added determines the order of computations during the forward pass.
        """
        pass

    def forward(self, x):
        """
        Perform the forward pass of the neural network model.

        Args:
        x (numpy.ndarray): The input data with shape (batch_size, input_size).

        Returns:
        numpy.ndarray: The model's output after performing the forward pass.

        Notes:
        - This method computes the forward pass of the entire neural network model by sequentially passing the input data
          through each added layer in the order they were added.
        - The input data (x) is passed through each layer's forward method in sequence, and the output of one layer
          becomes the input of the next layer.
        - The final output of the last layer in the sequence is returned as the model's output.
        """
        pass

    def load_weights(self, weights):
        """
        Load pre-trained weights into the model.

        Args:
        weights (dict): A dictionary containing pre-trained weights for the model's layers.

        Notes:
        - This method allows you to load pre-trained weights into the layers of the model.
        - The weights dictionary should map layer names or identifiers to their corresponding weight matrices and biases.
        - The weights should match the shape and structure of the layers in the model.
        - Loading pre-trained weights is useful for transfer learning and fine-tuning pre-trained models.
        """
        pass


# Define the neural network architecture
class Sequential(Model):
    def __init__(self):
        """
        Initialize a Sequential neural network model.

        Notes:
        - The Sequential model is a linear stack of layers where you can add layers sequentially.
        - The layers are stored in the 'layers' list, and the 'cache' dictionary can be used to store intermediate
          values during the forward pass.
        - The 'saved_weights' and 'saved_biases' attributes can be used to save the weights and biases of the model.
        """
        self.saved_biases = None
        self.saved_weights = None
        self.layers = []
        self.cache = {}

    def add(self, layer):
        """
        Add a layer to the Sequential model.

        Args:
        layer (Layer): An instance of a Layer subclass to be added to the model.

        Notes:
        - This method allows you to add layers to the Sequential model sequentially.
        - The order in which layers are added determines the order of computations during the forward pass.
        """
        self.layers.append(layer)

    def forward(self, x):
        """
        Perform the forward pass of the Sequential model.

        Args:
        x (numpy.ndarray): The input data with shape (batch_size, input_size).

        Returns:
        numpy.ndarray: The model's output after performing the forward pass.

        Notes:
        - This method computes the forward pass of the entire Sequential model by sequentially passing the input data
          through each added layer in the order they were added.
        - The input data (x) is passed through each layer's forward method in sequence, and the output of one layer
          becomes the input of the next layer.
        - The final output of the last layer in the sequence is returned as the model's output.
        """
        self.cache["A0"] = x
        for idx, layer in enumerate(self.layers):
            x, z = layer.forward(x)
            self.cache[f"A{idx+1}"] = x
            self.cache[f"Z{idx+1}"] = z
        return x

    def load_weights(self, weights):
        """
        Load pre-trained weights into the layers of the Sequential model.

        Args:
        weights (dict): A dictionary containing pre-trained weights for the model's layers.

        Notes:
        - This method allows you to load pre-trained weights into the layers of the Sequential model.
        - The weights dictionary should map layer names or identifiers to their corresponding weight matrices and biases.
        - The weights should match the shape and structure of the layers in the model.
        - Loading pre-trained weights is useful for transfer learning and fine-tuning pre-trained models.
        """
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, "weights"):  # Check if the layer has weights
                if (
                    f"W{idx+1}" in weights and f"b{idx+1}" in weights
                ):  # Check if weights for this layer exist
                    layer.weights = weights[f"W{idx+1}"]
                    layer.bias = weights[f"b{idx+1}"]

    def save_weights(self):
        """
        Save the weights and biases of the Sequential model.

        Notes:
        - This method allows you to save the current weights and biases of the model in the 'saved_weights' and
          'saved_biases' attributes.
        - The saved weights and biases can be useful for checkpointing and model persistence.
        """
        self.saved_weights = {}
        self.saved_biases = {}

        for idx, layer in enumerate(self.layers):
            if hasattr(layer, "weights"):
                self.saved_weights[f"W{idx+1}"] = layer.weights
                self.saved_biases[f"b{idx+1}"] = layer.bias

    def save_model(self, filename, best_saved=False):
        """
        Save the model's weights and biases to CSV files.

        Args:
        filename (str): The base name for the CSV files where weights and biases will be saved.
        best_saved (bool, optional): Whether to save the best weights and biases if available (default is False).

        Notes:
        - This method allows you to save the model's weights and biases to CSV files for persistence or later use.
        - The filename argument specifies the base name for the CSV files. Two separate files will be created:
          one for weights (weights_filename.csv) and one for biases (biases_filename.csv).
        - If best_saved is set to True, it assumes that the best weights and biases have been previously saved in the
          'saved_weights' and 'saved_biases' attributes of the model. It extracts and saves these best values.
        - If best_saved is set to False, you can implement logic to gather weights and biases from the current model's layers
          (self.layers) and save them in a similar manner as shown for the 'best_saved' case.
        """
        file_name_extension = filename.strip().split(".")[-1]

        if file_name_extension == "csv":
            if best_saved:
                weights_data = []
                biases_data = []

                for i, (layer_name, weights) in enumerate(self.saved_weights.items()):
                    # print(weights)
                    # weights_row.extend(weights.flatten().tolist())
                    for weight_row in weights:
                        weights_row = [f"weights btw layer{i} to layer{i + 1}"]
                        weights_row.extend(weight_row.flatten().tolist())
                        weights_data.append(weights_row)

                for i, (layer_name, biases) in enumerate(self.saved_biases.items()):
                    biases_row = [f"bias for layer{i+1}"]
                    biases_row.extend(biases.flatten().tolist())
                    biases_data.append(biases_row)

                # Save weights and biases into CSV files
                with open(f"weights_{filename}", "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(weights_data)

                with open(f"biases_{filename}", "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(biases_data)

            else:
                # Similar logic can be applied for non-best_saved case,
                # where you loop through self.layers to gather weights and biases
                pass  # Your implementation here

    def load_model(self, weights_filename, biases_filename):
        """
        Load weights and biases from CSV files and update the model's layer parameters.

        Args:
        weights_filename (str): The name of the CSV file containing weights data.
        biases_filename (str): The name of the CSV file containing biases data.

        Notes:
        - This method allows you to load weights and biases from CSV files and update the model's layer parameters with
          the loaded values.
        - The weights_filename and biases_filename arguments specify the filenames of the CSV files where the data is stored.
        - The method assumes that the CSV files follow a specific format, where each row corresponds to a layer's weights
          or biases.
        - The loaded weights and biases are stored in dictionaries, and then the model's layer parameters are updated based
          on these loaded values.
        - The specific implementation of updating the model's layer parameters depends on how the model and layers are
          structured. In the provided example, it assumes the model has a list of Layer objects (self.layers), and it updates
          each layer's 'weights' and 'biases' attributes based on the loaded data.
        """
        # Initialize empty dictionaries to store the loaded weights and biases
        loaded_weights = {}
        loaded_biases = {}

        # Load weights from CSV file
        with open(weights_filename, "r", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                layer_info = row[0]
                weights = np.array(row[1:], dtype=float)  # Convert string to float
                layer_index = int(layer_info.split("layer")[2].split(" ")[0])

                if f"layer{layer_index}" in loaded_weights.keys():
                    loaded_weights[f"layer{layer_index}"].append(weights)
                else:
                    loaded_weights[f"layer{layer_index}"] = [weights]

        # Load biases from CSV file
        with open(biases_filename, "r", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                layer_info = row[0]
                biases = np.array(row[1:], dtype=float)  # Convert string to float
                layer_index = int(layer_info.split("layer")[1])
                loaded_biases[f"layer{layer_index}"] = biases

        # Update the model's layer parameters with the loaded weights and biases
        for i, layer in enumerate(self.layers):
            if isinstance(layer, InputLayer):
                continue

            if f"layer{i}" in loaded_weights:
                layer.weights = np.array(loaded_weights[f"layer{i}"], dtype=float)
            if f"layer{i}" in loaded_biases:
                layer.bias = np.array([loaded_biases[f"layer{i}"]], dtype=float)

    def train(
        self,
        input_data,
        labels,
        learning_rate=0.1,
        num_epochs=1000,
        print_details=False,
        print_every=100,
        patience=10,
        redu_lr=False,
        redu_lr_factor=0.6,
        save_dw_db=False,
    ):
        """
        Train the neural network model using backpropagation and stochastic gradient descent.

        Args:
        input_data (numpy.ndarray): The training input data with shape (batch_size, input_size).
        labels (numpy.ndarray): The training labels or target values with shape (batch_size, output_size).
        learning_rate (float, optional): The learning rate for gradient descent (default is 0.1).
        num_epochs (int, optional): The number of training epochs (default is 1000).
        print_details (bool, optional): Whether to print training details (default is False).
        print_every (int, optional): Print details every `print_every` epochs (default is 100).
        patience (int, optional): Number of epochs to wait for loss improvement before reducing learning rate (default is 10).
        redu_lr (bool, optional): Whether to reduce the learning rate when loss stagnates (default is False).
        redu_lr_factor (float, optional): Factor by which to reduce learning rate when loss stagnates (default is 0.6).
        save_dw_db (bool, optional): Whether to save weight and bias gradients to CSV files (default is False).

        Notes:
        - This method trains the neural network model using backpropagation and stochastic gradient descent (SGD).
        - The input_data and labels are used for training.
        - The learning_rate controls the step size for weight updates during training.
        - The num_epochs determines the number of training iterations.
        - If print_details is set to True, training progress will be printed every `print_every` epochs.
        - If redu_lr is set to True, the learning rate will be reduced if the loss stagnates for `patience` epochs.
        - If save_dw_db is set to True, weight and bias gradients will be saved to CSV files for each epoch.
        - The method performs forward and backward passes for each epoch, updating the model's weights and biases.
        - The best model's weights are saved, and training results are printed at the end of training.
        """
        prev_loss = None
        patience_ = 0
        least_error = np.inf
        best_epoch = None

        dw_lst = []
        db_lst = []

        for epoch in range(1, num_epochs + 1):
            # Forward pass
            layer_output = self.forward(input_data)

            # Calculate loss
            loss = cross_entropy_loss(labels, layer_output)

            # Compute loss derivative
            dA = cross_entropy_loss_derivative(
                labels, layer_output, self.layers[-1].cache["Z"]
            )

            # Backpropagation
            for idx, layer in reversed(list(enumerate(self.layers))):
                if isinstance(layer, InputLayer):
                    continue

                A_prev = self.cache[f"A{idx}"]  # Use A of the previous layer
                dA, dW, db = layer.backward(dA, A_prev)

                if save_dw_db and epoch == 1:
                    dw_lst.append(dW)
                    db_lst.append(db)

                if hasattr(layer, "weights"):  # Only update if it's a Dense layer
                    layer.weights -= learning_rate * dW
                    layer.bias -= learning_rate * db

            if epoch % print_every == 0:
                if print_details:
                    print(f"Epoch {epoch}, Loss: {loss:.6f}")

            if redu_lr:
                if prev_loss is not None:
                    if loss > prev_loss:
                        if patience_ == patience:
                            multiplier = redu_lr_factor

                            print(
                                f"Changing learning rate from {learning_rate:.6f} to {learning_rate * multiplier:.6f} at "
                                f"epoch {epoch}"
                            )
                            learning_rate = learning_rate * multiplier

                            patience_ = 0
                        else:
                            patience_ += 1

            prev_loss = loss

            if loss < least_error:
                least_error = loss
                self.save_weights()
                best_epoch = epoch

        print(f"Best epoch: {best_epoch}, Least error: {least_error:.6f}")

        if save_dw_db:
            # if /output/ folder doesn't exist, create it
            if not os.path.exists(os.curdir + "/output/"):
                os.makedirs(os.curdir + "/output/")

            # Flatten the list
            flat_list = [
                item
                for sublist in [arr.tolist() for arr in reversed(dw_lst)]
                for item in sublist
            ]

            # Write to CSV
            with open(os.curdir + "/output/my_dw.csv", "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(flat_list)

            # Flatten the list
            flat_list = [
                item
                for sublist in [arr.tolist() for arr in reversed(db_lst)]
                for item in sublist
            ]

            # Write to CSV
            with open(os.curdir + "/output/my_db.csv", "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(flat_list)

    def validate(
        self,
        train_x,
        train_y,
        validation_x,
        validation_y,
        learning_rate=0.1,
        num_epochs=1000,
        print_details=False,
        print_every=100,
        patience=10,
        redu_lr=False,
        redu_lr_factor=0.6,
    ):
        """
        Validate and optionally train the neural network model on a validation dataset.

        Args:
        train_x (numpy.ndarray): The training input data with shape (batch_size, input_size).
        train_y (numpy.ndarray): The training labels or target values with shape (batch_size, output_size).
        validation_x (numpy.ndarray): The validation input data with shape (batch_size, input_size).
        validation_y (numpy.ndarray): The validation labels or target values with shape (batch_size, output_size).
        learning_rate (float, optional): The learning rate for gradient descent (default is 0.1).
        num_epochs (int, optional): The number of training epochs (default is 1000).
        print_details (bool, optional): Whether to print training details (default is False).
        print_every (int, optional): Print details every `print_every` epochs (default is 100).
        patience (int, optional): Number of epochs to wait for loss improvement before reducing learning rate (default is 10).
        redu_lr (bool, optional): Whether to reduce the learning rate when loss stagnates (default is False).
        redu_lr_factor (float, optional): Factor by which to reduce learning rate when loss stagnates (default is 0.6).

        Returns:
        tuple: A tuple containing lists of training and validation metrics:
            - training_loss_lst: List of training loss values for each epoch.
            - validation_loss_lst: List of validation loss values for each epoch.
            - training_accuracy: List of training accuracy values for each epoch.
            - validation_accuracy: List of validation accuracy values for each epoch.
            - lr_for_epochs: List of learning rates used for each epoch.

        Notes:
        - This method validates the neural network model on a separate validation dataset.
        - It can also train the model on the training dataset if `num_epochs` is greater than 0.
        - The input data and labels for both training and validation datasets are provided.
        - Training progress is printed if `print_details` is set to True.
        - If `redu_lr` is set to True, the learning rate may be reduced if the loss stagnates for `patience` epochs.
        - The method returns lists of training and validation metrics such as loss and accuracy for each epoch.
        - The model's weights are saved when the best validation loss is achieved.
        """
        prev_loss = None
        patience_ = 0
        least_error = np.inf
        best_epoch = None

        training_loss_lst = []
        validation_loss_lst = []

        training_accuracy = []
        validation_accuracy = []

        lr_for_epochs = []

        for epoch in range(1, num_epochs + 1):
            lr_for_epochs.append(learning_rate)

            # Forward pass
            layer_output = self.forward(train_x)

            # Calculate loss
            loss = cross_entropy_loss(train_y, layer_output)

            training_loss_lst.append(loss)

            # Calculate accuracy
            predicted_labels = np.argmax(layer_output, axis=1)
            true_labels = np.argmax(train_y, axis=1)
            accuracy = np.sum(predicted_labels == true_labels) / len(true_labels)
            training_accuracy.append(accuracy)

            # Compute loss derivative
            dA = cross_entropy_loss_derivative(
                train_y, layer_output, self.layers[-1].cache["Z"]
            )

            # Backpropagation
            for idx, layer in reversed(list(enumerate(self.layers))):
                if isinstance(layer, InputLayer):
                    continue

                A_prev = self.cache[f"A{idx}"]  # Use A of the previous layer
                dA, dW, db = layer.backward(dA, A_prev)

                if hasattr(layer, "weights"):  # Only update if it's a Dense layer
                    layer.weights -= learning_rate * dW
                    layer.bias -= learning_rate * db

            if epoch % print_every == 0:
                if print_details:
                    print(f"Epoch {epoch}, Loss: {loss:.6f}")

            if redu_lr:
                if prev_loss is not None:
                    if loss > prev_loss:
                        if patience_ == patience:
                            multiplier = redu_lr_factor

                            print(
                                f"Changing learning rate from {learning_rate:.6f} to {learning_rate * multiplier:.6f} at "
                                f"epoch {epoch}"
                            )
                            # print()
                            learning_rate = learning_rate * multiplier

                            patience_ = 0

                            # Load the saved weights
                            for idx, layer in enumerate(self.layers):
                                if hasattr(layer, "weights"):
                                    layer.weights = self.saved_weights[f"W{idx+1}"]
                        else:
                            patience_ += 1

            prev_loss = loss

            if loss < least_error:
                least_error = loss
                self.save_weights()
                best_epoch = epoch

            # Validation
            layer_output = self.forward(validation_x)
            loss = cross_entropy_loss(validation_y, layer_output)
            validation_loss_lst.append(loss)

            # Calculate accuracy
            predicted_labels = np.argmax(layer_output, axis=1)
            true_labels = np.argmax(validation_y, axis=1)
            accuracy = np.sum(predicted_labels == true_labels) / len(true_labels)
            validation_accuracy.append(accuracy)

        print(f"Best epoch: {best_epoch}, Least error: {least_error:.6f}")

        return (
            training_loss_lst,
            validation_loss_lst,
            training_accuracy,
            validation_accuracy,
            lr_for_epochs,
        )

    def predict(self, x, use_saved_weights=False):
        """
        Make predictions using the neural network model.

        Args:
        x (numpy.ndarray): The input data for which predictions are to be made with shape (batch_size, input_size).
        use_saved_weights (bool, optional): Whether to temporarily use the saved model weights for prediction (default is False).

        Returns:
        numpy.ndarray: Predicted output of the model for the input data with shape (batch_size, output_size).

        Notes:
        - This method allows you to make predictions using the neural network model for a given input dataset.
        - The input data (x) is passed through the model's forward pass to obtain predictions.
        - If use_saved_weights is set to True, the method temporarily switches to the saved model weights before
          making predictions and then restores the current weights.
        - The method returns the predicted output of the model.
        """
        current_weights = {}

        if use_saved_weights:
            # Save the current weights and load the saved weights
            for idx, layer in enumerate(self.layers):
                if hasattr(layer, "weights"):
                    current_weights[f"W{idx+1}"] = layer.weights
                    layer.weights = self.saved_weights[f"W{idx+1}"]

        # Perform a forward pass through the network to get predictions
        predictions = self.forward(x)

        if use_saved_weights:
            # Load the current weights
            for idx, layer in enumerate(self.layers):
                if hasattr(layer, "weights"):
                    layer.weights = current_weights[f"W{idx+1}"]

        return predictions
