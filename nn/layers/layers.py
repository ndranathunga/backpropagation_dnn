# nn/layers/layers.py

import numpy as np

from nn.utils import (
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    softmax,
    softmax_derivative,
)


class Layer:
    def forward(self, x):
        """
        Perform the forward pass of the layer.

        Args:
        x (numpy.ndarray): The input data or activations from the previous layer, with shape (batch_size, input_size).

        Returns:
        tuple: A tuple containing two elements:
            - A numpy.ndarray representing the layer's output after forward propagation.
            - A dictionary containing any intermediate values needed for backpropagation.

        Notes:
        - This method computes the forward pass of a neural network layer. It takes the input data (x) and computes the
          layer's output based on its specific operation (e.g., linear transformation, activation function).
        - The intermediate values necessary for backpropagation should be stored in the dictionary within the tuple.
          These values may include pre-activation values (Z) or any other relevant information.
        - This method is typically used during the forward pass of a neural network to compute the output of a single layer.
        """
        pass

    def backward(self, dA, Z_prev):
        """
        Perform the backward pass of the layer to compute gradients.

        Args:
        dA (numpy.ndarray): The gradient of the loss with respect to the layer's output, with shape (batch_size, output_size).
        Z_prev (numpy.ndarray): The pre-activation values from the previous layer, with shape (batch_size, input_size).

        Returns:
        tuple: A tuple containing two elements:
            - A numpy.ndarray representing the gradient of the loss with respect to the layer's inputs (dZ_prev),
              with the same shape as Z_prev.
            - A dictionary containing any gradients or intermediate values needed for further backpropagation.

        Notes:
        - This method computes the backward pass of a neural network layer to calculate gradients for weight updates
          and to pass the gradients backward to the previous layer.
        - The input dA represents the gradient of the loss with respect to the layer's output.
        - The input Z_prev represents the pre-activation values from the previous layer.
        - The method should return the gradient of the loss with respect to the layer's inputs (dZ_prev) and any gradients
          or intermediate values required for backpropagation.
        - Implementations of this method should follow the chain rule to compute gradients with respect to the inputs and
          update any layer-specific parameters as needed (e.g., weights, biases).
        """
        pass


class InputLayer(Layer):
    def __init__(self, input_shape, name):
        """
        Initialize an InputLayer.

        Args:
        input_shape (tuple): The shape of the input data (excluding batch size).
        name (str): A name or identifier for the input layer.

        Notes:
        - The InputLayer is a special type of neural network layer used to define the input shape of the network.
        - It does not perform any computation during the forward pass, but it simply passes the input data as is.
        - The input_shape argument should specify the shape of a single input sample, excluding the batch dimension.
        - The name argument can be used to provide a name or identifier for the input layer.
        """
        self.input_shape = input_shape
        self.name = name

    def forward(self, x):
        """
        Perform the forward pass of the InputLayer.

        Args:
        x (numpy.ndarray): The input data, with shape (batch_size, input_shape).

        Returns:
        tuple: A tuple containing two identical numpy.ndarray instances:
            - The input data itself.
            - The input data itself.

        Notes:
        - This method is used during the forward pass of the neural network, and it simply returns the input data as is.
        - The returned tuple contains two identical copies of the input data to maintain consistency with other layers,
          even though this layer does not perform any computation.
        """
        return x, x


class Dense(Layer):
    def __init__(self, units, activation, name):
        """
        Initialize a Dense layer.

        Args:
        units (int): The number of units or neurons in the layer.
        activation (str): The activation function to be used in the layer (e.g., 'relu', 'softmax', 'sigmoid').
        name (str): A name or identifier for the Dense layer.

        Notes:
        - The Dense layer is a fully connected layer in a neural network that performs linear transformations followed by
          an activation function on the input data.
        - The units argument specifies the number of neurons in the layer.
        - The activation argument specifies the activation function to be applied to the layer's output.
        - The name argument can be used to provide a name or identifier for the Dense layer.
        - The weights and bias attributes are initialized as None and should be initialized later based on input data shape.
        - The cache attribute is an empty dictionary that can be used to store intermediate values during forward pass.
        """
        self.units = units
        self.activation = activation
        self.name = name
        self.weights = None
        self.bias = None
        self.cache = {}

    def forward(self, x):
        """
        Forward pass of a neural network layer.

        Args:
        x (numpy.ndarray): The input data or activations from the previous layer, with shape (batch_size, input_size).

        Returns:
        tuple: A tuple containing two elements:
            - A numpy.ndarray representing the layer's output after applying the specified activation function.
            - A numpy.ndarray representing the pre-activation values before applying the activation function.

        Raises:
        ValueError: If the layer's weights have not been initialized properly.

        Notes:
        - This method computes the forward pass of a neural network layer by performing the following steps:
            1. If the layer's weights have not been initialized, it initializes them based on the input data's shape.
            2. Computes the pre-activation values (z) using the dot product of input data (x) and the layer's weights, and adds the bias.
            3. Stores the pre-activation values in the 'cache' dictionary for later use during backpropagation.
            4. Applies the specified activation function based on the 'activation' attribute:
               - If 'activation' is 'relu', applies the ReLU activation function.
               - If 'activation' is 'softmax', applies the softmax activation function.
               - If 'activation' is 'sigmoid', applies the sigmoid activation function.

        - This method is typically used during the forward pass of a neural network to compute the output of a single layer.

        """
        if self.weights is None:
            self.initialize_weights(x.shape)
        z = np.dot(x, self.weights) + self.bias
        self.cache["Z"] = z
        if self.activation == "relu":
            return relu(z), z
        elif self.activation == "softmax":
            return softmax(z), z
        elif self.activation == "sigmoid":
            return sigmoid(z), z

    def backward(self, dA, A_prev):
        """
        Perform the backward pass of the Dense layer to compute gradients.

        Args:
        dA (numpy.ndarray): The gradient of the loss with respect to the layer's output, with shape (batch_size, units).
        A_prev (numpy.ndarray): The activations from the previous layer, with shape (batch_size, input_size).

        Returns:
        tuple: A tuple containing three numpy.ndarray instances:
            - The gradient of the loss with respect to the layer's inputs (dA_prev), with the same shape as A_prev.
            - The gradient of the loss with respect to the layer's weights (dW), with shape (input_size, units).
            - The gradient of the loss with respect to the layer's biases (db), with shape (1, units).

        Notes:
        - This method computes the backward pass of a Dense layer to calculate gradients for weight updates and to pass
          gradients backward to the previous layer.
        - The input dA represents the gradient of the loss with respect to the layer's output.
        - The input A_prev represents the activations from the previous layer.
        - The method calculates the gradients with respect to the layer's inputs (dA_prev), weights (dW), and biases (db).
        - The activation function used during the forward pass determines the calculation of dZ (pre-activation gradients):
            - If 'activation' is 'relu', it applies the derivative of the ReLU activation function.
            - If 'activation' is 'softmax', it assumes the derivative of the softmax function is 1 (no effect on gradients).
            - If 'activation' is 'sigmoid', it applies the derivative of the sigmoid activation function.
        - The gradients are normalized by the batch size (m) to compute the average gradient over the mini-batch.

        """
        m = A_prev.shape[0]
        if self.activation == "relu":
            dZ = dA * relu_derivative(self.cache["Z"])
        elif self.activation == "softmax":
            dZ = dA
            # dZ = dA * softmax_derivative(self.cache["Z"])
        elif self.activation == "sigmoid":
            dZ = dA * sigmoid_derivative(self.cache["Z"])

        # dW = np.dot(A_prev.T, dZ)
        dW = np.dot(A_prev.T, dZ) / m
        # print(dW)
        db = np.sum(dZ, axis=0, keepdims=True) / m
        # db = np.sum(dZ, axis=0, keepdims=True)
        dA_prev = np.dot(dZ, self.weights.T)

        return dA_prev, dW, db

    def initialize_weights(self, input_shape):
        """
        Initialize the weights and biases of the Dense layer.

        Args:
        input_shape (tuple): The shape of the input data (excluding batch size).

        Notes:
        - This method initializes the weights and biases of the Dense layer based on the input data shape.
        - The input_shape argument should specify the shape of a single input sample, excluding the batch dimension.
        - The weights are initialized using a uniform distribution within a range determined by the Xavier/Glorot initialization,
          which helps with weight initialization for improved training stability.
        - The biases are initialized to zeros.
        """
        limit = 1 / np.sqrt(input_shape[1])
        self.weights = np.random.uniform(-limit, limit, (input_shape[1], self.units))
        self.bias = np.zeros((1, self.units))
