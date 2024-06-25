# imports
import numpy as np


def activation_function(a):
    """have to decide if we want sigmoid, ReLU or softmax"""
    return 1 / (1 + np.exp(-a))    # this one is sigmoid (page 37)


def activation_derivative(a):
    """see above :D"""
    return activation_function(a) * (1-activation_function(a))


class NeuralNetwork(object):
    def __init__(self, sizes):
        """
        Class for a multilayer perceptron
        """
        self._sizes = sizes
        self._layers = len(sizes)
        # create random biases for each neuron in each layer (besides input layer)
        self._biases = [np.random.randn(neurons, 1) for neurons in sizes[1:]]
        # create random weight matrices in a similar way
        self._weights = [np.random.randn(n_after, n_before) for n_after, n_before in zip(sizes[1:], sizes[:-1])]

    def predict(self, data):
        """
        Calculates the values of the output layer based on the inputs and the current model
        :param data: the data we want to make predictions for
        :return: 0-1 values of how confident the model is for each class
        """
        predictions = []
        # for each sample, we want to find what the output would be
        for x in data:
            # move through each layer of the NN and find the next neuron's values
            for b, W in zip(self._biases, self._weights):
                x = activation_function(np.dot(W, x) + b.reshape(-1))   #func 15 on page 35
            predictions.append(x)
        predictions = np.array(predictions)
        return predictions

    def train(self, x, y, epochs, learning_rate):
        """
        trains the neural network.
        (made this a separate function because usually there's more stuff you can do here)
        :param x: training data, input
        :param y: training data, desired output
        :param epochs: how many iterations we want to train it for
        :param learning_rate: how big the steps in the gradient descent should be
        :return:
        """
        for i in range(epochs):
            self._gradient_descent(x, y, learning_rate)
            print(f"Completed epoch {i} of {epochs}")

    def _gradient_descent(self, x, y, mu):
        """

        :param x: training data input
        :param y: correct answers to the training samples
        :param mu: learning rate
        :return:
        """
        # We start with no changes to the weights and biases
        changes_b = [np.zeros(b.shape) for b in self._biases]
        changes_w = [np.zeros(w.shape) for w in self._weights]

        for x, y in zip(x, y):
            # for each training sample, we calculate how much we'd want the weights to change
            change_b, change_w = self._back_propagation(x, y)

            # we add these changes to the overall changes
            changes_b = [b1 + b2 for b1, b2 in zip(changes_b, change_b)]
            changes_w = [w1 + w2 for w1, w2 in zip(changes_w, change_w)]

        # we update the weights based on the backpropagation result
        # (page 44)
        self._weights = [w - mu * cw for w, cw in zip(self._weights, changes_w)]
        self._biases = [b - mu * cb for b, cb in zip(self._biases, changes_b)]

    def _loss(self, pred, true):
        """
        defines a loss function
        (we have to choose which one we want, these are some examples)
        :param pred: the predicted values by the model
        :param true: the values we would want instead
        :return: the mean absolute error (in this case)
        """
        mean_absolute_error = np.mean(np.abs(true - pred))
        mean_square_error = np.mean(np.sqrt(true - pred))
        return mean_absolute_error

    def _back_propagation(self, x, y):
        """
        performs backpropagation for a single sample
        :param x: inputs
        :param y: correct outputs
        :return: the changes this sample wants to make to the weights and biases
        """
        change_b = [np.zeros(b.shape) for b in self._biases]
        change_w = [np.zeros(w.shape) for w in self._weights]

        # first we feedforward through the network
        activation = x      # start with the input
        activations = [x]   # stores the values in each neuron, layer by layer
        z_values = []       # stores the z values that we need later when going back
        for b, w in zip(self._biases, self._weights):
            z = np.dot(w, activation) + b.reshape(-1)
            activation = activation_function(z)
            z_values.append(z)
            activations.append(activation)

        # do backpropagation

        return change_b, change_w

    def cost_derivative(self, output_activations, y):
        return output_activations - y
