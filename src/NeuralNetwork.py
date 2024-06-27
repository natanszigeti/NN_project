# imports
import numpy as np


def activation_function(z, function):
    """
    Defines several activation functions, as necessary
    :param z: Vector representing input to layer, consisting of the sum of weighted inputs and biases
    :param function: String representing activation function to use
    :return: Activation of layer
    """
    if function == "relu":
        return [i if i > 0 else 0 for i in z]
    if function == "softmax":
        return np.exp(z) / sum(np.exp(z))
    return None


def activation_derivative(z, function):
    """

    :param z:
    :param function:
    :return:
    """
    if function == "relu":
        return [1 if i > 0 else 0 for i in z]
    if function == "softmax":
        return z * (np.ones(len(z)) - z)
    return None


def loss(pred, true):
    """
    Defines a loss function
    :param pred: the predicted values by the model
    :param true: the values we would want instead
    :return: Cross entropy
    """
    return np.mean(np.negative(true) * np.log(pred) - (np.ones(len(true)) - true) * np.log(np.ones(len(pred)) - pred))


def loss_derivative(pred, true):
    """
    idek
    :param pred:
    :param true:
    :return:
    """
    if len(pred) != len(true):
        raise Exception("Pred and true not same length!")
    return [(1 - true[i])/(1 - pred[i]) - true[i]/pred[i] for i in range(len(pred))]


def regularizer(weights):
    """
    L1 regularizer, this is probably wrong
    :param weights: Weight matrix
    :return: L1 :D
    """
    return np.sum(np.abs(np.hstack([np.matrix.flatten(item) for item in weights])))


def regularizer_derivative(weights):
    """

    :param weights:
    :return:
    """
    return None


class NeuralNetwork(object):
    def __init__(self, sizes, activ_funcs):
        """
        Class for a multilayer perceptron
        """
        self._sizes = sizes
        self._layers = len(sizes)
        # create random biases for each neuron in each layer (besides input layer)
        self._biases = [np.random.randn(neurons, 1) for neurons in sizes[1:]]
        # create random weight matrices in a similar way
        self._weights = [np.random.randn(n_after, n_before) for n_after, n_before in zip(sizes[1:], sizes[:-1])]
        self._activ_funcs = activ_funcs

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
            for b, W, a in zip(self._biases, self._weights, self._activ_funcs):
                x = activation_function(np.dot(W, x) + b.reshape(-1), a)   # func 15 on page 35
            predictions.append(x)
        predictions = np.array(predictions)
        return predictions

    def train(self, x, y, epochs, learning_rate, batch_count):
        """
        trains the neural network.
        (made this a separate function because usually there's more stuff you can do here)
        :param x: training data, input
        :param y: training data, desired output
        :param epochs: how many iterations we want to train it for
        :param learning_rate: how big the steps in the gradient descent should be
        :param batch_count: number of batches to split data into per epoch
        :return:
        """
        data_size = len(x)
        if data_size != len(y):
            raise Exception("X and y are not the same length!")
        for i in range(epochs):
            indices = np.random.permutation(data_size)
            for j in range(batch_count):
                this_batch = indices[int(j*data_size/batch_count):int((j+1)*data_size/batch_count)]
                self._gradient_descent(x[this_batch], y[this_batch], learning_rate)
            print(f"Completed epoch {i+1} of {epochs}")

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

    def _back_propagation(self, x, y):
        """
        performs backpropagation for a single sample
        :param x: inputs
        :param y: correct outputs
        :return: the changes this sample wants to make to the weights and biases
        """
        change_b = []
        change_w = []

        # first we feed forward through the network
        activation = x      # start with the input
        activations = [x]   # stores the values in each neuron, layer by layer
        z_values = [x]       # stores the z values that we need later when going back
        for b, w, a in zip(self._biases, self._weights, self._activ_funcs):
            z = np.dot(w, activation) + b.reshape(-1)
            activation = activation_function(z, a)
            z_values.append(z)
            activations.append(activation)

        print("activ = ", z_values[-1])
        print("pred = ", activations[-1], ", true = ", y)

        # I got the following bit from the pseudocode on page 209 of the bible, I hope it works
        # do backpropagation

        # we don't need to calculate loss but it's good for debugging
        out_loss = loss(activations[-1], y)
        print("Loss: ", out_loss)

        delta_loss = loss_derivative(activations[-1], y)
        for i in range(self._layers - 1, 0, -1):
            # print(z_values[i])
            # print(self._activ_funcs[i-1])
            delta_activation = delta_loss * activation_derivative(z_values[i], self._activ_funcs[i-1])
            # print("delta_activation = ", delta_activation)
            delta_bias = delta_activation  # + gradient of regularization function wrt b
            # print("prev layer activation shape = ", np.shape(z_values[i-1].T))
            delta_weights = np.outer(delta_activation, z_values[i-1].T)  # + gradient of regularization function wrt W
            # print("delta_weights shape = ", np.shape(delta_weights))
            change_b.append(delta_bias)
            change_w.append(delta_weights)
            # print("weights shape ", np.shape(self._weights[i-1].T))
            delta_loss = np.matmul(self._weights[i-1].T, delta_activation)

        change_b.reverse()
        change_w.reverse()

        print(change_b)

        return change_b, change_w
