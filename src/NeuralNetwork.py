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
    elif function == "softmax":
        return np.clip(np.exp(z - np.max(z)) / sum(np.exp(z - np.max(z))), 1e-15, 1 - 1e-15)
    return None


def activation_derivative(z, function, true=None):
    """

    :param true:
    :param z:
    :param function:
    :return:
    """
    if function == "relu":
        return [1 if i > 0 else 0 for i in z]
    elif function == "softmax":
        return activation_function(z, "softmax") - true
    return None


def loss(pred, true):
    """
    Defines a loss function
    :param pred: the predicted values by the model
    :param true: the values we would want instead
    :return: Cross entropy
    """
    return np.negative(np.sum(true * np.log(pred)))


# DEPRECATED
# DON'T ACTUALLY NEED THIS ANYMORE OOPS
def loss_derivative(pred, true):
    """
    idek
    :param pred:
    :param true:
    :return:
    """
    if len(pred) != len(true):
        raise Exception("Pred and true not same length!")
    return [(1 - true[i]) / (1 - pred[i]) - true[i] / pred[i] for i in range(len(pred))]

# Dont need this actually i think
def regularizer(weights, biases):
    """
    L1 regularizer, this is probably wrong
    :param weights: Weight matrix
    :param biases: Bias matrix
    :return: L1 :D
    """
    return np.sum(np.abs(np.hstack([np.matrix.flatten(item) for item in weights])))  # + \
        # np.sum(np.abs(np.hstack([np.matrix.flatten(item) for item in biases])))


def regularizer_derivative(theta):
    """

    :param theta:
    :return:
    """
    return np.sign(theta)


def he_init(n_before, n_after):
    return np.random.randn(n_before, n_after) * np.sqrt(2 / n_before)


class NeuralNetwork(object):
    def __init__(self, sizes, activ_funcs):
        """
        Class for a multilayer perceptron
        """
        self._sizes = sizes
        self._layers = len(sizes)
        # create random biases for each neuron in each layer (besides input layer)
        self._biases = [np.random.randn(neurons) for neurons in sizes[1:]]
        # create random weight matrices in a similar way
        self._weights = [he_init(n_after, n_before) for n_after, n_before in zip(sizes[1:], sizes[:-1])]
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
                x = activation_function(np.dot(W, x) + b, a)  # func 15 on page 35
            predictions.append(x)
        predictions = np.array(predictions)
        return predictions

    def train(self, x, y, epochs, learning_rate, batch_count, reg_const):
        """
        trains the neural network.
        (made this a separate function because usually there's more stuff you can do here)
        :param x: training data, input
        :param y: training data, desired output
        :param epochs: how many iterations we want to train it for
        :param learning_rate: how big the steps in the gradient descent should be
        :param batch_count: number of batches to split data into per epoch
        :param reg_const: regularization constant
        :return: mean loss per training epoch
        """
        data_size = len(x)
        loss_per_epoch = []
        if data_size != len(y):
            raise Exception("X and y are not the same length!")
        for i in range(epochs):
            indices = np.random.permutation(data_size)
            for j in range(batch_count):
                this_batch = indices[int(j * data_size / batch_count):int((j + 1) * data_size / batch_count)]
                avg_loss = self._gradient_descent(x[this_batch], y[this_batch], learning_rate, reg_const)
                loss_per_epoch.append(avg_loss)
            print(f"Completed epoch {i + 1} of {epochs}, average loss: ", avg_loss)

        print(self._weights)
        return loss_per_epoch

    def _gradient_descent(self, inp, outp, mu, lam):
        """

        :param x: training data input
        :param y: correct answers to the training samples
        :param mu: learning rate
        :return:
        """
        # We start with no changes to the weights and biases
        changes_b = [np.zeros(b.shape) for b in self._biases]
        # print([b.shape for b in self._biases])
        # print([np.shape(i) for i in changes_b])
        changes_w = [np.zeros(w.shape) for w in self._weights]

        avg_loss = []

        for x, y in zip(inp, outp):
            # for each training sample, we calculate how much we'd want the weights to change
            change_b, change_w, out_loss = self._back_propagation(x, y, lam)

            change_b = [np.divide(change, len(inp)) for change in change_b]
            change_w = [np.divide(change, len(inp)) for change in change_w]

            # we add these changes to the overall changes
            changes_b = [b1 + b2 for b1, b2 in zip(changes_b, change_b)]
            changes_w = [w1 + w2 for w1, w2 in zip(changes_w, change_w)]

            avg_loss.append(out_loss)

        # print([np.shape(i) for i in changes_b])

        # we update the weights based on the backpropagation result
        # (page 44)
        self._weights = [w - mu * cw for w, cw in zip(self._weights, changes_w)]
        self._biases = [b - mu * cb for b, cb in zip(self._biases, changes_b)]
        # print([np.shape(i) for i in self._weights])
        # print([np.shape(i) for i in self._biases])
        return np.mean(avg_loss)

    def _back_propagation(self, x, y, lam):
        """
        performs backpropagation for a single sample
        :param x: inputs
        :param y: correct outputs
        :return: the changes this sample wants to make to the weights and biases
        """
        change_b = []
        change_w = []

        # first we feed forward through the network
        activation = x  # start with the input
        activations = [x]  # stores the values in each neuron, layer by layer
        z_values = [x]  # stores the z values that we need later when going back
        for b, w, a in zip(self._biases, self._weights, self._activ_funcs):
            z = np.dot(w, activation) + b
            activation = activation_function(z, a)
            z_values.append(z)
            activations.append(activation)

        # print("activ = ", z_values[-1])
        # print("pred = ", activations[-1], ", true = ", y)

        # I got the following bit from the pseudocode on page 209 of the bible, I hope it works
        # do backpropagation

        # we don't need to calculate loss but it's good for debugging
        out_loss = loss(activations[-1], y)  # + lam * regularizer(self._weights, self._biases)
        # print("Loss: ", out_loss)

        delta_loss = np.ones(len(y))
        for i in range(self._layers - 1, 0, -1):
            # print(z_values[i])
            # print(self._activ_funcs[i-1])
            delta_activation = delta_loss * activation_derivative(z_values[i], self._activ_funcs[i - 1], y)
            # print("delta_activation = ", delta_activation)
            delta_bias = delta_activation  # + lam * regularizer_derivative(self._biases[i - 1])
            # print("prev layer activation shape = ", np.shape(z_values[i-1].T))
            delta_weights = np.outer(delta_activation, z_values[i - 1].T) + \
                            lam * regularizer_derivative(self._weights[i - 1])
            # print("delta_weights shape = ", np.shape(delta_weights))
            change_b.append(delta_bias)
            change_w.append(delta_weights)
            # print("weights shape ", np.shape(self._weights[i-1].T))
            delta_loss = np.matmul(self._weights[i - 1].T, delta_activation)

        change_b.reverse()
        change_w.reverse()

        # print([np.shape(i) for i in change_b])
        # print([np.shape(i) for i in change_w])

        return change_b, change_w, out_loss
