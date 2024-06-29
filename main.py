import os
import time
import numpy as np
from matplotlib import pyplot as plt
import math

from src.pca import PCA
from src.NeuralNetwork import *


def load_and_split_digits(data: np.ndarray, train_split_ratio: float, random: bool = True):
    """
    function for splitting the digits task into a train and test set
    :param data: the data from mfeat
    :param train_split_ratio: how we want to split the data
    :param random: whether we want the data to be returned in a random order
    :return: the train and test data like so (X_train, y_train), (X_test, y_test)
    """
    sample_size = data.shape[0]
    samples_per_digit = int(sample_size/10)

    train_per_digit = int(samples_per_digit*train_split_ratio)
    test_per_digit = samples_per_digit - train_per_digit

    # create train and test labels
    y_train = np.repeat(np.arange(10), train_per_digit)
    y_test = np.repeat(np.arange(10), test_per_digit)

    X_train = []
    X_test = []
    for i in range(0, sample_size, samples_per_digit):
        chunk = data[i:i + samples_per_digit]  # get 200 samples (all the same label)
        # extend the data with the right amount of samples
        if random:
            train_indices = np.random.permutation(samples_per_digit)[:train_per_digit]
            test_indices = np.random.permutation(samples_per_digit)[train_per_digit:]
        else:
            train_indices = np.arange(train_per_digit)
            test_indices = np.arange(train_per_digit, samples_per_digit)
        X_train.extend(chunk[train_indices])
        X_test.extend(chunk[test_indices])
    # convert the lists to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    return (X_train, y_train), (X_test, y_test)


def show_digits(images, labels, figsize=(8, 8), x=16, y=15):
    """
    Plots some of the digits and their labels
    :param images: a list of images or a single image
    :param labels: a list of labels or a single label
    :param figsize: how big the figure should be (influences the size of the labels)
    :param x: how tall the image is     (for reshaping the data)
    :param y: how broad the image is    (for reshaping the data)
    :return:
    """
    # reshape the data so we can plot it
    if len(images.shape) == 2:
        images = images.reshape(images.shape[0], x, y)
    else:
        images = images.reshape(x, y)

    # check if we have a matching number of images and labels
    num_images = images.shape[0] if len(images.shape) == 3 else 1
    num_labels = labels.shape[0] if len(labels.shape) == 2 else 1
    if num_images < num_labels:
        raise Exception("You have too many labels to plot")
    elif num_images > num_labels:
        raise Exception("You don't have enough labels to plot")

    if num_images == 1:
        # plot just a single digit
        plt.figure()
        plt.title(labels.argmax())
        plt.imshow(images, cmap='gray')
        plt.axis('off')
        plt.show()
        return

    # divide the images into rows and columns
    cols = round(math.sqrt(num_images))
    rows = cols
    if cols*rows < num_images:
        rows = rows+1

    # set up the figure
    fig, axes = plt.subplots(ncols=cols, nrows=rows, sharex=True, sharey=True, figsize=figsize)
    fig.tight_layout()

    # format the axes and add the digits
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.set_title(labels[i].argmax())
            ax.imshow(images[i], cmap='gray')
        ax.axis('off')
    # show the plot
    plt.show()


def preprocess_images(data):
    """
    Normalizes (and flattens if we want to use mnist) the data
    :param data:
    :return:
    """
    # data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    data = data/6  # 255 for mnist
    return data


def preprocess_labels(labels):
    """
    Changes the single int labels into a vector of 10 where the index of the 1 is now the label
    :param labels: list of int which are the labels
    :return: list of length 10 vectors which are the labels
    """
    # create matrix of zeros
    vector_labels = np.zeros((len(labels), 10))
    # place a 1 in the correct place for each of the labels
    for i, label in enumerate(labels):
        vector_labels[i][label] = 1
    return vector_labels


def check_predictions(pred, true):
    """

    :param pred:
    :param true:
    :return:
    """
    accuracy = np.zeros(10)
    digit_count = np.zeros(10)
    for p, t in zip(pred, true):
        pr = np.argmax(p)
        tr = np.argmax(t)
        digit_count[tr] += 1
        accuracy[tr] += 1 if pr == tr else 0
    for i in range(10):
        accuracy[i] /= digit_count[i]
    return accuracy


def cross_validate(x: np.ndarray, y: np.ndarray, epochs: int, learn_rate: float, batch_count: int, reg_const: float,
                   reg_norm: str, input_layer_size: int, hidden_layer_size: int, k: int) -> list[np.ndarray]:
    """
    Calculates the accuracy of an MLP model with a single hidden layer on the data provided, using k-fold
    cross-validation.
    :param x: The training data set
    :param y: The training labels
    :param epochs: Number of epochs to train the MLP for
    :param learn_rate: The learning rate of the MLP
    :param batch_count: The number of batches for stochastic gradient descent in MLP training
    :param reg_const: Regularization constant
    :param reg_norm: Regularization norm ("L1" or "L2")
    :param input_layer_size: Number of neurons in the input layer
    :param hidden_layer_size: Number of neurons in the hidden layer of the MLP
    :param k: Number of folds for cross-validation
    :return: List of lists of accuracies of shape (k, 10)
    """
    sample_size = len(x)
    if sample_size != len(y):
        raise Exception("X and y not the same size!")
    indices = np.random.permutation(sample_size)
    split_x = [x[indices[int(i*sample_size/k):int((i+1)*sample_size/k)]] for i in range(k)]
    split_y = [y[indices[int(i*sample_size/k):int((i+1)*sample_size/k)]] for i in range(k)]
    accuracies = []
    for i in range(len(split_x)):
        train_x = np.vstack([split_x[j] for j in range(len(split_x)) if j != i])
        train_y = np.vstack([split_y[j] for j in range(len(split_y)) if j != i])
        test_x = split_x[i]
        test_y = split_y[i]
        pca = PCA(input_layer_size).fit(train_x)
        MLP = NeuralNetwork([input_layer_size, hidden_layer_size, y.shape[1]], ["relu", "softmax"])
        MLP.train(pca.transform(train_x), train_y, epochs, learn_rate, batch_count, reg_const, reg_norm)
        accuracies.append(check_predictions(MLP.predict(pca.transform(test_x)), test_y))
    return accuracies


def random_search(data, param_range, iterations):
    """
    Performs a random search for the hyperparameters
    :param data: The array with data, with no preprocessing or loading done to it
    :param param_range: a dictionary with the ranges for the parameters, "input_layer" (ints), "min_hidden_layer" (int)
        "mu" (floats), "lamda" (floats), "epochs" (ints), "batch" (ints), "reg_norm" ("L1", "L2")
    :param iterations: how many random combinations to do
    """
    results = []
    for i in range(iterations):
        # Choose a random parameter for each in the correct shape
        input_size = np.random.randint(*param_range["input_layer"])
        hidden_size = np.random.randint(param_range["min_hidden_layer"], input_size)
        mu = np.random.uniform(*param_range["mu"])
        lamda = np.random.uniform(*param_range["lamda"])
        epochs = np.random.randint(*param_range["epochs"])
        batch_count = np.random.randint(*param_range["batch"])
        norm = np.random.choice(param_range["reg_norm"])

        (X_train, y_train), _ = load_and_split_digits(data, param_range["train_split"], False)
        X_train = preprocess_images(X_train)
        y_train = preprocess_labels(y_train)

        score = cross_validate(X_train, y_train, epochs, mu, batch_count, lamda,
                               norm, input_size, hidden_size, param_range["k"])
        score = sum(score) / len(score)

        print(f"{i+1}/{iterations}, Score:", score)
        print(f"    Dimensions: {input_size}")
        print(f"    Hidden layer size: {hidden_size}")
        print(f"    Mu: {mu}")
        print(f"    Lamda: {lamda}")
        print(f"    Epochs: {epochs}")
        print(f"    Batch count: {batch_count}")
        print(f"    Regularization: {norm}")
        results.append([score, input_size, hidden_size, mu, lamda, epochs, batch_count, norm])

    results = np.array(results)
    br = results[:, 0].argmax()
    print("Best run:", br, "Score:", results[br, 0])
    print(f"    Dimensions: {int(results[br, 1])}")
    print(f"    Hidden layer size: {int(results[br, 2])}")
    print(f"    Mu: {results[br, 3]}")
    print(f"    Lamda: {results[br, 4]}")
    print(f"    Epochs: {int(results[br, 5])}")
    print(f"    Batch count: {int(results[br, 6])}")
    print(f"    Regularization: {results[br, 7]}")


if __name__ == "__main__":
    # load dataset
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    cwd = os.getcwd()
    mfeat_pix = np.loadtxt(cwd + r'\src\mfeat.pix.txt')

    params = {
        "input_layer": [11, 241],
        "min_hidden_layer": 10,
        "mu": [0.0001, 0.01],
        "lamda": [0.0001, 0.01],
        "epochs": [100, 1000],
        "batch": [1, 10],
        "reg_norm": ["L1", "L2"],
        "train_split": 0.5,
        "k": 10
    }

    random_search(mfeat_pix, params, 4)

    '''
    (X_train, y_train), (X_test, y_test) = load_and_split_digits(mfeat_pix, 0.5, False)

    # flatten and normalize the images if its from mnist
    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)

    pca = PCA(100).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # plt.scatter([i[0] for i in X_train_pca], [i[1] for i in X_train_pca], c=y_train, cmap="tab10")
    # plt.show()

    # change the single answer into a class vector
    y_train = preprocess_labels(y_train)
    y_test = preprocess_labels(y_test)

    # plot some of the digits
    # indices = np.random.permutation(len(X_train))
    # show_digits(X_train[200:225], y_train[200:225])

    accs = cross_validate(X_train, y_train, 200, 0.005, 5, 0.001, "L2", 50, 5)

    print(np.mean([np.mean(acc) for acc in accs]))

    # MLP = NeuralNetwork([240, 50, 10], ["relu", "softmax"])
    # start_time = time.time()
    # epoch_loss = MLP.train(X_train, y_train, 200, 0.005, 5, 0.9, "L2")
    # print("training mlp took ", time.time() - start_time)
    #
    # plt.plot(np.arange(999), epoch_loss[1:])
    # plt.show()
    #
    # y_pred = MLP.predict(X_test)
    #
    # print("Test loss = ", np.mean([loss(pred, test) for pred, test in zip(y_pred, y_test)]))
    #
    # acc = check_predictions(y_pred, y_test)
    # print(acc, " mean: ", np.mean(acc))

    # print([(np.argmax(y_test[i]), np.argmax(y_pred[i])) for i in range(len(y_test))])

    # show_digits(X_test[:25], y_pred[:25])
    '''
