import os
import numpy as np
# from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import math

from src.NeuralNetwork import NeuralNetwork


def load_digits(data: np.ndarray, train_split: int, random: bool = True):
    """
    function for splitting the digits task into a train and test set
    :param data: the data from mfeat
    :param train_split: how we want to split the data
    :param random: whether we want the data to be returned in a random order
    :return: the train and test data like so (X_train, y_train), (X_test, y_test)
    """
    # calculate how much the test split is
    test_split = 100 - train_split

    # create train and test labels
    y_train = np.repeat(np.arange(10), (train_split * (data.shape[0] // 100)) // 10)
    y_test = np.repeat(np.arange(10), (test_split * (data.shape[0] // 100)) // 10)

    X_train = []
    X_test = []
    for i in range(0, data.shape[0], 100):
        chunk = data[i:i + 100]  # get 100 samples (all the same label)
        # extend the data with the right amount of samples
        X_train.extend(chunk[:train_split])
        X_test.extend(chunk[test_split:])
    # convert the lists to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    if random:
        # generate a random permutation of indices and shuffle the train and test data
        train_indices = np.random.permutation(len(y_train))
        test_indices = np.random.permutation(len(y_test))
        X_train = X_train[train_indices]
        X_test = X_test[test_indices]
        y_train = y_train[train_indices]
        y_test = y_test[test_indices]

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
    data = data/6 #255 for mnist
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


if __name__ == "__main__":
    # load dataset
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    cwd = os.getcwd()
    mfeat_pix = np.loadtxt(cwd + '\src\mfeat.pix.txt')

    (X_train, y_train), (X_test, y_test) = load_digits(mfeat_pix, 80)

    # flatten and normalize the images if its from mnist
    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)

    # change the single answer into a class vector
    y_train = preprocess_labels(y_train)
    y_test = preprocess_labels(y_test)

    # plot some of the digits
    # show_digits(X_train[200:225], y_train[200:225])

    MLP = NeuralNetwork([240, 30, 10])
    MLP.train(X_train, y_train, 3, 0.05)
    y_pred = MLP.predict(X_test)

    show_digits(X_test[:25], y_pred[:25])
