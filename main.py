import numpy as np
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import math

from src.NeuralNetwork import NeuralNetwork


def show_digits(images, labels, figsize=(10,10)):
    # reshape the data so we can plot it
    images = images.reshape(images.shape[0], 28, 28)

    # find how many digits we want to plot
    amount = len(images)
    if amount < len(labels):
        raise Exception("You have too many labels to plot")
    elif amount > len(labels):
        raise Exception("You don't have enough labels to plot")

    # divide the images into rows and columns
    cols = round(math.sqrt(amount))
    rows = cols
    if cols*rows < amount:
        rows = rows+1

    # set up the figure
    fig, axes = plt.subplots(ncols=cols, nrows=rows, sharex=True,
                             sharey=True, figsize=figsize)
    fig.tight_layout()

    # format the axes and add the digits
    for i, ax in enumerate(axes.flat):
        if i < amount:
            ax.set_title(labels[i].argmax())
            ax.imshow(images[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # show the plot
    plt.show()


def preprocess_images(data):
    data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    data = data/255
    return data


def preprocess_labels(labels):
    vector_labels = np.zeros((len(labels), 10))
    for i, label in enumerate(labels):
        vector_labels[i][label] = 1
    return vector_labels


if __name__ == "__main__":
    # load dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # summarize loaded dataset
    print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))

    # flatten and normalize the images
    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)

    # change the single answer into a class vector
    y_train = preprocess_labels(y_train)
    y_test = preprocess_labels(y_test)

    # plot some of the digits
    # show_digits(X_train[0:25], y_train[0:25])

    MLP = NeuralNetwork([784, 30, 10])
    MLP.train(X_train[:10000], y_train[:10000], 3, 0.05)
    y_pred = MLP.predict(X_test[:25])
    # print(y_pred)

    show_digits(X_test[:25], y_pred[:25])
