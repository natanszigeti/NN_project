from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import math

def show_digits(images, labels, figsize=(10,10)):
    # Find how many digits we want to plot
    amount = len(images)
    if amount < len(labels):
        raise Exception("You have too many labels to plot")
    elif amount > len(labels):
        raise Exception("You don't have enough labels to plot")

    # divide the images into rows and columns
    cols = round(math.sqrt(amount))
    rows = cols
    if (cols*rows < amount):
        rows = rows+1

    # set up the figure
    fig, axes = plt.subplots(ncols=cols, nrows=rows, sharex=True,
                             sharey=True, figsize=figsize)
    fig.tight_layout()

    # format the axes and add the digits
    for i, ax in enumerate(axes.flat):
        if i < amount:
            ax.set_title(labels[i])
            ax.imshow(images[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # show the plot
    plt.show()


if __name__ == "__main__":
    # load dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # summarize loaded dataset
    print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))
    max = 0
    for i in range(28):
        for j in range(28):
            if X_train[9][i][j] > max:
                max = X_train[5][i][j]

    print(max)
    # plot some of the digits
    show_digits(X_train[0:25], y_train[0:25])
