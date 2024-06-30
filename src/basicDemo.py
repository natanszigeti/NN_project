# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

# Load the pixel data
mfeat_pix = np.loadtxt('mfeat.pix.txt')

# Plot the figure from the lecture notes
plt.figure(1)
for i in range(10):
    for j in range(10):
        pic = mfeat_pix[200 * i + j, :]
        picmatreverse = np.zeros((15, 16))
        picmatreverse[:] = -pic.reshape((15, 16), order='F')
        picmat = np.zeros((15, 16))
        for k in range(15):
            picmat[:, k] = picmatreverse[:, 15 - k]
        plt.subplot(10, 10, i * 10 + j + 1)
        plt.pcolor(picmat.T, cmap='gray', edgecolors='k', linewidths=0.1)
        plt.axis('off')
plt.show()

# Split the data into a training and a testing dataset
pick_indices = np.concatenate([np.arange(1, 101), np.arange(201, 301), np.arange(401, 501),
                               np.arange(601, 701), np.arange(801, 901), np.arange(1001, 1101),
                               np.arange(1201, 1301), np.arange(1401, 1501), np.arange(1601, 1701),
                               np.arange(1801, 1901)]) - 1
train_patterns = mfeat_pix[pick_indices, :]
test_patterns = mfeat_pix[pick_indices + 100, :]

# Create indicator matrices size 10 x 1000 with the class labels coded by binary indicator vectors
b = np.ones(100)
train_labels = block_diag(b, b, b, b, b, b, b, b, b, b)
test_labels = train_labels

# Create a row vector of correct class labels (from 1 ... 10 for the 10 classes)
correct_labels = np.concatenate([b, 2*b, 3*b, 4*b, 5*b, 6*b, 7*b, 8*b, 9*b, 10*b])

# Compute meanTrainImages
mean_train_images = np.zeros((240, 10))
for i in range(10):
    mean_train_images[:, i] = np.mean(train_patterns[i*100:(i+1)*100, :], axis=0)

feature_values_train = mean_train_images.T @ train_patterns.T
feature_values_test = mean_train_images.T @ test_patterns.T

# Compute linear regression weights W
W = np.linalg.inv(feature_values_train @ feature_values_train.T) @ feature_values_train @ train_labels.T
W = W.T

# Compute train misclassification rate
classification_hypotheses_train = W @ feature_values_train
max_indices_train = np.argmax(classification_hypotheses_train, axis=0) + 1
nr_of_misclassifications_train = np.sum(correct_labels != max_indices_train)
print(f'train misclassification rate = {nr_of_misclassifications_train / 1000:.3g}')

# Compute test misclassification rate
classification_hypotheses_test = W @ feature_values_test
max_indices_test = np.argmax(classification_hypotheses_test, axis=0) + 1
nr_of_misclassifications_test = np.sum(correct_labels != max_indices_test)
print(f'test misclassification rate = {nr_of_misclassifications_test / 1000:.3g}')

# Ridge regression on raw pics
alpha = 10000
Wridge = np.linalg.inv(train_patterns.T @ train_patterns + alpha * np.eye(240)) @ train_patterns.T @ train_labels.T
Wridge = Wridge.T

# Compute train misclassification rate for ridge regression
classification_hypotheses_train_ridge = Wridge @ train_patterns.T
max_indices_train_ridge = np.argmax(classification_hypotheses_train_ridge, axis=0) + 1
nr_of_misclassifications_train_ridge = np.sum(correct_labels != max_indices_train_ridge)
print(f'train misclassification rate ridge = {nr_of_misclassifications_train_ridge / 1000:.3g}')

# Compute test misclassification rate for ridge regression
classification_hypotheses_test_ridge = Wridge @ test_patterns.T
max_indices_test_ridge = np.argmax(classification_hypotheses_test_ridge, axis=0) + 1
nr_of_misclassifications_test_ridge = np.sum(correct_labels != max_indices_test_ridge)
print(f'test misclassification rate ridge = {nr_of_misclassifications_test_ridge / 1000:.3g}')
