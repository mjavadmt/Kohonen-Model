# Q1_graded
# Do not change the above line.
import numpy as np
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(5, 5), dpi=80)

np.random.seed(0)

grid_width = 20
data_counts = 5000
data_shape = 28
epochs = 400
# without learning rate decay and radius decay lr = 0.25 , radius = 1
# with just learning rate decay lr = 10 , radius = 1
# with both learning rate decay and radius decay lr = 2, radius = 2 epoch = 150
learning_rate = 0.25
radius = 1

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

data = X_train / 255

data = data[:data_counts]

# flatten each image
data = data.reshape(data_counts, data_shape * data_shape)
# initialize weights
weights = np.random.rand(grid_width * grid_width, data_shape * data_shape)


def find_dists(shape, index):
    i, j = np.indices(shape, sparse=True)
    return (i - index[0]) ** 2 + (j - index[1]) ** 2


def transform_flattened_to_2d(index):
    return index // grid_width, index % grid_width


# make grid
def plot_weights(weights):
    weights_grid = np.zeros((grid_width * data_shape, grid_width * data_shape))
    for i in range(len(weights)):
        each_weight = weights[i]
        convert_to_image = each_weight.reshape(data_shape, data_shape)
        weights_grid[(i // grid_width) * data_shape: ((i // grid_width) + 1) * data_shape,
        (i % grid_width) * data_shape: ((i % grid_width) + 1) * data_shape] = convert_to_image
    figure(figsize=(5, 5), dpi=80)
    plt.imshow(weights_grid, cmap="gray")
    plt.show()


stop_points = [80, 160, 240, 320, 400]
# training
for e in range(epochs):
    print(f"{e} epoch")
    batch_data = data[np.random.choice(len(data), size=128, replace=False)]
    if (e + 1) in stop_points:
        plot_weights(weights)
    for sample in batch_data:
        # competition
        distances = np.sum((weights - sample) ** 2, axis=1)  # Euclidean distance between sample and weights
        best_neuron = distances.argmin()  # finding best neuron index
        neuron_index = transform_flattened_to_2d(best_neuron)  # change one dimensional to two dimensional
        distances_matrix = find_dists((grid_width, grid_width), neuron_index)  # distances to best neuron index
        distances_matrix = distances_matrix.reshape(-1, 1)  # flatten distance matrix to be multipliable
        distances_matrix = (-1 / (2 * radius ** 2)) * distances_matrix
        cooperation_matrix = np.exp(distances_matrix)  # neighborhood function (gaussian)
        weights += learning_rate * (cooperation_matrix * (sample - weights))  # update weights (adaption)



# Q1_graded
# Do not change the above line.
grid_width = 20
data_counts = 5000
data_shape = 28
epochs = 400
# without learning rate decay and radius decay lr = 0.25 , radius = 1
# with just learning rate decay lr = 10 , radius = 1
# with both learning rate decay and radius decay lr = 2, radius = 2 epoch = 150
learning_rate = 10
radius = 1

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

data = X_train / 255

data = data[:data_counts]

# flatten each image
data = data.reshape(data_counts, data_shape * data_shape)
# initialize weights
weights = np.random.rand(grid_width * grid_width, data_shape * data_shape)


def find_dists(shape, index):
    i, j = np.indices(shape, sparse=True)
    return (i - index[0]) ** 2 + (j - index[1]) ** 2


def transform_flattened_to_2d(index):
    return index // grid_width, index % grid_width


# make grid
def plot_weights(weights):
    weights_grid = np.zeros((grid_width * data_shape, grid_width * data_shape))
    for i in range(len(weights)):
        each_weight = weights[i]
        convert_to_image = each_weight.reshape(data_shape, data_shape)
        weights_grid[(i // grid_width) * data_shape: ((i // grid_width) + 1) * data_shape,
        (i % grid_width) * data_shape: ((i % grid_width) + 1) * data_shape] = convert_to_image
    figure(figsize=(5, 5), dpi=80)
    plt.imshow(weights_grid, cmap="gray")
    plt.show()


stop_points = [80, 160, 240, 320, 400]
# training
for e in range(epochs):
    print(f"{e} epoch")
    batch_data = data[np.random.choice(len(data), size=128, replace=False)]
    if (e + 1) in stop_points:
        plot_weights(weights)
    for sample in batch_data:
        # competition
        distances = np.sum((weights - sample) ** 2, axis=1)  # Euclidean distance between sample and weights
        best_neuron = distances.argmin()  # finding best neuron index
        neuron_index = transform_flattened_to_2d(best_neuron)  # change one dimensional to two dimensional
        distances_matrix = find_dists((grid_width, grid_width), neuron_index)  # distances to best neuron index
        distances_matrix = distances_matrix.reshape(-1, 1)  # flatten distance matrix to be multipliable
        distances_matrix = (-1 / (2 * radius ** 2)) * distances_matrix
        cooperation_matrix = np.exp(distances_matrix)  # neighborhood function (gaussian)
        weights += learning_rate * (cooperation_matrix * (sample - weights))  # update weights (adaption)
    learning_rate *= 0.98
# Type your code here

# Q1_graded
# Do not change the above line.
grid_width = 20
data_counts = 5000
data_shape = 28
epochs = 150
# without learning rate decay and radius decay lr = 0.25 , radius = 1
# with just learning rate decay lr = 10 , radius = 1
# with both learning rate decay and radius decay lr = 2, radius = 2 epoch = 150
learning_rate = 2
radius = 2

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

data = X_train / 255

data = data[:data_counts]

# flatten each image
data = data.reshape(data_counts, data_shape * data_shape)
# initialize weights
weights = np.random.rand(grid_width * grid_width, data_shape * data_shape)


def find_dists(shape, index):
    i, j = np.indices(shape, sparse=True)
    return (i - index[0]) ** 2 + (j - index[1]) ** 2


def transform_flattened_to_2d(index):
    return index // grid_width, index % grid_width


# make grid
def plot_weights(weights):
    weights_grid = np.zeros((grid_width * data_shape, grid_width * data_shape))
    for i in range(len(weights)):
        each_weight = weights[i]
        convert_to_image = each_weight.reshape(data_shape, data_shape)
        weights_grid[(i // grid_width) * data_shape: ((i // grid_width) + 1) * data_shape,
        (i % grid_width) * data_shape: ((i % grid_width) + 1) * data_shape] = convert_to_image
    figure(figsize=(5, 5), dpi=80)
    plt.imshow(weights_grid, cmap="gray")
    plt.show()


stop_points = [30, 60, 90, 120, 150]
# training
for e in range(epochs):
    print(f"{e} epoch")
    batch_data = data[np.random.choice(len(data), size=128, replace=False)]
    if (e + 1) in stop_points:
        plot_weights(weights)
    for sample in batch_data:
        # competition
        distances = np.sum((weights - sample) ** 2, axis=1)  # Euclidean distance between sample and weights
        best_neuron = distances.argmin()  # finding best neuron index
        neuron_index = transform_flattened_to_2d(best_neuron)  # change one dimensional to two dimensional
        distances_matrix = find_dists((grid_width, grid_width), neuron_index)  # distances to best neuron index
        distances_matrix = distances_matrix.reshape(-1, 1)  # flatten distance matrix to be multipliable
        distances_matrix = (-1 / (2 * radius ** 2)) * distances_matrix
        cooperation_matrix = np.exp(distances_matrix)  # neighborhood function (gaussian)
        weights += learning_rate * (cooperation_matrix * (sample - weights))  # update weights (adaption)
    learning_rate *= 0.975
    radius *= 0.999
# Type your code here

