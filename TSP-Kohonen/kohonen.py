# Q3_graded
# Do not change the above line.

import numpy as np
# from tensorflow.keras import datasets
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import networkx as nx

np.random.seed(0)


data_shape = 2
epochs = 160

learning_rate = 1
radius = 1
cities_location = np.genfromtxt('Cities.csv', delimiter=' ')
# (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

data = cities_location[:, [1, 2]] / 100000
plt.figure()
plt.scatter(data[:, 0], data[:, 1])
data_length = len(data)
# data = X_train / 255

# data = data[:data_counts]

# flatten each image
# data = data.reshape(data_counts, data_shape * data_shape)
# initialize weights
weights = np.random.rand(data_length, data_shape)


def find_distance_to_index(neuron_index):
    return np.abs(np.arange(0, data_length) - neuron_index)


def plot_cities():
    mapped_cities = []
    for i, city in enumerate(data):
        distances = np.sum((weights - city) ** 2, axis=1)  # Euclidean distance between sample and weights
        best_neuron = distances.argmin()  # finding best neuron index
        mapped_cities.append((best_neuron, i))  # first is neuron index on weights and second is sample index on data

    sorted_city = sorted(mapped_cities, key=lambda x: x[
        0])  # sort based on neuron on weight index because every weights which are closer to each other should be
    # appeared one after another
    cities_order = list(map(lambda x: x[1], sorted_city))  # get index of cities which has been saved as tuple
    g = nx.Graph()
    g.add_node(cities_order[0], pos=data[cities_order[0]])
    for idx, node in enumerate(cities_order[1:]):
        g.add_node(node, pos=data[node])
        g.add_edge(cities_order[idx], cities_order[idx + 1])

    # output_plot_data = data[np.array(cities_order)]
    plt.figure()
    # plt.plot(output_plot_data[:, 0], output_plot_data[:, 1], marker='o')
    pos = nx.get_node_attributes(g, 'pos')
    nx.draw(g, pos, node_color="red", node_size=2, edge_color="#2f5fa8")
    plt.show()


stop_points = np.linspace(32, 160, 5)

for e in range(epochs):
    print(f"{e + 1} epoch")
    if e + 1 in stop_points:
        plot_cities()
    for sample in data:
        # competition
        distances = np.sum((weights - sample) ** 2, axis=1)  # Euclidean distance between sample and weights
        best_neuron = distances.argmin()  # finding best neuron index
        distances_matrix = find_distance_to_index(best_neuron)
        distances_matrix = distances_matrix.reshape(-1, 1)
        distances_matrix = (-1 / (2 * radius ** 2)) * distances_matrix
        cooperation_matrix = np.exp(distances_matrix)  # neighborhood function (gaussian)
        weights += learning_rate * (cooperation_matrix * (sample - weights))  # update weights (adaption)
    # learning_rate *= 0.98


