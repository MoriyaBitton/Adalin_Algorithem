import random
import numpy as np
import matplotlib.pyplot as plt


class Kohonen(object):
    def __init__(self, h, w, alpha_start=0.6, seed=9, r=3):
        """
        Initialize the Kohonen object with a given map size
        :param seed: {int} random seed to use
        :param h: {int} height of the map
        :param w: {int} width of the map
        :param dim: {int} dimensions of the map
        :param alpha_start: {float} initial alpha (learning rate) at training start
        :param sigma_start: {float} initial sigma (restraint / neighborhood function) at training start; if `None`: w / 2
        """
        np.random.seed(seed)
        self.shape = (h, w)
        self.alpha_start = alpha_start
        self.radius_strength = r
        self.d = None

    def fit(self, data, interval=1000, print_mode=5):
        """
        Train the SOM on the given data for several iterations
        :param print_mode:
        :param data: {numpy.ndarray} data to train on
        :param interval: {int} interval of epochs to use for saving training errors
        """
        self.d = data
        self.iteration_limit = interval
        x_min = np.min(data[:, 0])
        y_min = np.min(data[:, 1])
        x_max = np.max(data[:, 0])
        y_max = np.max(data[:, 1])

        self.map = np.array([[(random.uniform(x_min + 0.001, x_max),
                               random.uniform(y_min + 0.001, y_max)) for i in range(self.shape[1])] for j in
                             range(self.shape[0])])
        for t in range(self.iteration_limit):
            # randomly pick an input vector
            n = random.randint(0, len(data) - 1)
            # the input vector who chosen
            bmu = self.best_neuron(data[n])
            if t % print_mode == 0:
                Draw(self, x_max, x_min, t)
            self.update_map(bmu, data[n], t)

    def best_neuron(self, vector):
        """
        Compute the winner neuron closest to the vector (Euclidean distance)
        :param vector: {numpy.ndarray} vector of current data point(s)
        :return: indices of winning neuron
        """
        min_neuron_dest = np.inf
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                euclid_dist = np.linalg.norm(vector - self.map[x, y])
                if euclid_dist < min_neuron_dest:
                    min_neuron_dest = euclid_dist
                    ans = (x, y)
        return ans

    def update_map(self, bmu, X_i, t):
        """
        Update map by found BMU at iteration t.
        :param bmu: best neuron indices
        :param X_i: sample - target input vector
        :param t: current iteration
        """
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                dist_from_bmu = np.linalg.norm(np.array(bmu) - np.array([x, y]))
                alpha = self.alpha_start * np.exp(-t / 300)  # update alpha
                radius = np.exp(-np.power(dist_from_bmu, 2) / self.radius_strength)  # update radius
                self.map[(x, y)] += alpha * radius * (X_i - self.map[(x, y)])


def Draw(self, max, min, t):
    xs = []
    ys = []
    for i in range(self.map.shape[0]):
        for j in range(self.map.shape[1]):
            xs.append(self.map[i, j, 0])
            ys.append(self.map[i, j, 1])

    fig, ax = plt.subplots()
    ax.scatter([xs], [ys], c='r')
    ax.set_xlim(min, max)
    ax.set_ylim(min, max)
    ax.plot(xs, ys, 'b-')
    ax.scatter(self.d[:, 0], self.d[:, 1], alpha=0.3)
    ax.set_title("Data size:" + str(len(self.d)) + " | "
                 + "Iter:" + str(self.iteration_limit) + " | "+
                 "Epoch:" + str(t) +
                 "\n" +
                 "LR:" + str(self.alpha_start) + " | "
                 + "Net size:" + str(self.shape) + " | "
                 + "R:" + str(self.radius_strength))
    plt.show()
