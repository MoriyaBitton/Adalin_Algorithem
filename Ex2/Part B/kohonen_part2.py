import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn_som.som import SOM


def getdata():
    """
    Make 2 random data sets at "monkey hand" shape using 20X20 matrix's.
    :return: data1 , data2
    """
    data1 = np.zeros((1500, 2))
    data2 = np.zeros((1500, 2))

    random.seed(11)
    Mask1 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    Mask2 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    Mask1 = Mask1[::-1].T
    Mask2 = Mask2[::-1].T
    n = 0
    random.seed(1)
    while n < 1500:
        x = random.uniform(0, 20)
        y = random.uniform(0, 20)
        i = int(x)
        j = int(y)
        if Mask1[i, j] == 1:
            data1[n, 0] = x / 20
            data1[n, 1] = y / 20
            n += 1
    n = 0
    while n < 1500:
        x = random.uniform(0, 20)
        y = random.uniform(0, 20)
        i = int(x)
        j = int(y)
        if Mask2[i, j] == 1:
            data2[n, 0] = x / 20
            data2[n, 1] = y / 20
            n += 1

    return data1, data2


def modify_fit(modle, X, epochs=1, shuffle=True, data=1):
    """
    Take data (a tensor of type float64) as input and fit the SOM to that
    data for the specified number of epochs.

    Parameters
    ----------
    X : ndarray
        Training data. Must have shape (n, self.dim) where n is the number
        of training samples.
    epochs : int, default=1
        The number of times to loop through the training data when fitting.
    shuffle : bool, default True
        Whether or not to randomize the order of train data when fitting.
        Can be seeded with np.random.seed() prior to calling fit.

    Returns
    -------
    None
        Fits the SOM to the given data but does not return anything.
        :param X:
        :param modle:
        :param shuffle:
        :param epochs:
        :param data:
    """
    # Count total number of iterations
    global_iter_counter = 0
    n_samples = X.shape[0]
    total_iterations = np.minimum(epochs * n_samples, modle.max_iter)

    for epoch in range(epochs):
        # Break if past max number of iterations
        if global_iter_counter > modle.max_iter:
            break
        np.random.seed(1)
        if shuffle:
            indices = np.random.permutation(n_samples)
        else:
            indices = np.arange(n_samples)
        # Train
        c = 0
        for idx in indices:
            # Break if past max number of iterations
            if global_iter_counter > modle.max_iter:
                break
            input = X[idx]
            # Do one step of training
            modle.step(input)
            # Update learning rate
            global_iter_counter += 1
            modle.lr = (1 - (global_iter_counter / total_iterations)) * modle.initial_lr

            if c % 499 == 0 and data == 1:
                draw(modle, X, "data: " + str(data) + "\nepoch: " + str(epoch) + " | indices: " + str(c)
                     + " | "
                     + "Iter:" + str(modle.max_iter) + " | " +
                     "LR:" + str(modle.initial_lr) + " | "
                     )
            if c % 100 == 0 and data == 2:
                draw(modle, X, "data: " + str(data) + "\nepoch: " + str(epoch) + " | indices: " + str(c)
                     + " | "
                     + "Iter:" + str(modle.max_iter) + " | " +
                     "LR:" + str(modle.initial_lr) + " | "
                     )

            c += 1

    # Compute inertia
    inertia = np.sum(np.array([float(modle._compute_point_intertia(x)) for x in X]))
    modle._inertia_ = inertia

    # Set n_iter_ attribute
    modle._n_iter_ = global_iter_counter

    return


def draw(model, X, s):
    """
    draw the data and neuron net.
    :param model:
    :param X:
    :param s:
    :return:
    """
    re_wx = model.weights[:, 0].reshape(model.m, model.n)
    re_wy = model.weights[:, 1].reshape(model.m, model.n)
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for i in range(re_wx.shape[0]):
        xh = []
        yh = []
        xs = []
        ys = []
        for j in range(re_wx.shape[1]):
            xs.append(re_wx[i, j])
            ys.append(re_wy[i, j])
            xh.append(re_wx[j, i])
            yh.append(re_wy[j, i])
        ax.plot(xs, ys, 'r-', markersize=0, linewidth=1)
        ax.plot(xh, yh, 'r-', markersize=0, linewidth=1)
    ax.plot(re_wx, re_wy, color='b', marker='o', linewidth=0, markersize=3)
    ax.scatter(X[:, 0], X[:, 1], c="b", alpha=0.08)
    plt.title(s)
    plt.savefig(s + ".png")
    plt.show()


def part2():
    d1, d2 = getdata()
    fig, axs = plt.subplots(1, 2)
    axs[0].scatter(d1[:, 0], d1[:, 1], c="b")
    axs[0].set_title('4 fingers')
    axs[1].scatter(d2[:, 0], d2[:, 1], c="b")
    axs[1].set_title('3 fingers')
    plt.show()

    model = SOM(m=15, n=15, dim=2, lr=1.5, sigma=1.25, max_iter=15000)
    modify_fit(model, d1, 10, data=1)
    model._inertia = None
    model._n_iter_ = None
    modify_fit(model, d2, 10, data=2)


if __name__ == '__main__':
    part2()
