import numpy as np


class AdalineAlgo:
    def __init__(self, rate=0.01, niter=15, shuffle=True):
        self.learning_rate = rate
        self.niter = niter
        self.shuffle = shuffle

        # Weight's vector
        self.weight = []

        # Cost function
        self.cost_ = []

    def fit(self, X, y):
        """
        Fit training data.
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.shape : [#samples]
        :param X:
        :param y:
        :return:
        """
        row = X.shape[0]
        col = X.shape[1]

        #  add bias to X
        X_bias = np.ones((row, col + 1))
        X_bias[:, 1:] = X
        X = X_bias

        # initialize weights
        np.random.seed(1)
        self.weight = np.random.rand(col + 1)

        # training
        for _ in range(self.niter):
            if self.shuffle:
                X, y = self._shuffle(X, y)

            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def _update_weights(self, xi, target):
        """
        output: Common output
        y: Desired output
        :param xi:
        :param target:
        :return:
        """
        output = self.net_input(xi)
        error = target - output
        """
        We defined: X.T[0] = 1 
        this is the reason why the bias calc (= self.weight[0])
        will be equal to: 
        self.weight[0] += self.learning_rate * errors
        """
        self.weight += self.learning_rate * xi.dot(error)
        cost = 0.5 * (error ** 2)
        return cost

    def _shuffle(self, X, y):
        """
        Shuffle training data with np random permutation
        :param X:
        :param y:
        :return:
        """
        per = np.random.permutation(len(y))
        return X[per], y[per]

    def net_input(self, X):
        """
        Calculate net input.
        :param X:
        :return:
        """
        return X @ self.weight

    def activation(self, X):
        """
        Compute linear activation.
        :param X:
        :return:
        """
        return self.net_input(X)

    def predict(self, X):
        """
        Return class label after unit step.
        :param X:
        :return:
        """
        # if x is list instead of np.array
        if type(X) is list:
            X = np.array(X)

        # add bias to x if he doesn't exist
        if len(X.T) != len(self.weight):
            X_bias = np.ones((X.shape[0], X.shape[1] + 1))
            X_bias[:, 1:] = X
            X = X_bias

        return np.where(self.activation(X) > 0.0, 1, -1)

    def score(self, X, y):
        """
        Model score is calculated based on comparison of
        expected value and predicted value.
        :param X:
        :param y:
        :return:
        """
        wrong_prediction = abs((self.predict(X) - y) / 2).sum()
        self.score_ = (len(X) - wrong_prediction) / len(X)
        return self.score_
