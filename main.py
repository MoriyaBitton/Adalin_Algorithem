import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from AdalineAlgo import AdalineAlgo
import seaborn as sns


def creatData(d_size, Part, n):
    data = np.empty((d_size, 2), dtype=object)
    random.seed(10)
    for i in range(d_size):
        data[i, 0] = (random.randint(-n, n) / n)
        data[i, 1] = (random.randint(-n, n) / n)

    train = np.zeros(d_size)

    if Part == "A":
        for i in range(d_size):
            if data[i][1] > 0.5 and data[i][0] > 0.5:
                train[i] = 1
            else:
                train[i] = -1

    if Part == "B":
        for i in range(d_size):
            if 0.5 <= (data[i][1] ** 2 + data[i][0] ** 2) <= 0.75:
                train[i] = 1
            else:
                train[i] = -1

    X = data.astype(np.float64)  # test
    y = train.astype(np.float64)  # train

    return X, y


def plot_decision_regions(X, y, classifier):
    # plot the decision surface
    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, 0.02),
                           np.arange(y_min, y_max, 0.02))
    pred = classifier.predict(np.array([xx1.flatten(), xx2.flatten()]).T)
    pred = pred.reshape(xx1.shape)
    colors = ListedColormap(('red', 'blue'))

    # background colors --> showed our prediction
    plt.contourf(xx1, xx2, pred, alpha=0.1, cmap=colors)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.scatter(x=X[y == -1, 1], y=X[y == -1, 0],
                alpha=0.9, c='red',
                marker='s', label=-1.0)

    plt.scatter(x=X[y == 1, 1], y=X[y == 1, 0],
                alpha=0.9, c='blue',
                marker='x', label=1.0)


def partA():
    print("\nPart A\n")

    # create a Adaline classifier and train on our data
    d_size = 1000
    X, y = creatData(d_size, "A", 100)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # learning rate = 0.01 || data = 1,000 || n = 100
    classifier = AdalineAlgo(0.01, 10).fit(X, y)
    ax[0].plot(range(1, len(classifier.cost_) + 1), classifier.cost_, marker='o')
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Cost')
    ax[0].set_title('Part A: Adaline Algorithm \ndata 1,000 || Learning rate: 0.01 || n = 100')

    # learning rate = 0.0001 || data = 1,000 || n = 100
    classifier2 = AdalineAlgo(0.0001, 10).fit(X, y)
    ax[1].plot(range(1, len(classifier2.cost_) + 1), classifier2.cost_, marker='o')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Cost')
    ax[1].set_title('Part A: Adaline Algorithm \ndata 1,000 || Learning rate: 0.0001 || n = 100')
    plt.show()

    # ______________________________
    # case 1
    # ------------------------------

    # Learning rate: 1/100 || data = 1,000 || n = 100
    print("Learning rate: 1/100 || data = 1,000 || n = 100\n")
    print("score: ", classifier.score(X, y) * 100, "%")
    print("cost: ", np.array(classifier.cost_).min())

    # plot our miss-classification error after each iteration of training
    plot_decision_regions(X, y, classifier=classifier)
    plt.title('Part A: Adaline Algorithm \ndata 1,000 || Learning rate: 0.01 || n = 100')
    plt.legend(loc='upper left')
    plt.show()

    # confusion_matrix
    cm = confusion_matrix(classifier.predict(X), y)
    plt.subplots()
    sns.heatmap(cm, fmt=".0f", annot=True)
    plt.title("confusion matrix \ndata 1,000 || Learning rate: 0.01 || n = 100")
    plt.xlabel("Test")
    plt.ylabel("Predict")
    plt.show()

    # ______________________________
    # case 2
    # ------------------------------

    # Learning rate: 1/10,000 || data = 1,000 || n = 100
    print("\nLearning rate: 1/10,000 || data = 1,000 || n = 100\n")
    print("score: ", classifier2.score(X, y) * 100, "%")
    print("cost: ", np.array(classifier2.cost_).min())

    # plot our miss-classification error after each iteration of training
    plot_decision_regions(X, y, classifier=classifier2)
    plt.title('Part A: Adaline Algorithm \ndata 1,000 || Learning rate: 0.0001 || n = 100')
    plt.legend(loc='upper left')
    plt.show()

    # confusion_matrix
    cm = confusion_matrix(classifier2.predict(X), y)
    plt.subplots()
    sns.heatmap(cm, fmt=".0f", annot=True)
    plt.title("confusion matrix \ndata 1,000 || Learning rate: 0.0001 || n = 100")
    plt.xlabel("Test")
    plt.ylabel("Predict")
    plt.show()

    # ______________________________
    # case 3
    # ------------------------------

    # create a Adaline classifier and train on our data
    X_, y_ = creatData(10000, "A", 10000)
    classifier3 = AdalineAlgo(0.1, 10).fit(X_, y_)

    # Learning rate: 1/100 || data = 1,000 || n = 10,000
    print("\nLearning rate: 1/100 || data = 1,000 || n = 10,000\n")
    print("score: ", classifier3.score(X_, y_) * 100, "%")
    print("cost: ", np.array(classifier3.cost_).min())

    # plot our miss-classification error after each iteration of training
    plot_decision_regions(X_, y_, classifier=classifier3)
    plt.title('Part A: Adaline Algorithm \ndata 10,000 || Learning rate: 0.1 || n = 10,000')
    plt.legend(loc='upper left')
    plt.show()

    # confusion_matrix
    cm_ = confusion_matrix(classifier3.predict(X_), y_)
    plt.subplots()
    sns.heatmap(cm_, fmt=".0f", annot=True)
    plt.title("confusion matrix \ndata 10,000 || Learning rate: 0.1 || n = 10,000")
    plt.xlabel("Test")
    plt.ylabel("Predict")
    plt.show()


def partB():
    print("\nPart B\n")

    # create a Adaline classifier and train on our data
    # data = 1,000
    X1, y1 = creatData(1000, "B", 100)
    classifier = AdalineAlgo(0.0001, 50).fit(X1, y1)

    # data = 100,000
    X2, y2 = creatData(100000, "B", 100)
    classifier2 = AdalineAlgo(0.0001, 50).fit(X2, y2)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # learning rate = 0.0001 || d_type = 1,000 || n = 100
    ax[0].plot(range(1, len(classifier.cost_) + 1), classifier.cost_, marker='o')
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Cost')
    ax[0].set_title('Part B: Adaline Algorithm \ndata 1,000 || Learning rate: 0.0001')

    # learning rate = 0.0001 || d_type = 100,000 || n = 100
    ax[1].plot(range(1, len(classifier2.cost_) + 1), classifier2.cost_, marker='o')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Cost')
    ax[1].set_title('Part B: Adaline Algorithm \ndata 100,000 || Learning rate: 0.0001')
    plt.show()

    # ______________________________
    # case 1
    # ------------------------------

    # learning rate = 0.0001 || d_type = 1,000 || n = 100
    print("Learning rate: 1/10,000 || data = 1,000\n")
    print("score: ", classifier.score(X1, y1) * 100, "%")
    print("cost: ", np.array(classifier.cost_).min())

    # plot our miss-classification error after each iteration of training
    plot_decision_regions(X1, y1, classifier=classifier)
    plt.title('Part B: Adaline Algorithm \ndata 1,000 || Learning rate: 0.0001')
    plt.legend(loc='upper left')
    plt.show()

    # confusion_matrix
    cm = confusion_matrix(classifier.predict(X1), y1)
    plt.subplots()
    sns.heatmap(cm, fmt=".0f", annot=True)
    plt.title("confusion matrix \ndata 1,000 || Learning rate: 0.0001")
    plt.xlabel("Test")
    plt.ylabel("Predict")
    plt.show()

    # ______________________________
    # case 2
    # ------------------------------

    # learning rate = 0.0001 || d_type = 100,000 || n = 100
    print("\nLearning rate: 1/10,000 || data = 100,000\n")
    print("score: ", classifier2.score(X2, y2) * 100, "%")
    print("cost: ", np.array(classifier2.cost_).min())

    # plot our miss-classification error after each iteration of training
    plot_decision_regions(X2, y2, classifier=classifier2)
    plt.title('Part B: Adaline Algorithm \ndata 100,000 || Learning rate: 0.0001')
    plt.legend(loc='upper left')
    plt.show()

    # confusion_matrix
    cm = confusion_matrix(classifier.predict(X2), y2)
    plt.subplots()
    sns.heatmap(cm, fmt=".0f", annot=True)
    plt.title("confusion matrix \ndata 100,000 || Learning rate: 0.0001")
    plt.xlabel("Test")
    plt.ylabel("Predict")
    plt.show()


if __name__ == '__main__':
    partA()
    partB()
