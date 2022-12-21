import numpy as np
import matplotlib.pyplot as plt
import random

from code.get_train_test.prepare_fashion_mnist_dataset import FashionMnistData

random.seed(69)


def extract_sample_in_class(X, y, cls):
    mask = np.where(y == cls)
    X = X[mask]
    x = random.choice(X).reshape(28, 28)
    return x


def sample_from_each_class():
    data = FashionMnistData()
    X_train, y_train, X_test, y_test = data.get()
    samples = [extract_sample_in_class(X_train, y_train, i) for i in range(5)]
    title = lambda i: f"{i}"
    fig = plt.figure(figsize=(15, 6))
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0.01)

    for i in range(len(samples)):
        ax = fig.add_subplot(1, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(samples[i], cmap=plt.cm.bone, interpolation='nearest')
        plt.title(title(i), fontdict={"fontsize": 15})
    fig_name = "samples_from_classes.png"
    plt.savefig(f"reports/figures_for_report/{fig_name}", bbox_inches='tight', pad_inches=0.2)


if __name__ == "__main__":
    sample_from_each_class()
