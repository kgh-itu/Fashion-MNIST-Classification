from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import accuracy_score

from src.models.decision_tree.decision_tree_classifier import DecisionTree


def accuracy(X, y,
             fitted_sk: list[DecisionTreeClassifier],
             fitted_our: list[DecisionTree]):
    our_accuracy = [accuracy_score(m.predict(X), y) for m in fitted_our]
    sk_accuracy = [accuracy_score(m.predict(X), y) for m in fitted_sk]

    ax = _get_ax()
    ax[0].set_ylabel("Count")
    ax[0].hist(our_accuracy)
    ax[0].set_title("Our implementation Accuracy")
    ax[1].set_ylabel("Count")
    ax[1].hist(sk_accuracy)
    ax[1].set_title("Sklearn implementation Accuracy")

    plt.savefig(f"reports/figures_for_report/our_vs_sklearn_accuracy")


def depth(fitted_sk: list[DecisionTreeClassifier],
          fitted_our: list[DecisionTree]):
    our_depth = [m.get_depth() for m in fitted_our]
    sklearn_depth = [m.get_depth() for m in fitted_sk]

    ax = _get_ax()
    ax[0].set_ylabel("Count")
    ax[0].hist(our_depth)
    ax[0].set_title("Our implementation Tree Depth")
    ax[1].set_ylabel("Count")
    ax[1].hist(sklearn_depth)
    ax[1].set_title("Sklearn implementation Tree Depth")

    plt.savefig(f"reports/figures_for_report/our_vs_sklearn_depth")


def leaf(fitted_sk: list[DecisionTreeClassifier],
         fitted_our: list[DecisionTree]):
    our_leaves = [m.get_n_leaves() for m in fitted_our]
    sklearn_leaves = [m.get_n_leaves() for m in fitted_sk]

    ax = _get_ax()
    ax[0].set_ylabel("Count")
    ax[0].hist(our_leaves)
    ax[0].set_title("Our implementation Tree Leaves")
    ax[1].set_ylabel("Count")
    ax[1].hist(sklearn_leaves)
    ax[1].set_title("Sklearn implementation Tree Leaves")

    plt.savefig(f"reports/figures_for_report/our_vs_sklearn_leaves")


def _get_ax():
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5), tight_layout=True)
    seaborn.set_style("dark")
    seaborn.set(font="Futura")
    return ax


if __name__ == "__main__":
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    fitted_sk = [DecisionTreeClassifier().fit(X_train, y_train) for _ in range(1)]
    fitted_our = [DecisionTree().fit(X_train, y_train) for _ in range(1)]

    accuracy(X_test, y_test, fitted_sk, fitted_our)
    depth(fitted_sk, fitted_our)
    leaf(fitted_sk, fitted_our)
    plt.show()
