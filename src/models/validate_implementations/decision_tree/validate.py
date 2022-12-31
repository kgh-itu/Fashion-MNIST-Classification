from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import accuracy_score
import numpy as np

from src.models.decision_tree.decision_tree_classifier import DecisionTreeClassifier

LABEL_SIZE = 15


def accuracy(X, y,
             fitted_sk: list[DecisionTreeClassifier],
             fitted_our: list[DecisionTreeClassifier]):

    our_accuracy = [accuracy_score(m.predict(X), y) for m in fitted_our]
    sk_accuracy = [accuracy_score(m.predict(X), y) for m in fitted_sk]

    our_mean = round(np.mean(our_accuracy), 3)
    sk_mean = round(np.mean(sk_accuracy), 3)

    ax = _get_ax()
    ax[0].set_ylabel("Count", fontsize=LABEL_SIZE)
    ax[0].hist(our_accuracy)
    ax[0].set_title(f"Our Accuracy (mean = {our_mean})", fontsize=LABEL_SIZE)
    ax[1].hist(sk_accuracy)
    ax[1].set_title(f"Sklearn Accuracy (mean = {sk_mean})", fontsize=LABEL_SIZE)

    ax[0].tick_params(axis='both', which='both', labelsize=13)
    ax[1].tick_params(axis='both', which='both', labelsize=13)

    plt.savefig(f"reports/figures_for_report/our_vs_sklearn_accuracy")


def depth(fitted_sk: list[DecisionTreeClassifier],
          fitted_our: list[DecisionTreeClassifier]):
    our_depth = [m.get_depth() for m in fitted_our]
    sklearn_depth = [m.get_depth() for m in fitted_sk]

    ax = _get_ax()
    ax[0].set_ylabel("Count", fontsize=LABEL_SIZE)
    ax[0].hist(our_depth)
    ax[0].set_title("Our implementation Tree Depth", fontsize=LABEL_SIZE)
    ax[1].hist(sklearn_depth)
    ax[1].set_title("Sklearn implementation Tree Depth", fontsize=LABEL_SIZE)

    plt.savefig(f"reports/figures_for_report/our_vs_sklearn_depth")


def leaf(fitted_sk: list[DecisionTreeClassifier],
         fitted_our: list[DecisionTreeClassifier]):

    our_leaves = [m.get_n_leaves() for m in fitted_our]
    sklearn_leaves = [m.get_n_leaves() for m in fitted_sk]

    ax = _get_ax()
    ax[0].set_ylabel("Count", fontsize=LABEL_SIZE)
    ax[0].hist(our_leaves)
    ax[0].set_title("Our implementation Tree Leaves", fontsize=LABEL_SIZE)
    ax[1].hist(sklearn_leaves)
    ax[1].set_title("Sklearn implementation Tree Leaves", fontsize=LABEL_SIZE)

    plt.savefig(f"reports/figures_for_report/our_vs_sklearn_leaves")


def _get_ax():
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5), tight_layout=True, sharey="all", sharex="all")
    seaborn.set_style("dark")
    return ax


if __name__ == "__main__":
    data = load_digits()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    fitted_our = [DecisionTreeClassifier().fit(X_train, y_train) for _ in range(100)]
    fitted_sk = [DecisionTreeClassifier().fit(X_train, y_train) for _ in range(100)]
    accuracy(X_test, y_test, fitted_sk, fitted_our)
    depth(fitted_sk, fitted_our)
    leaf(fitted_sk, fitted_our)

