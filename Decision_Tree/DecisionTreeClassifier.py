import numpy as np
from sklearn.datasets import load_iris
import random
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

data = pd.read_csv(
    r"https://raw.githubusercontent.com/Eligijus112/decision-tree-python/main/data/classification/train.csv")
data = data[["Survived", "Age", "Fare"]].dropna()
y = data["Survived"].to_numpy()
x = data[["Age", "Fare"]].to_numpy()


class MyDecisionTreeClassifier:
    def __init__(self,
                 max_depth=None,
                 min_samples_split=2):
        self.X = None
        self.Y = None
        self.root = None

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.is_fitted = False

    def fit(self, X, Y):
        if self.is_fitted:
            raise Exception("Already Fitted")

        self.X = X
        self.Y = Y

        self.root = Node(self.X,
                         self.Y,
                         node_type="root",
                         max_depth=self.max_depth,
                         min_samples_split=self.min_samples_split)

        self.root.grow()

        self.is_fitted = True

    def predict(self, X):

        cur_node = self.root
        while cur_node.depth < cur_node.max_depth:
            best_feature = cur_node.best_feature
            best_value = cur_node.best_cutoff
            if cur_node.n < cur_node.min_samples_split:
                break

            if best_value is None or best_feature is None:
                break

            if X[best_feature] < best_value:
                if cur_node.left is not None:
                    cur_node = cur_node.left
            else:
                if cur_node.right is not None:
                    cur_node = cur_node.right

        return max(set(list(cur_node.Y)), key=list(cur_node.Y).count)


class Node:
    def __init__(self, X, Y, depth=None, max_depth=None, min_samples_split=2, node_type=None):

        self.X = X
        self.Y = Y

        self.n = len(self.Y)
        self.gini = self._get_leaf_gini(self.Y)

        self.left = None
        self.right = None

        self.n_features = self.X.shape[1]
        self.n_samples = len(X)
        self.n_classes = len(np.unique(Y))

        self.depth = depth if depth else 0
        self.max_depth = max_depth if max_depth else 5
        self.min_samples_split = min_samples_split if min_samples_split else 2
        self.node_type = node_type if node_type else 'root'

        self.best_feature = None
        self.best_cutoff = None

    def grow(self):
        best_feature, cutoff = self.get_best_split()

        if best_feature is not None:

            if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):
                self.best_feature = best_feature
                self.best_cutoff = cutoff

                left_idx = np.where(self.X[:, best_feature] <= cutoff)
                right_idx = np.where(self.X[:, best_feature] > cutoff)
                left_X, left_Y = self.X[left_idx], self.Y[left_idx]
                right_X, right_Y = self.X[right_idx], self.Y[right_idx]

                left = Node(left_X, left_Y,
                            depth=self.depth + 1,
                            max_depth=self.max_depth,
                            node_type="left",
                            min_samples_split=self.min_samples_split)

                self.left = left
                self.left.grow()

                right = Node(right_X, right_Y,
                             depth=self.depth + 1,
                             max_depth=self.max_depth,
                             node_type="right",
                             min_samples_split=self.min_samples_split)
                self.right = right
                self.right.grow()

    def get_best_split(self):
        max_gain = 0
        best_feature = None
        best_cutoff = None

        for feature in range(self.n_features):
            curr_x = self.X[:, feature]
            current_feature_rolling_mean = np.sort(self.rolling_average(np.unique(curr_x), 2))

            for cutoff in current_feature_rolling_mean:
                left_split, right_split = np.where(curr_x <= cutoff), np.where(curr_x > cutoff)
                left_Y, right_Y = self.Y[left_split], self.Y[right_split]

                gini = calculate_total_gini_of_split(left_Y, right_Y)
                base_gini = self.gini

                gini_gain = base_gini - gini

                if gini_gain > max_gain:
                    best_feature = feature
                    best_cutoff = cutoff

                    # Setting the best gain to the current one
                    max_gain = gini_gain

        self.best_feature = best_feature
        self.best_cutoff = best_cutoff

        return best_feature, best_cutoff

    def calculate_total_gini_of_split(self, left_Y, right_Y):
        left_gini = self._get_leaf_gini(left_Y)
        right_gini = self._get_leaf_gini(right_Y)

        left_weight = len(left_Y) / (len(left_Y) + len(right_Y))
        right_weight = len(right_Y) / (len(left_Y) + len(right_Y))

        return left_gini * left_weight + right_gini * right_weight

    @staticmethod
    def _get_leaf_gini(Y: np.array):
        labels, count = np.unique(Y, return_counts=True)
        probs_squared = (count / len(Y)) ** 2
        return 1 - probs_squared.sum()

    @staticmethod
    def rolling_average(x: np.array, window) -> np.array:
        return np.convolve(x, np.ones(window), 'valid') / window


def calculate_total_gini_of_split(left_Y, right_Y):
    left_gini = _get_leaf_gini(left_Y)
    right_gini = _get_leaf_gini(right_Y)

    left_weight = len(left_Y) / (len(left_Y) + len(right_Y))
    right_weight = len(right_Y) / (len(left_Y) + len(right_Y))

    return left_gini * left_weight + right_gini * right_weight


def _get_leaf_gini(Y: np.array):
    labels, count = np.unique(Y, return_counts=True)
    probs_squared = (count / len(Y)) ** 2
    return 1 - probs_squared.sum()


a = MyDecisionTreeClassifier(min_samples_split=200, max_depth=100)
a.fit(x, y)

my_preds = []

for i in range(len(x)):
    val = x[i]
    preds = a.predict(val)
    my_preds.append(preds)

print("my accuracy score", accuracy_score(my_preds, y))

sklearndtree = DecisionTreeClassifier(min_samples_split=200, max_depth=100).fit(x, y)
sklearn_preds = sklearndtree.predict(x)
print("sklearn accuracy score", accuracy_score(sklearn_preds, y))
