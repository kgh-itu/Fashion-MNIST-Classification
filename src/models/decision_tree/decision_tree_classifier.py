from statistics import mode
from src.models.decision_tree.node import Node

import numpy as np
import random


class DecisionTreeClassifier:
    def __init__(self,
                 max_depth=1.e10,
                 min_samples_split=2,
                 criterion="gini",
                 random_state=None):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_state = random_state

        if self.random_state:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        self.X = None
        self.Y = None
        self.root = None
        self.is_fitted = False

    def fit(self, X, y):
        self.X = X
        self.Y = y

        self.root = Node(self.X,
                         self.Y,
                         depth=0,
                         max_depth=self.max_depth,
                         min_samples_split=self.min_samples_split,
                         criterion=self.criterion,
                         random_state=self.random_state)

        self.root._split()
        self.is_fitted = True

        return self

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, X):
        cur_node = self.root
        while cur_node.split_allowed():
            best_feature = cur_node.best_feature
            best_value = cur_node.best_cutoff

            if not cur_node.split_exists():
                break

            if X[best_feature] < best_value:
                if cur_node.left:
                    cur_node = cur_node.left
            else:
                if cur_node.right:
                    cur_node = cur_node.right

        return mode(cur_node.y)

    def get_depth(self):
        return self.count_depth(self.root) - 1

    def get_n_leaves(self):
        return self.count_leaves(self.root)

    def count_leaves(self, root):
        if root is None:
            return 0
        if root.left is None and root.right is None:
            return 1
        return self.count_leaves(root.left) + self.count_leaves(root.right)

    def count_depth(self, root):
        current_depth = 0
        if root.left:
            current_depth = max(current_depth, self.count_depth(root.left))
        if root.right:
            current_depth = max(current_depth, self.count_depth(root.right))
        return current_depth + 1
