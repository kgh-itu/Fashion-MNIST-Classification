import numpy as np
import random

from src.models.decision_tree.gain import (calculate_gini,
                                           get_node_gini,
                                           calculate_entropy,
                                           get_node_entropy)

from src.models.decision_tree.cutoff import get_possible_cutoffs


class Node:
    def __init__(self, X, y,
                 depth=None,
                 max_depth=1.e10,
                 min_samples_split=2,
                 criterion="gini",
                 random_state=None):

        self.X = X
        self.y = y
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        if random_state:
            np.random.seed(random_state)
            random.seed(random_state)

        if criterion == "gini":
            self.gain = get_node_gini(self.y)
            self.criterion = calculate_gini

        elif criterion == "entropy":
            self.gain = get_node_entropy(self.y)
            self.criterion = calculate_entropy

        self.num_features = self.X.shape[1]
        self.size = len(self.y)
        self.left = None
        self.right = None
        self.best_feature = None
        self.best_cutoff = None
        self.best_gain = 0
        self._split_exists = False

    def _split(self):
        self.best_feature, self.best_cutoff = self.get_best_split()

        if self.split_exists() and self.split_allowed():
            left_split = np.where(self.X[:, self.best_feature] <= self.best_cutoff)
            right_split = np.where(self.X[:, self.best_feature] > self.best_cutoff)

            if len(left_split) == 0 or len(right_split) == 0:
                return None

            left_X, left_Y = self.X[left_split], self.y[left_split]
            right_X, right_Y = self.X[right_split], self.y[right_split]

            left = Node(left_X, left_Y,
                        depth=self.depth + 1,
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        random_state=self.random_state)

            self.left = left
            self.left.split()

            right = Node(right_X, right_Y,
                         depth=self.depth + 1,
                         max_depth=self.max_depth,
                         min_samples_split=self.min_samples_split,
                         random_state=self.random_state)

            self.right = right
            self.right._split()

    def get_best_split(self):
        features = list(range(self.num_features))
        random.shuffle(features)

        for feature in features:
            curr_feature = self.X[:, feature]
            possible_cutoffs = get_possible_cutoffs(curr_feature)
            random.shuffle(possible_cutoffs)
            for cutoff in possible_cutoffs:
                left_Y = self.y[np.where(curr_feature <= cutoff)]
                right_Y = self.y[np.where(curr_feature > cutoff)]

                gain_of_split = self.criterion(left_Y, right_Y)
                gain_current_node = self.gain
                gain = gain_current_node - gain_of_split

                if gain > self.best_gain:
                    self.best_feature = feature
                    self.best_cutoff = cutoff
                    self.best_gain = gain

        if self.best_feature is not None and self.best_cutoff is not None:
            self._split_exists = True

        return self.best_feature, self.best_cutoff

    def split_allowed(self):
        max_depth_satisfied = self.depth < self.max_depth
        min_samples_split_satisfied = self.size >= self.min_samples_split
        return max_depth_satisfied and min_samples_split_satisfied

    def split_exists(self):
        return self._split_exists
