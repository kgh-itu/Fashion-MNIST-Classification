from numpy import where
from random import shuffle

from src.models.decision_tree.gini import (calculate_gini,
                                           get_node_gini)

from src.models.decision_tree.cutoff import get_possible_cutoffs


class Node:
    def __init__(self, X, Y,
                 depth=None,
                 max_depth=1.e10,
                 min_samples_split=2):

        self.X = X
        self.Y = Y
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.num_features = self.X.shape[1]
        self.size = len(self.Y)
        self.gini = get_node_gini(self.Y)
        self.left = None
        self.right = None
        self.best_feature = None
        self.best_cutoff = None
        self.best_gain = 0
        self._split_exists = False

    def build_tree(self):
        self.best_feature, self.best_cutoff = self.get_best_split()

        if self.split_exists() and self.split_allowed():
            left_split = where(self.X[:, self.best_feature] <= self.best_cutoff)
            right_split = where(self.X[:, self.best_feature] > self.best_cutoff)

            if len(left_split) == 0 or len(right_split) == 0:
                return None

            left_X, left_Y = self.X[left_split], self.Y[left_split]
            right_X, right_Y = self.X[right_split], self.Y[right_split]

            left = Node(left_X, left_Y,
                        depth=self.depth + 1,
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split)

            self.left = left
            self.left.build_tree()

            right = Node(right_X, right_Y,
                         depth=self.depth + 1,
                         max_depth=self.max_depth,
                         min_samples_split=self.min_samples_split)
            self.right = right
            self.right.build_tree()

    def get_best_split(self):
        features = list(range(self.num_features))
        shuffle(features)

        for feature in features:
            curr_feature = self.X[:, feature]
            possible_cutoffs = get_possible_cutoffs(curr_feature)
            shuffle(possible_cutoffs)
            for cutoff in possible_cutoffs:
                left_Y = self.Y[where(curr_feature <= cutoff)]
                right_Y = self.Y[where(curr_feature > cutoff)]

                gini_of_split = calculate_gini(left_Y, right_Y)
                gini_of_current_node = self.gini
                gini_gain = gini_of_current_node - gini_of_split

                if gini_gain > self.best_gain:
                    self.best_feature = feature
                    self.best_cutoff = cutoff
                    self.best_gain = gini_gain

        if self.best_feature is not None and self.best_cutoff is not None:
            self._split_exists = True

        return self.best_feature, self.best_cutoff

    def split_allowed(self):
        max_depth_satisfied = self.depth < self.max_depth
        min_samples_split_satisfied = self.size >= self.min_samples_split
        return max_depth_satisfied and min_samples_split_satisfied

    def split_exists(self):
        return self._split_exists
