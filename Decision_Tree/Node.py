from numpy import where
from random import shuffle

from Decision_Tree.helpers import (calculate_gini,
                                   get_node_gini,
                                   get_possible_cutoffs)


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

        self.left = None
        self.right = None

        self.current_gini = get_node_gini(self.Y)

        self.num_features = self.X.shape[1]
        self.size = len(self.Y)

        self.best_feature = None
        self.best_cutoff = None
        self.best_gain = 0

        # can we split further
        self.max_depth_satisfied = self.depth < self.max_depth
        self.min_samples_split_satisfied = self.size >= self.min_samples_split

        self.split_exists = False
        self.split_allowed = (self.max_depth_satisfied
                              and self.min_samples_split_satisfied)

    def build_tree(self):
        self.best_feature, self.best_cutoff = self.get_best_split()

        if self.split_exists and self.split_allowed:
            left_mask = where(self.X[:, self.best_feature] <= self.best_cutoff)
            right_mask = where(self.X[:, self.best_feature] > self.best_cutoff)

            if len(left_mask) == 0 or len(right_mask) == 0:
                return None

            left_X, left_Y = self.X[left_mask], self.Y[left_mask]
            right_X, right_Y = self.X[right_mask], self.Y[right_mask]

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
        shuffle(features)  # important for randomness of choosing best split

        for feature in features:
            curr_feature = self.X[:, feature]
            possible_cutoffs = get_possible_cutoffs(curr_feature)
            shuffle(possible_cutoffs)  # important for randomness of choosing best split
            for cutoff in possible_cutoffs:
                left_Y = self.Y[where(curr_feature <= cutoff)]
                right_Y = self.Y[where(curr_feature > cutoff)]

                gini_of_split = calculate_gini(left_Y, right_Y)
                gini_of_current_node = self.current_gini
                gini_gain = gini_of_current_node - gini_of_split

                if gini_gain > self.best_gain:
                    self.best_feature = feature
                    self.best_cutoff = cutoff
                    self.best_gain = gini_gain

        if self.best_feature is not None and self.best_cutoff is not None:
            self.split_exists = True

        return self.best_feature, self.best_cutoff

    def get_depth(self):
        current_depth = 0
        if self.left:
            current_depth = max(current_depth, self.left.get_depth())
        if self.right:
            current_depth = max(current_depth, self.right.get_depth())
        return current_depth + 1

    def get_n_leaves(self):
        if self.left and self.right:
            return self.left.get_n_leaves() + self.right.get_n_leaves()
        else:
            return 1
