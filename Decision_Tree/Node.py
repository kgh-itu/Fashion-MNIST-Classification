import numpy as np
import random

from Decision_Tree.helpers import (calculate_gini,
                                   get_node_gini,
                                   get_possible_cutoffs)


class Node:
    def __init__(self, X, Y,
                 depth=None,
                 max_depth=np.inf,
                 min_samples_split=2):

        self.X = X
        self.Y = Y

        self.gini = get_node_gini(self.Y)

        self.left = None
        self.right = None

        self.num_features = self.X.shape[1]
        self.n = len(self.Y)

        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.best_feature = None
        self.best_cutoff = None

    def build_tree(self):
        best_feature, cutoff = self.get_best_split()

        if best_feature is not None:

            if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):
                self.best_feature = best_feature
                self.best_cutoff = cutoff

                left_ = np.where(self.X[:, best_feature] <= cutoff)
                right_ = np.where(self.X[:, best_feature] > cutoff)

                left_X, left_Y = self.X[left_], self.Y[left_]
                right_X, right_Y = self.X[right_], self.Y[right_]

                if len(left_Y) == 0 or len(right_Y) == 0:
                    print(len(left_Y))
                    print(len(right_Y))
                    print(True)
                    return

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
        best_gain = 0
        best_feature = None
        best_cutoff = None

        for feature in range(self.num_features):
            curr_feature = self.X[:, feature]
            possible_cutoffs = get_possible_cutoffs(curr_feature)

            for cutoff in possible_cutoffs:
                left_Y = self.Y[np.where(curr_feature <= cutoff)]
                right_Y = self.Y[np.where(curr_feature > cutoff)]

                gini_of_split = calculate_gini(left_Y, right_Y)
                gini_of_current_node = self.gini
                gain = gini_of_current_node - gini_of_split

                # todo decide whether we want to introduce randomness to our model
                #  meaning if the new split is as good as the previous best
                #  choose a random one of them
                #   the following uncommented lines does that....

                if gain == best_gain and gain != 0:
                    # if the new split is as good as the previous best choose a random one of them
                    # best_feature, best_cutoff = random.choice(
                    # [(best_feature, best_cutoff),
                    # (feature, cutoff)])
                    pass

                elif gain > best_gain:
                    best_feature = feature
                    best_cutoff = cutoff
                    best_gain = gain

        self.best_feature = best_feature
        self.best_cutoff = best_cutoff

        return best_feature, best_cutoff

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
