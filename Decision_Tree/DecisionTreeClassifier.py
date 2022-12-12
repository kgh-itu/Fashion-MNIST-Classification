from statistics import mode
from Decision_Tree.Node import Node


class DecisionTree:
    def __init__(self,
                 max_depth=1.e10,
                 min_samples_split=2):

        self.X = None
        self.Y = None
        self.root = None

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.is_fitted = False

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        self.root = Node(self.X,
                         self.Y,
                         depth=0,
                         max_depth=self.max_depth,
                         min_samples_split=self.min_samples_split)

        self.root.build_tree()

        self.is_fitted = True

    def predict(self, X):
        return [self._predict(x) for x in X]

    def get_depth(self):
        assert self.is_fitted
        return self.root.get_depth() - 1

    def get_n_leaves(self):
        assert self.is_fitted
        return self.root.get_n_leaves()

    def _predict(self, x):
        cur_node = self.root
        while cur_node.split_allowed:
            best_feature = cur_node.best_feature
            best_value = cur_node.best_cutoff

            if not cur_node.split_exists:
                break

            if x[best_feature] < best_value:
                if cur_node.left:
                    cur_node = cur_node.left
            else:
                if cur_node.right:
                    cur_node = cur_node.right

        return mode(cur_node.Y)
