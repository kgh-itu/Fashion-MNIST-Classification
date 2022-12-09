import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from statistics import mode
from Data.prepare_fashion_mnist_dataset import FashionMnistData
import pickle
import datetime
import time


class DecisionTree:
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
        self.X = X
        self.Y = Y

        self.root = Node(self.X,
                         self.Y,
                         node_type="root",
                         max_depth=self.max_depth,
                         min_samples_split=self.min_samples_split)

        self.root.build_tree()

    def predict(self, X):
        predictions = []
        for x_ in X:
            cur_node = self.root
            while cur_node.depth < cur_node.max_depth:
                best_feature = cur_node.best_feature
                best_value = cur_node.best_cutoff
                if cur_node.n <= cur_node.min_samples_split:
                    break

                if best_value is None or best_feature is None:
                    break

                if x_[best_feature] < best_value:
                    if cur_node.left is not None:
                        cur_node = cur_node.left
                else:
                    if cur_node.right is not None:
                        cur_node = cur_node.right

            predictions.append(cur_node.predict)

        return predictions


class Node:
    def __init__(self, X, Y,
                 depth=None, max_depth=None,
                 min_samples_split=2, node_type=None):

        self.X = X
        self.Y = Y

        self.predict = mode(self.Y)
        self.gini = self._get_leaf_gini(self.Y)

        self.left = None
        self.right = None

        self.num_features = self.X.shape[1]
        self.n = len(self.Y)

        self.depth = depth if depth else 0
        self.max_depth = max_depth if max_depth else 5
        self.min_samples_split = min_samples_split if min_samples_split else 2
        self.type = node_type if node_type else 'root'

        self.best_feature = None
        self.best_cutoff = None

    def build_tree(self):
        best_feature, cutoff = self.get_best_split()

        if best_feature is not None:

            if (self.depth < self.max_depth) and (self.n > self.min_samples_split):
                self.best_feature = best_feature
                self.best_cutoff = cutoff

                left_ = np.where(self.X[:, best_feature] <= cutoff)
                right_ = np.where(self.X[:, best_feature] > cutoff)

                left_X, left_Y = self.X[left_], self.Y[left_]
                right_X, right_Y = self.X[right_], self.Y[right_]

                left = Node(left_X, left_Y,
                            depth=self.depth + 1,
                            max_depth=self.max_depth,
                            node_type="left",
                            min_samples_split=self.min_samples_split)

                self.left = left
                self.left.build_tree()

                right = Node(right_X, right_Y,
                             depth=self.depth + 1,
                             max_depth=self.max_depth,
                             node_type="right",
                             min_samples_split=self.min_samples_split)
                self.right = right
                self.right.build_tree()

    def get_best_split(self):
        curr_gini_gain = 0
        best_feature = None
        best_cutoff = None

        for feature in range(self.num_features):
            curr_feature = self.X[:, feature]
            current_feature_rolling_mean = np.sort(self.rolling_average(np.unique(curr_feature), 2))

            for cutoff in current_feature_rolling_mean:
                left_Y = self.Y[np.where(curr_feature <= cutoff)]
                right_Y = self.Y[np.where(curr_feature > cutoff)]

                gini_of_split = self.calculate_gini(left_Y, right_Y)
                gini_of_current_node = self.gini
                gini_gain = gini_of_current_node - gini_of_split

                if gini_gain > curr_gini_gain:
                    best_feature = feature
                    best_cutoff = cutoff
                    curr_gini_gain = gini_gain

        self.best_feature = best_feature
        self.best_cutoff = best_cutoff

        return best_feature, best_cutoff

    def calculate_gini(self, left_Y, right_Y):
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


if __name__ == "__main__":
    start_time = time.time()
    data = FashionMnistData().get()
    x_train = data[0]
    y_train = data[1]
    x_test = data[2]
    y_test = data[3]
    # depth = 30
    # min_samples_split = 100
    # Tree = DecisionTree(max_depth=depth, min_samples_split=min_samples_split)
    # with open("pre_trained_model/config.txt", "w") as f:
    # f.write(f"Model last trained {datetime.datetime.today()}"
    # f"\nwith max_depth={depth}, min_samples_split={min_samples_split}")

    # Tree.fit(x_train, y_train)

    # pickle.dump(Tree, open("pre_trained_model/decision_tree.sav", 'wb'))

    # print("--- %s seconds ---" % (time.time() - start_time))

    # preds = Tree.predict(x_train)
    # print(accuracy_score(preds, y_train))

    loaded_model = pickle.load(open("pre_trained_model/decision_tree.sav", 'rb'))
    preds = loaded_model.predict(x_test)

    print("my tree accuracy score", accuracy_score(preds, y_test))

    tree = DecisionTreeClassifier(max_depth=30, min_samples_split=100).fit(x_train, y_train)
    preds_ = tree.predict(x_test)
    print("sklearn tree accuracy score", accuracy_score(preds_, y_test))
