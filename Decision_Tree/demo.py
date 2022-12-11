# Demo of performance of sklearn DecisionTreeClassifier vs our own implementation
from sklearn.datasets import (load_breast_cancer,
                              load_iris,
                              load_digits)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from Decision_Tree.DecisionTreeClassifier import DecisionTree


def print_summary(classifier, x_test_, y_test_):
    print("Accuracy Score", accuracy_score(classifier.predict(x_test_), y_test_), "")
    print("Number of leaf nodes", classifier.get_n_leaves())
    print("Depth of tree", classifier.get_depth())


if __name__ == "__main__":
    data_sets = [load_digits(), load_breast_cancer(), load_iris()]
    data_sets_names = ["DIGITS (load_digits())",
                       "BREAST CANCER (load_breast_cancer())",
                       "IRIS (load_iris())"]

    for i, data_set in enumerate(data_sets):
        print("DATASET", data_sets_names[i])
        X, y = data_sets[i].data, data_sets[i].target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("-----------SKLEARN DTREE-----------")
        sk_dtree = DecisionTreeClassifier().fit(X_train, y_train)
        print_summary(sk_dtree, X_test, y_test)
        print("-----------SKLEARN DTREE-----------")
        print()
        print("DATASET", data_sets_names[i])
        print("-----------OUR DTREE-----------")
        our_dtree = DecisionTree()
        our_dtree.fit(X_train, y_train)
        print_summary(our_dtree, X_test, y_test)
        print("-----------OUR DTREE-----------")
        print()

