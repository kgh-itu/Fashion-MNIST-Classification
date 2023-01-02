from src.get_train_test_split.fashion_mnist_data import FashionMnistData

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

data = FashionMnistData()
x_train, y_train, x_test, y_test = data.get_train_test_split()


def best_random_forest():
    model = RandomForestClassifier(n_estimators=100, bootstrap=True, criterion="gini",
                                   min_samples_split=4, max_depth=21,
                                   random_state=42).fit(x_train, y_train)
    return model


if __name__ == "__main__":
    model = best_random_forest()
    training_preds = model.predict(x_train)
    test_preds = model.predict(x_test)

    print("Test ACCURACY", accuracy_score(y_test, test_preds))
    print("Train ACCURACY", accuracy_score(y_train, training_preds))
    print("F1", list(enumerate(f1_score(y_test, test_preds, average=None))))
    print("RECALL", list(enumerate(recall_score(y_test, test_preds, average=None))))
    print("PRECISION", list(enumerate(precision_score(y_test, test_preds, average=None))))
