from src.get_train_test_split.fashion_mnist_data import FashionMnistData

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from src.get_train_test_split.fashion_mnist_data import FashionMnistData


def find_best_hyper_parameters():
    dat = FashionMnistData()
    x_train, y_train, x_test, y_test = dat.get_train_test_split()

    param_grid = {"max_depth": [None, 10, 15, 20, 21, 22, 23],
                  "min_samples_split": [2, 4, 8],
                  'criterion': ["gini", "entropy"]}

    model = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(model, param_grid)
    grid.fit(x_train, y_train)

    best_min_samples_split = grid.best_estimator_.get_params()['min_samples_split']
    best_criterion = grid.best_estimator_.get_params()['criterion']
    best_max_depth = grid.best_estimator_.get_params()['max_depth']
    best_bootstrap = grid.best_estimator_.get_params()['bootstrap']

    print('Best min_samples_split:', best_min_samples_split)
    print('Best Criterion:', best_criterion)
    print('Best Max Depth:', best_max_depth)
    print('Best bootstrap:', best_bootstrap)

    return grid


def get_best_model(gridsearch_instance):
    return gridsearch_instance.best_estimator_


def get_accuracy(clf, x_test, y_test):
    y_preds = clf.predict(x_test)
    acc = accuracy_score(y_preds, y_test)
    print(acc)
    return acc


if __name__ == "__main__":
    data = FashionMnistData()
    x_tr, y_tr, x_te, y_te = data.get_train_test_split()
    grid = find_best_hyper_parameters()
    best_model = get_best_model(grid).fit(x_tr, y_tr)
    preds = get_accuracy(best_model, x_te, y_te)
