from src.get_train_test_split.fashion_mnist_data import FashionMnistData

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# best random forest model based on hyper-parameters found in:

# src/models/models_for_reporting/best_hyper_parameters/random_forest.py

data = FashionMnistData()
x_train, y_train, x_test, y_test = data.get_train_test_split()

model = RandomForestClassifier(bootstrap=False, criterion="entropy",
                               min_samples_leaf=2, max_depth=None,
                               max_features="auto",
                               random_state=42).fit(x_train, y_train,)
preds = model.predict(x_test)
print(accuracy_score(y_test, preds))