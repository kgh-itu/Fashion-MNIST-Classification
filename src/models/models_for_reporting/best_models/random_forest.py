from src.get_train_test_split.fashion_mnist_data import FashionMnistData

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

data = FashionMnistData()
x_train, y_train, x_test, y_test = data.get_train_test_split()
x_train.shape[0]

model = RandomForestClassifier(n_estimators=100, bootstrap=False, criterion="entropy",
                               min_samples_leaf=2, max_depth=None,
                               max_features="auto",
                               random_state=42).fit(x_train, y_train,)

training_preds = model.predict(x_train)
preds = model.predict(x_test)

print("Test ACCURACY", accuracy_score(y_test, preds))
print("Train ACCURACY", accuracy_score(y_train, training_preds))
print("F1", list(enumerate(f1_score(y_test, preds, average=None))))
print("RECALL", list(enumerate(recall_score(y_test, preds, average=None))))
print("PRECISION", list(enumerate(precision_score(y_test, preds, average=None))))