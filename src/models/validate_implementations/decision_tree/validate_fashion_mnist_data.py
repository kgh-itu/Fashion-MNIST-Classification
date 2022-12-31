from src.get_train_test_split.fashion_mnist_data import FashionMnistData
from src.models.decision_tree.decision_tree_classifier import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score

data = FashionMnistData()
x_train, y_train, x_test, y_test = data.get_train_test_split()

model = DecisionTreeClassifier(criterion="entropy", max_depth=5)

model.fit(x_train, y_train)
print(accuracy_score(model.predict(x_train), y_train))