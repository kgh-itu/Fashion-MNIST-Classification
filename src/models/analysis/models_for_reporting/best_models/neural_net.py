from src.get_train_test_split.fashion_mnist_data import FashionMnistData
from src.models.neural_network.neural_network_classifier import NeuralNetworkClassifier
from src.models.neural_network.layer import DenseLayer

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

data = FashionMnistData()
X_train, y_train, X_test, y_test = data.get_train_test_split()


def best_neural_network():
    our_model = NeuralNetworkClassifier(layers=[DenseLayer(128, "relu"),
                                                DenseLayer(64, "relu"),
                                                DenseLayer(5, "softmax")],
                                        learning_rate=0.01,
                                        epochs=70,
                                        random_state=42)
    our_model.fit(X_train, y_train, batch_size=32, validation_size=0.2)

    return our_model


if __name__ == "__main__":
    model = best_neural_network()
    training_preds = model.predict(X_train)
    preds = model.predict(X_test)

    print("Test ACCURACY", accuracy_score(y_test, preds))
    print("Train ACCURACY", accuracy_score(y_train, training_preds))
    print("F1", list(enumerate(f1_score(y_test, preds, average=None))))
    print("RECALL", list(enumerate(recall_score(y_test, preds, average=None))))
    print("PRECISION", list(enumerate(precision_score(y_test, preds, average=None))))
