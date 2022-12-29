import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
import seaborn

from src.get_train_test_split.fashion_mnist_data import FashionMnistData
from src.models.neural_network.neural_network_classifier import NeuralNetworkClassifier
from src.models.neural_network.layer import DenseLayer


def accuracy(our_model, tf_model):
    fig, ax = plt.subplots()
    seaborn.set_style("dark")
    seaborn.set(font="Futura")
    ax.plot(list(range(100)), tf_model.history["accuracy"], label="Tensorflow Accuracy")
    ax.plot(list(range(100)), our_model.train_accuracy, label="Our Accuracy")
    ax.legend()
    ax.set_title("Loss Comparison")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    plt.savefig(f"reports/figures_for_report/tf_acc_vs_our_acc")



def loss(our_model, tf_model):
    fig, ax = plt.subplots()
    seaborn.set_style("dark")
    seaborn.set(font="Futura")
    ax.plot(list(range(100)), tf_model.history["loss"], label="Tensorflow Loss")
    ax.plot(list(range(100)), our_model.train_loss, label="Our Loss")
    ax.legend()
    ax.set_title("Accuracy Comparison")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    plt.savefig(f"reports/figures_for_report/tf_loss_vs_our_loss")


def train_models():
    epochs = 100
    lr = 0.01
    data = FashionMnistData()
    X_train, y_train, X_test, y_test = data.get_train_test_split(normalize=True)

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                    test_size=0.05,
                                                                    train_size=0.95,
                                                                    random_state=42)
    our_model = NeuralNetworkClassifier(layers=[DenseLayer(128, "relu"),
                                                DenseLayer(64, "relu"),
                                                DenseLayer(5, "softmax")],
                                        learning_rate=lr,
                                        epochs=epochs)

    our_model.fit(X_train, y_train, batch_size=32, X_validation=X_validation, y_validation=y_validation)
    our_model.plot_performance()

    y_train = keras.utils.to_categorical(y_train, 5)
    X_train = X_train.reshape(-1, 28, 28)

    tf_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(5, activation='softmax'),
    ])

    # Compile the model
    tf_model.compile(optimizer='sgd',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

    history = tf_model.fit(X_train, y_train, epochs=epochs, batch_size=32)
    return our_model, history





if __name__ == "__main__":
    our_nn, tf_nn = train_models()
    accuracy(our_nn, tf_nn)
    loss(our_nn, tf_nn)