import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import seaborn

from src.get_train_test_split.fashion_mnist_data import FashionMnistData
from src.models.neural_network.neural_network_classifier import NeuralNetworkClassifier
from src.models.neural_network.layer import DenseLayer

seed = 42
tf.random.set_seed(seed)


def accuracy(our_hist, tf_hist):
    epochs = our_hist["epochs"]

    fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
    seaborn.set_style("dark")
    seaborn.set(font="Futura")
    seaborn.lineplot(x=epochs, y=our_hist["train_accuracy"], label="Our Accuracy")
    seaborn.lineplot(x=epochs, y=tf_hist["accuracy"], label="Tensorflow Accuracy")
    ax.legend()
    ax.set_title("Training Loss Comparison")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    plt.savefig(f"reports/figures_for_report/tf_acc_vs_our_acc")


def loss(our_hist, tf_hist):
    epochs = our_hist["epochs"]

    fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
    seaborn.set_style("darkgrid")
    seaborn.lineplot(x=epochs, y=our_hist["train_loss"], label="Our Loss")
    seaborn.lineplot(x=epochs, y=tf_hist["loss"], label="Tensorflow Loss")
    ax.legend()
    ax.set_title("Training Accuracy Comparison")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    plt.savefig(f"reports/figures_for_report/tf_loss_vs_our_loss")


def train_models(epochs=100):
    lr = 0.01
    data = FashionMnistData()
    X_train, y_train, X_test, y_test = data.get_train_test_split()

    our_model = NeuralNetworkClassifier(layers=[DenseLayer(128, "relu"),
                                                DenseLayer(64, "relu"),
                                                DenseLayer(5, "softmax")],
                                        learning_rate=lr,
                                        epochs=epochs)

    our_history = our_model.fit(X_train, y_train, batch_size=32)

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

    tf_history = tf_model.fit(X_train, y_train, epochs=epochs, batch_size=32)
    return our_history, tf_history.history


if __name__ == "__main__":
    our_history, tf_history = train_models(epochs=200)
    accuracy(our_history, tf_history)
    loss(our_history, tf_history)
    plt.show()
