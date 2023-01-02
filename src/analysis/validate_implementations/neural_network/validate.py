import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import seaborn

from src.get_data.fashion_mnist_data import FashionMnistData
from src.models.neural_network.neural_network_classifier import NeuralNetworkClassifier
from src.models.neural_network.layer import DenseLayer

seed = 42
tf.random.set_seed(seed)


def plot(our_hist,
         tf_hist,
         savefig=False):

    epochs = our_hist["epochs"]

    FONTSIZE = 16
    fig, ax = plt.subplots(figsize=(13, 6), tight_layout=True,
                           ncols=2, sharex="all")

    seaborn.set_style("darkgrid")
    seaborn.set_palette("pastel")
    seaborn.lineplot(x=epochs, y=our_hist["train_accuracy"], label="Our Accuracy", ax=ax[0])
    seaborn.lineplot(x=epochs, y=tf_hist["accuracy"], label="Tensorflow Accuracy", ax=ax[0])
    ax[0].legend(fontsize=FONTSIZE)
    ax[0].set_title("Training Accuracy Comparison", fontsize=FONTSIZE)
    ax[0].set_xlabel("Epochs", fontsize=FONTSIZE)
    ax[0].set_ylabel("Accuracy", fontsize=FONTSIZE)

    seaborn.lineplot(x=epochs, y=our_hist["train_loss"], label="Our Loss", ax=ax[1])
    seaborn.lineplot(x=epochs, y=tf_hist["loss"], label="Tensorflow Loss", ax=ax[1])
    ax[1].legend(fontsize=FONTSIZE)
    ax[1].set_title("Training Loss Comparison", fontsize=FONTSIZE)
    ax[1].set_xlabel("Epochs", fontsize=FONTSIZE)
    ax[1].set_ylabel("Loss", fontsize=FONTSIZE)

    ax[0].tick_params(axis='both', which='both', labelsize=FONTSIZE)
    ax[1].tick_params(axis='both', which='both', labelsize=FONTSIZE)

    if savefig:
        plt.savefig("reports/figures_for_report/tensorflow_vs_our")

    plt.show()


def train_models(epochs=100, random_state=42):
    lr = 0.01
    data = FashionMnistData()
    X_train, y_train, X_test, y_test = data.get_train_test_split()

    our_model = NeuralNetworkClassifier(layers=[DenseLayer(128, "relu"),
                                                DenseLayer(64, "relu"),
                                                DenseLayer(5, "softmax")],
                                        learning_rate=lr,
                                        epochs=epochs,
                                        random_state=42)

    our_history = our_model.fit(X_train, y_train, batch_size=32)

    y_train = keras.utils.to_categorical(y_train, 5)
    X_train = X_train.reshape(-1, 28, 28)
    tf.random.set_seed(random_state)
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
    our_hist, tf_hist = train_models(epochs=200)
    plot(our_hist, tf_hist)
