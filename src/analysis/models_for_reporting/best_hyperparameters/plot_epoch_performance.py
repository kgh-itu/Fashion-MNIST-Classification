import matplotlib.pyplot as plt
import seaborn

from src.get_data.fashion_mnist_data import FashionMnistData
from src.models.neural_network.neural_network_classifier import NeuralNetworkClassifier
from src.models.neural_network.layer import DenseLayer


def plot_performance_after_epochs(history,
                                  savefig=False):
    epochs = history["epochs"]
    fig, ax = plt.subplots(nrows=2, ncols=2,
                           figsize=(15, 5), tight_layout=True,
                           sharex="all")
    seaborn.set_style("darkgrid")
    seaborn.set_palette("pastel")
    seaborn.lineplot(x=epochs, y=history["train_accuracy"], ax=ax[0, 0])
    seaborn.lineplot(x=epochs, y=history["train_loss"], ax=ax[1, 0])
    seaborn.lineplot(x=epochs, y=history["validation_accuracy"], ax=ax[0, 1])
    seaborn.lineplot(x=epochs, y=history["validation_loss"], ax=ax[1, 1])
    ax[0, 0].set_title("Train Accuracy", fontsize=15)
    ax[0, 1].set_title("Validation Accuracy", fontsize=15)
    ax[1, 0].set_title("Train Loss", fontsize=15)
    ax[1, 1].set_title("Validation Loss", fontsize=15)
    ax[1, 0].set_xlabel("Epochs", fontsize=15)
    ax[1, 1].set_xlabel("Epochs", fontsize=15)

    ax[0, 0].tick_params(axis='both', which='both', labelsize=13)
    ax[0, 1].tick_params(axis='both', which='both', labelsize=13)
    ax[1, 0].tick_params(axis='both', which='both', labelsize=13)
    ax[1, 1].tick_params(axis='both', which='both', labelsize=13)
    fig.suptitle("Training and Validation Performance", fontsize=17)

    if savefig:
        plt.savefig(f"reports/figures_for_report/train_validation_nn_performance")

    plt.show()


def train_model(lr=0.01,
                epochs=10,
                validation_size=0.1,
                batch_size=32):

    data = FashionMnistData()

    X_train, y_train, _, _ = data.get_train_test_split()

    our_model = NeuralNetworkClassifier(layers=[DenseLayer(128, "relu"),
                                                DenseLayer(64, "relu"),
                                                DenseLayer(5, "softmax")],
                                        learning_rate=lr,
                                        epochs=epochs,
                                        random_state=42)

    history = our_model.fit(X_train, y_train, batch_size=batch_size, validation_size=validation_size)

    return history


if __name__ == "__main__":
    history_ = train_model(epochs=200)
    plot_performance_after_epochs(history_)
