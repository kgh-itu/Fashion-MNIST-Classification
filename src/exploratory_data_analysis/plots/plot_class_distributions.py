import matplotlib.pyplot as plt
import numpy as np
import seaborn

from src.get_train_test_split.fashion_mnist_data import FashionMnistData
from src.colors import *


def plot_class_distribution():
    data = FashionMnistData()
    _, y_train, _, y_test = data.get_train_test_split()
    x = ["T-shirt", "Pants", "Sweatshirt", "Dress", "Shirt"]
    train_count = np.bincount(y_train)
    test_count = np.bincount(y_test)
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5), tight_layout=True)
    seaborn.set_style("dark")
    seaborn.set(font="Futura")

    seaborn.barplot(y=train_count, x=x,
                    palette={"T-shirt": t_shirt_color, "Pants": pants_color,
                             "Sweatshirt": sweatshirt_color, "Dress": dress_color,
                             "Shirt": shirt_color},
                    hue_order=["T-shirt", "Pants", "Sweatshirt", "Dress", "Shirt"],
                    ax=ax[0])
    ax[0].set_title("Training Data")

    seaborn.barplot(y=test_count, x=x,
                    palette={"T-shirt": t_shirt_color, "Pants": pants_color,
                             "Sweatshirt": sweatshirt_color, "Dress": dress_color,
                             "Shirt": shirt_color},
                    hue_order=["T-shirt", "Pants", "Sweatshirt", "Dress", "Shirt"],
                    ax=ax[1])

    ax[1].set_title("Test Data")
    fig_name = "class_distribution"
    plt.savefig(f"reports/figures_for_report/{fig_name}", bbox_inches='tight', pad_inches=0.2)


if __name__ == "__main__":
    plot_class_distribution()
    plt.show()
