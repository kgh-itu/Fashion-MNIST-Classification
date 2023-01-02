import matplotlib.pyplot as plt
import numpy as np
import seaborn

from src.get_data.fashion_mnist_data import FashionMnistData
from src.colors.colors import (t_shirt_color,
                               trousers_color,
                               pullover_color,
                               dress_color,
                               shirt_color)


def plot_class_distribution():
    data = FashionMnistData()
    _, y_train, _, y_test = data.get_train_test_split()
    x = ["T-shirt/Top", "Trousers", "Pullover", "Dress", "Shirt"]
    train_count = np.bincount(y_train)
    test_count = np.bincount(y_test)
    fig, ax = plt.subplots(ncols=2, figsize=(11, 6), tight_layout=True, sharey="all")
    seaborn.set_style("darkgrid")

    seaborn.barplot(y=train_count, x=x,
                    palette={"T-shirt/Top": t_shirt_color, "Trousers": trousers_color,
                             "Pullover": pullover_color, "Dress": dress_color,
                             "Shirt": shirt_color},
                    hue_order=["T-shirt/Top", "Trousers", "Pullover", "Dress", "Shirt"],
                    ax=ax[0])
    ax[0].set_title("Training Data", fontsize=15)

    seaborn.barplot(y=test_count, x=x,
                    palette={"T-shirt/Top": t_shirt_color, "Trousers": trousers_color,
                             "Pullover": pullover_color, "Dress": dress_color,
                             "Shirt": shirt_color},
                    hue_order=["T-shirt/Top", "Trousers", "Pullover", "Dress", "Shirt"],
                    ax=ax[1])

    ax[0].tick_params(axis='both', which='both', labelsize=13)
    ax[1].tick_params(axis='both', which='both', labelsize=13)
    ax[1].set_title("Test Data", fontsize=15)
    fig_name = "class_distribution"
    plt.savefig(f"reports/figures_for_report/{fig_name}", bbox_inches='tight', pad_inches=0.2)


if __name__ == "__main__":
    plot_class_distribution()
    plt.show()
