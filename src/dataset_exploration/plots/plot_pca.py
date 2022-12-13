import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import numpy as np

from src.dataset_exploration.pca.pca import get_n_pca
from src.map_cls_to_clothing import map_cls_to_clothing
from colors import *

color_palette = seaborn.color_palette("Paired")


def plot_pc1_pca2(fig_name="pca_plot"):
    X_pca, y_train = get_n_pca(n=2, normalize=True)
    pca1 = X_pca[:, 0]
    pca2 = X_pca[:, 1]
    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
    seaborn.set_style("darkgrid")
    seaborn.set(font="Futura")
    y_train = _format_y_train(y_train)
    seaborn.scatterplot(x=pca1, y=pca2,
                        ax=ax, alpha=0.9, hue=y_train,
                        palette={"T-shirt": t_shirt_color, "Pants": pants_color,
                                 "Sweatshirt": sweatshirt_color, "Dress": dress_color,
                                 "Shirt": shirt_color},
                        hue_order=["T-shirt", "Pants", "Sweatshirt", "Dress", "Shirt"],
                        legend=True)

    plt.legend(loc="upper right", fancybox=False)

    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()
    fig.savefig(f"reports/figures_for_report/{fig_name}")


def _format_y_train(y_train):
    mapping = map_cls_to_clothing()
    y_train = pd.Series(y_train)
    y_train = y_train.replace(mapping).tolist()
    return y_train


def _get_data_for_class(data, cls):
    return np.where(data == cls)


if __name__ == "__main__":
    plot_pc1_pca2()
