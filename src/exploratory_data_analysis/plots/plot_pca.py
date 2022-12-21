import matplotlib.pyplot as plt
import pandas as pd
import seaborn

from src.exploratory_data_analysis.pca.pca import get_n_pca
from src.class_clothing_map import map_cls_to_clothing
from src.colors import *


def plot_pca(fig_name):
    X_pca, y_train = get_n_pca(normalize=True)
    pca1 = X_pca[:, 0]
    pca2 = X_pca[:, 1]
    pca3 = X_pca[:, 2]
    y_train = _format_y_train(y_train)

    PALETTE = {"T-shirt": t_shirt_color, "Pants": pants_color,
               "Sweatshirt": sweatshirt_color, "Dress": dress_color,
               "Shirt": shirt_color}
    HUE_ORDER = ["T-shirt", "Pants", "Sweatshirt", "Dress", "Shirt"]
    ALPHA = 0.5
    LABEL_SIZE = 25
    LEGEND_SIZE = 19

    fig, ax = plt.subplots(ncols=2, figsize=(22, 8), tight_layout=True)
    seaborn.set_style("dark")
    seaborn.set(font="Futura")

    seaborn.scatterplot(x=pca1, y=pca2,
                        ax=ax[0], alpha=ALPHA, hue=y_train,
                        palette=PALETTE,
                        hue_order=HUE_ORDER,
                        legend=False)
    seaborn.scatterplot(x=pca1, y=pca3,
                        ax=ax[1], alpha=ALPHA, hue=y_train,
                        palette=PALETTE,
                        hue_order=HUE_ORDER,
                        legend=True)

    ax[1].legend(loc="upper right", fancybox=True,
                 fontsize=LEGEND_SIZE, shadow=True)

    ax[0].set_xlabel(f'PCA 1', fontsize=LABEL_SIZE)
    ax[0].set_ylabel(f'PCA 2', fontsize=LABEL_SIZE)
    ax[1].set_xlabel(f'PCA 1', fontsize=LABEL_SIZE)
    ax[1].set_ylabel(f'PCA 3', fontsize=LABEL_SIZE)

    for ax_ in ax:
        ax_.set_xticklabels([])
        ax_.set_yticklabels([])

    plt.show()

    fig.savefig(f"reports/figures_for_report/{fig_name}")

    return fig


def _format_y_train(y_train):
    mapping = map_cls_to_clothing()
    y_train = pd.Series(y_train)
    y_train = y_train.replace(mapping).tolist()
    return y_train


if __name__ == "__main__":
    plot_pca("PCA")
