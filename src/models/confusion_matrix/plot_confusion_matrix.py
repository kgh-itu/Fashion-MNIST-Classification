import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn
import matplotlib.pyplot as plt
import pandas as pd

from src.models.models_for_reporting.best_models.decision_tree import best_decision_tree
from src.models.models_for_reporting.best_models.neural_net import best_neural_network
from src.models.models_for_reporting.best_models.random_forest import best_random_forest
from src.get_train_test_split.fashion_mnist_data import FashionMnistData
from src.class_clothing_map import map_cls_to_clothing


def get_neural_network_confusion_matrix():
    data = FashionMnistData()
    x_train, y_train, x_test, y_test = data.get_train_test_split()
    model = best_neural_network()
    y_preds = model.predict(x_test)
    return confusion_matrix(y_true=y_test, y_pred=y_preds)


if __name__ == "__main__":
    c_matrix = get_neural_network_confusion_matrix()
    df_cm = pd.DataFrame(c_matrix, index=[i for i in map_cls_to_clothing().values()],
                         columns=[i for i in map_cls_to_clothing().values()])
    fig, ax = plt.subplots(figsize=(12, 7))
    seaborn.set(font_scale=1.4)  # for label size
    seaborn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g', ax=ax)
    ax.set_title("Confusion Matrix Neural Network", fontdict=dict(weight='bold'))
    ax.set_ylabel("Predicted Class", fontdict=dict(weight='bold'))
    ax.set_xlabel("Actual Class", fontdict=dict(weight='bold'))
    ax.xaxis.set_label_coords(.5, -.1)
    ax.yaxis.set_label_coords(-.1, .5)
    plt.show()

