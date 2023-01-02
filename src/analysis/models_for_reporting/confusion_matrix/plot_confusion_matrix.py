from sklearn.metrics import confusion_matrix
import seaborn
import matplotlib.pyplot as plt
import pandas as pd

# from src.analysis import best_decision_tree
from src.analysis import best_neural_network
from src.analysis import best_random_forest
from src.get_data import FashionMnistData
from src.colors.colors import map_cls_to_clothing


def main(data, label):
    font_size = 22
    seaborn.set(font_scale=2.4)
    data = pd.DataFrame(data, index=[i for i in map_cls_to_clothing().values()],
                        columns=[i for i in map_cls_to_clothing().values()])
    fig, ax = plt.subplots(figsize=(14, 8), tight_layout=True)
    seaborn.heatmap(data, annot=True, cmap='Blues', fmt='g', ax=ax, cbar=False)
    ax.set_title(label, fontdict=dict(weight='bold'), fontsize=font_size + 8)
    ax.tick_params(axis='both', labelsize=font_size + 5)
    ax.xaxis.set_label_coords(.5, -.1)
    ax.yaxis.set_label_coords(-.1, .5)
    plt.savefig(f"reports/figures_for_report/{label}")


def get_neural_network_confusion_matrix():
    data = FashionMnistData()
    x_train, y_train, x_test, y_test = data.get_train_test_split()
    model = best_neural_network()
    y_preds = model.predict(x_test)
    return confusion_matrix(y_true=y_test, y_pred=y_preds)


def get_decision_tree_confusion_matrix():
    data = FashionMnistData()
    x_train, y_train, x_test, y_test = data.get_train_test_split()
    model = best_decision_tree()
    y_preds = model.predict(x_test)
    return confusion_matrix(y_true=y_test, y_pred=y_preds)


def get_random_forest_confusion_matrix():
    data = FashionMnistData()
    x_train, y_train, x_test, y_test = data.get_train_test_split()
    model = best_random_forest()
    y_preds = model.predict(x_test)
    return confusion_matrix(y_true=y_test, y_pred=y_preds)


if __name__ == "__main__":
    nn = get_neural_network_confusion_matrix()
    dt = get_decision_tree_confusion_matrix()
    rf = get_random_forest_confusion_matrix()
    main(nn, "Neural Network Confusion Matrix")
    main(dt, "Decision Tree Confusion Matrix")
    main(rf, "Random Forest Confusion Matrix")
    plt.show()
