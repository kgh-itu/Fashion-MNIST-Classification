from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np


from src.get_train_test.prepare_fashion_mnist_dataset import FashionMnistData


def get_n_pca(normalize=True):
    data = FashionMnistData(get_test=False, normalize=normalize)
    X_train, y_train, _, _ = data.get()
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA())])
    X_pca = pipeline.fit_transform(X_train)

    return X_pca, y_train




