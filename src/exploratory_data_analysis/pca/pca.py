from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.get_data.fashion_mnist_data import FashionMnistData


def get_n_pca(n=3):
    data = FashionMnistData(normalize=True)
    X_train, y_train, _, _ = data.get_train_test_split()

    X_train = StandardScaler().fit_transform(X_train)

    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X_train)

    print(pca.explained_variance_ratio_)

    return X_pca, y_train




