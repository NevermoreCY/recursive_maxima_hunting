from sklearn.decomposition import PCA


def getPCA_realData(X_train, X_test, y_train):
    pca = PCA(n_components=0.92)
    pca.fit(X_train.data_matrix.reshape(X_train.data_matrix.shape[:-1]), y_train)
    X_train = pca.transform(X_train.data_matrix.reshape(X_train.data_matrix.shape[:-1]))
    X_test = pca.transform(X_test.data_matrix.reshape(X_test.data_matrix.shape[:-1]))
    return X_train, X_test


def getPCA_simuData(X_train, X_test, y_train):
    pca = PCA(n_components=0.95, svd_solver='full')
    pca.fit(X_train, y_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test
