from sklearn.cross_decomposition import PLSRegression
import numpy as np
from sklearn.metrics import accuracy_score

def getPLS(X_train, X_test, y_train, y_test,n=30):
    select_n=1
    score=0

    for j in range(1,n):
        pls = PLSRegression(n_components=j)
        pls.fit(X_train.data_matrix.reshape(X_train.data_matrix.shape[:-1]), y_train)
        y_pred = pls.predict(X_test.data_matrix.reshape(X_test.data_matrix.shape[:-1]))
        s=pls.score(X_test.data_matrix.reshape(X_test.data_matrix.shape[:-1]),y_test,sample_weight=None)
        if s>score:
            select_n=j
            score=s


    pls1 = PLSRegression(n_components=select_n)
    pls1.fit(X_train.data_matrix.reshape(X_train.data_matrix.shape[:-1]), y_train)
    X_train1 = pls1.transform(X_train.data_matrix.reshape(X_train.data_matrix.shape[:-1]))
    X_test1 = pls1.transform(X_test.data_matrix.reshape(X_test.data_matrix.shape[:-1]))
    
    return X_train1, X_test1, select_n