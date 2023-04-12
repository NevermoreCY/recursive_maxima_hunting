import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
import skfda
from skfda.ml.classification import KNeighborsClassifier


# load and plot data
X, y = skfda.datasets.fetch_growth(return_X_y=True, as_frame=True)
X = X.iloc[:, 0].values
y = y.values

# Plot samples grouped by sex
X.plot(group=y.codes, group_names=y.categories)

y = y.codes


# Get dataset for KNN

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=(1/3), # paper use 1/3 for test set
    stratify=y,
    random_state=0,
)

train_data_size = X_train.data_matrix.shape[0]
# # Only odd numbers, to prevent ties
param_grid = {"n_neighbors": range(1, int(np.sqrt(train_data_size)), 2)}
#
#
knn = KNeighborsClassifier()
#
# Perform grid search with cross-validation
gscv = GridSearchCV(knn, param_grid, cv=10)
gscv.fit(X_train, y_train)
#
#
print("Best params:", gscv.best_params_)
print("Best cross-validation score:", gscv.best_score_) # note this score is still on training set

# print test set score
score = gscv.score(X_test, y_test)
print("test set score :", score)
