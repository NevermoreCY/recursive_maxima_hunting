import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from skfda.ml.classification import KNeighborsClassifier
from tqdm import tqdm
import random
import data_preprocessing
import Recursive_Maxima_Hunting


# load the data
X, y = data_preprocessing.get_growth()
model_name= "RMH"
data_set = "growth"

# we will split the dataset repeatly for 200 times
repetition = 200
error = []
dimension = []

for i in tqdm(range(repetition)):
    rand_int = random.randint(0, 9999999)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=(1 / 3),  # paper use 1/3 for test set
        stratify=y,
        random_state=rand_int,
    )


    # Load different dimension reduction algorithm here
    # rmh = Recursive_Maxima_Hunting.Recursive_maxima_hunting()
    # rmh_model = rmh.fit(X_train, y_train)
    # X_train = rmh.transform(X_train)
    # X_test = rmh.transform(X_test)


    dim = X_train.shape[1]
    dimension.append(dim)

    train_data_size = X_train.shape[0]
    # # Only odd numbers, to prevent ties
    param_grid = {"n_neighbors": range(1, int(np.sqrt(train_data_size)), 1)}
    #
    #
    knn = KNeighborsClassifier()
    #
    # Perform grid search with cross-validation
    gscv = GridSearchCV(knn, param_grid, cv=10)
    gscv.fit(X_train, y_train)

    score = gscv.score(X_test, y_test)
    error.append(1 - score)

avg_test_error = np.average(error)
avg_dimen = np.average(dimension)
print("average test set error :", avg_test_error )
print("average dimension  :", avg_dimen )

import pickle
# save the results
out_path = model_name + "_" + data_set + "_result.pkl"
out_result = {}
out_result['error'] = error
out_result['dimension'] = dimension
with open(out_path, 'wb') as f:
    pickle.dump(out_result, f)

