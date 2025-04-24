from main import X_train_scaled, y_train
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

gridSearchParameters = {
    'C': [2 ** (-5), 2 ** (-3), 2 ** (-1), 2 ** 1, 2 ** 3, 2 ** 5, 2 ** 7],
    'gamma': [2 ** (-15), 2 ** (-13), 2 ** (-11), 2 ** (-9), 2 ** (-7), 2 ** (-5),
                2 ** (-3), 2 ** (-1), 2 ** 1, 2 ** 3],
    'epsilon': [2 ** (-9), 2 ** (-7),  2 ** (-5), 2 ** (-3), 2 ** (-1), 2 ** 1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVR(), gridSearchParameters, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=3)
# Usually the scoring metric is r2, but I noticed that for this particular dataset the MSE has values suited for metric use
grid_search.fit(X_train_scaled, y_train)

print(grid_search.best_params_)

grid_searchDataframe = pd.DataFrame(grid_search.cv_results_)
grid_searchDataframe.to_excel("../datasets/results.xlsx")