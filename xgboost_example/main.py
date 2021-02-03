from sklearn import datasets
import xgboost as xgb
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import confusion_matrix  # multi-class confusion matrix içeren bir library
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
X = iris.data
y = iris.target

# accuracy genelde 1.0 çıktığı için test_size = 0.85 ayarlandi.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.85, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

estimator = xgb.XGBClassifier(
    objective='multi:softprob',
    use_label_encoder=False,
    verbosity=0,
    num_class=3,  # Iris verisi için 3 çıkış.
    nthread=4,
)

params = {
    'eta': [0.05, 0.1, 0.3],
    'max_depth': [6, 9, 12],
}

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=params,
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_index_

estimator = xgb.XGBClassifier(best_params, objective='multi:softprob', verbosity=0, use_label_encoder=False,
                              num_class=3,  # Iris verisi için 3 çıkış.
                              nthread=4, )
estimator.fit(X_train, y_train)

cm = confusion_matrix(y_target=y_test, y_predicted=estimator.predict(X_test), binary=False)
print('The Confusion Matrix is:\n', cm)

predict_accuracy_on_test_set = (cm[0, 0] + cm[1, 1] + cm[2, 2]) / \
                               (cm[0, 0] + cm[0, 1] + cm[0, 2] + cm[1, 0] + cm[1, 1] + cm[1, 2] + cm[2, 0] + cm[2, 1] +
                                cm[2, 2])
print('The Accuracy on Test Set is: ', predict_accuracy_on_test_set)
