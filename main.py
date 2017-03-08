# Load libraries
import pandas
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# Load dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
names = ['id', 'refractive-index', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium', 'Barium', 'Iron', 'class']
dataset = pandas.read_csv(url, names=names)


# Split-out validation dataset
array = dataset.values
X = array[:, 1:9]
print('X : ' + str(X))
Y = array[:, 10]
print('Y : ' + str(Y))
validation_size = 0.10
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# Spot Check Algorithms
models = []
models.append(('LogisticRegression', LogisticRegression()))
models.append(('KNeighborsClassifier', KNeighborsClassifier(3)))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
models.append(('SVM', SVC()))

# evaluate each model in turn
for name, model in models:
    kfold = model_selection.KFold(n_splits=2, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    msg = "%s: %0.2f (%0.2f)" % (name, cv_results.mean() * 100, cv_results.std() * 100)
    print(msg)
