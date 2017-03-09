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
import numpy as np
import pickle
import os.path


def open_dataset():
    # Load dataset
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
    names = [
        'id',
        'refractive-index',
        'Sodium',
        'Magnesium',
        'Aluminum',
        'Silicon',
        'Potassium',
        'Calcium',
        'Barium',
        'Iron',
        'class'
    ]
    return pandas.read_csv(url, names=names)


def split_data(dataset):
    # seperate the elements and type of glass
    array = dataset.values
    X = array[:, 1:10]
    Y = array[:, 10]
    return X, Y


def prepare_training(models):
    dataset = open_dataset()
    validation_size = 0.50
    seed = 7
    X, Y = split_data(dataset)
    # splits the dataset into a training set and a test set
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        X,
        Y,
        test_size=validation_size,
        random_state=seed
    )
    train_models(models, X_train, Y_train, X_test, Y_test)


def train_models(models, X_train, Y_train, X_test, Y_test):
    classifiers = []
    # iterates through the models
    for name, model in models:
        # chooses the index's for test and training set
        kfold = model_selection.KFold(n_splits=4)
        for traincv, testcv in kfold.split(X_train):
            # trains the models
            model.fit(X_train[traincv], Y_train[traincv])
            # tests the models, doesn't output the result
            model.predict(X_train[testcv])
        # final test on the original test set
        prediction = model.predict(X_test)
        print(name, accuracy_score(prediction, Y_test) * 100)
        with open(
            'pickle/' + name + '_classifier.pickle',
            'wb'
        ) as ph:
            pickle.dump(model, ph)
        classifiers.append((name, model))
    return classifiers


glass_types = {
    '1.0': 'Building Windows Float Processed',
    '2.0': 'Building Windows Non Float Processed',
    '3.0': 'Vehicle Windows Float Processed',
    '4.0': 'Vehicle Windows Non Float Processed',
    '5.0': 'Containers',
    '6.0': 'Tableware',
    '7.0': 'Headlamps'
}


classifiers = [
    'LogisticRegression',
    'KNeighborsClassifier',
    'DecisionTreeClassifier',
    'SVM'
]

models = []
# checks whether the classifiers are already created
# if there not, creates and tests them
if os.path.isfile('pickle/SVM_classifier.pickle'):
    for name in classifiers:
        with open('pickle/SVM_classifier.pickle', 'rb') as ph:
            models.append((name, pickle.load(ph)))
else:
    models.append(('LogisticRegression', LogisticRegression()))
    models.append(('KNeighborsClassifier', KNeighborsClassifier(3)))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
    models.append(('SVM', SVC()))
    models = prepare_training(models)

# inputs new data to test
ri = float(input("Enter Refractive Index: "))
na = float(input("Enter Sodium: "))
mg = float(input("Enter Magnesium: "))
al = float(input("Enter Aluminum: "))
si = float(input("Enter Silicon: "))
k = float(input("Enter Potassium: "))
ca = float(input("Enter Calcium: "))
ba = float(input("Enter Barium: "))
fe = float(input("Enter Iron: "))

# tests new data using the SVM classifier
new_data = np.array([ri, na, mg, al, si, k, ca, ba, fe])
prediction = models[3][1].predict(new_data.reshape(1, -1))

# outputs the type of glass
print('The type of glass is', glass_types[str(prediction[0])])
