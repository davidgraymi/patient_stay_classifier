import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

"""
Author: David Gray
Description: Tools utilized by the training and testing programs.
"""

def build_mlp(X_train, y_train):
    # Build and train MLP
    print("------training MLP")
    clf = MLPClassifier(hidden_layer_sizes=(64,64), activation='relu').fit(X_train, np.ravel(y_train))
    return clf

def build_knn(X_train, y_train):
    # Build and train KNN
    print("------training KNN")
    clf = KNeighborsClassifier(n_neighbors=10, weights='uniform', n_jobs=-1).fit(X_train, np.ravel(y_train))
    return clf

def save_model(fp, clf, model):
    # Save classifier
    print("------saving classifier")
    pickle_file = open(fp + "/models/" + model + ".pkl", 'wb')
    pickle.dump(clf, pickle_file)
    pickle_file.close()

def load_model(fp, model):
    # Load classifier
    print("------loading classifier")
    pickle_file = open(fp + "/models/" + model + ".pkl", 'rb')
    clf = pickle.load(pickle_file)
    pickle_file.close()
    return clf

def get_train_data(fp):
    # Get training data
    print("------getting training data")
    X_train = pd.read_csv(fp + "/processed_data/X_train.csv", header=None)
    y_train = pd.read_csv(fp + "/processed_data/y_train.csv", header=None)
    return X_train, y_train

def get_test_data(fp):
    # Get testing data
    print("------getting testing data")
    X_test = pd.read_csv(fp + "/processed_data/X_test.csv", header=None)
    y_test = pd.read_csv(fp + "/processed_data/y_test.csv", header=None)
    return X_test, y_test

def get_results(clf, X_test, y_test):
    print("------getting confusion matrix")
    # Get Confusion matrix results
    class_names = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '>100']
    title = "Confusion Matrix"
    disp = plot_confusion_matrix(clf, X_test, y_test,
                                    display_labels=class_names,
                                    cmap=plt.cm.Blues,
                                    normalize=None)
    disp.ax_.set_title(title)
    plt.show()
    plt.close()

    # Get classification report
    predict_test = clf.predict(X_test)
    print(classification_report(y_test,predict_test))

def train_mlp(fp):
    X_train, y_train = get_train_data(fp)
    clf = build_mlp(X_train, y_train)
    save_model(fp, clf, 'mlp')

def train_knn(fp):
    X_train, y_train = get_train_data(fp)
    clf = build_knn(X_train, y_train)
    save_model(fp, clf, 'knn')

def test_mlp(fp):
    X_test, y_test = get_test_data(fp)
    clf = load_model(fp, 'mlp')
    get_results(clf, X_test, y_test)

def test_knn(fp):
    X_test, y_test = get_test_data(fp)
    clf = load_model(fp, 'knn')
    get_results(clf, X_test, y_test)
