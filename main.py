import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
import time

def main ():
    DataSetSelection = input("Please Enter the name of DataSet (digits or ard): ")
    ClassifierSelection = input("Please Enter the name of Classification (perceptron or dt or knn or lr or svm-linear or svm-non-linear: ")
    start_time = time.time()
    if (DataSetSelection == "digits"):
        digits = load_digits()
        X = digits.data
        y = digits.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        if (ClassifierSelection == "perceptron"):
            per = Perceptron(n_iter=40, eta0=.1, random_state=1)
            per.fit(X_train, y_train)
            y_pred = per.predict(X_test)
            accuracy = ((y_test==y_pred).sum()/len(y_test)*100)
            print('accuracy %.2f' %accuracy)
        elif (ClassifierSelection == "dt"):
            clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=10, min_samples_leaf=5)
            clf_entropy.fit(X_train, y_train)
            y_pred = clf_entropy.predict(X_test)
            accuracy = ((y_test==y_pred).sum()/len(y_test)*100)
            print('accuracy %.2f' % accuracy)
        elif (ClassifierSelection == "knn"):
            knnn = KNeighborsClassifier(n_neighbors=23, metric='euclidean')#how did i choose n_neighbors = math.sqrt(len(X_test))
            knnn.fit(X_train, y_train)
            y_pred = knnn.predict(X_test)
            accuracy = ((y_test==y_pred).sum()/len(y_test)*100)
            print('accuracy %.2f' % accuracy)
        elif (ClassifierSelection == "lr"):
            logreg = LogisticRegression(multi_class='auto')
            logreg.fit(X_train, y_train)
            y_pred = logreg.predict(X_test)
            #y_pred_lr_prob = logreg.predict_log_proba(X_test)
            #print(y_pred_lr_prob.shape)
            #print(y_pred_lr_prob)
            accuracy = ((y_test==y_pred).sum()/len(y_test)*100)
            print('accuracy %.2f' % accuracy)
        elif (ClassifierSelection == "svm-linear"):
            clf = svm.SVC(kernel="linear", random_state=1, C=1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = ((y_test==y_pred).sum()/len(y_test)*100)
            print('accuracy %.2f' % accuracy)
        elif (ClassifierSelection == "svm-non-linear"):
            #clf = svm.SVC(kernel='rbf', random_state=1, gamma='auto', C=20.0)
            clf = svm.SVC(gamma='scale', C = 1.0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = ((y_test==y_pred).sum()/len(y_test)*100)
            print('accuracy %.2f' % accuracy)
        else:
            print("input right classifier")
    elif(DataSetSelection == "ard"):
        dataset = pd.read_csv('D:\ML_project3\subject1_ideal.csv', header=0)
        X = dataset.iloc[:, 0:118].values
        y = dataset.iloc[:, 119].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        if (ClassifierSelection == "perceptron"):
            per = Perceptron(n_iter=40, eta0=.1, random_state=1)
            per.fit(X_train, y_train)
            y_pred = per.predict(X_test)
            accuracy = ((y_test==y_pred).sum()/len(y_test)*100)
            print('accuracy %.2f' %accuracy)
        elif (ClassifierSelection == "dt"):
            clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=10, min_samples_leaf=5)
            clf_entropy.fit(X_train, y_train)
            y_pred = clf_entropy.predict(X_test)
            accuracy = ((y_test==y_pred).sum()/len(y_test)*100)
            print('accuracy %.2f' % accuracy)
        elif (ClassifierSelection == "knn"):
            knnn = KNeighborsClassifier(n_neighbors=231, metric='euclidean')
            knnn.fit(X_train, y_train)
            y_pred = knnn.predict(X_test)
            accuracy = ((y_test==y_pred).sum()/len(y_test)*100)
            print('accuracy %.2f' % accuracy)
        elif (ClassifierSelection == "lr"):
            logreg = LogisticRegression(multi_class='auto')
            logreg.fit(X_train, y_train)
            y_pred = logreg.predict(X_test)
            accuracy = ((y_test==y_pred).sum()/len(y_test)*100)
            print('accuracy %.2f' % accuracy)
        elif (ClassifierSelection == "svm-linear"):
            clf = svm.SVC(kernel="linear", random_state=1, C=10.0)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            accuracy = ((y_test == y_pred).sum() / len(y_test) * 100)
            print('accuracy %.2f' % accuracy)
        elif (ClassifierSelection == "svm-non-linear"):
            clf = svm.SVC(gamma='scale', C = 10.0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = ((y_test == y_pred).sum() / len(y_test) * 100)
            print('accuracy %.2f' % accuracy)
        else:
            print("input right classifier")
    else:
        print("please enter right parameters")
    print("Running time of classifier = %s seconds " % (time.time() - start_time))
main()