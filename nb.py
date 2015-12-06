import json 
import numpy as np
from pprint import pprint
import os
import logging
from optparse import OptionParser
import sys
from time import time
#import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn.linear_model import LogisticRegression
from sklearn import grid_search
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation
from sklearn.cross_validation import KFold

#path='/Box\ Sync/Kaggle/Yummly/'
#os.chdir(path)

with open('train.json') as data_file:
  data=json.load(data_file)

#input_dir = ''#'../input/'

#with open(os.path.join(input_dir, 'train.json')) as train_f:
#    data = json.loads(train_f.read())

n=len(data)
nsplit=n/2


def normalize_input(X):
    return (X.T / np.sum(X, axis=1)).T

X=[x['ingredients'] for x in data]
X= [dict(zip(x,np.ones(len(x)))) for x in X]

vec = DictVectorizer()

X= vec.fit_transform(X).toarray()
X= normalize_input(X)
X = X.astype(np.float32)

feature_names = np.array(vec.feature_names_)

lbl = LabelEncoder()

y= [y['cuisine'] for y in data]
y= lbl.fit_transform(y).astype(np.int32)

label_names = lbl.classes_ 
#for i, l in enumerate(label_names):
    #print('i: {}, l: {}'.format(i, l))

X_train=X[0:nsplit]
y_train=y[0:nsplit]
X_test=X[nsplit:]
y_test=y[nsplit:]

#kf=KFold(len(data),n_folds=10,shuffle=True,random_state=1)


def benchmark(clf,params=None):
    if params is not None:
		clf=grid_search.GridSearchCV(clf, params)

    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        print()

    if True:
        print("classification report:")
        print(metrics.classification_report(y_test, pred))

    if True:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
results.append(benchmark(MultinomialNB(alpha=.01))) #.604
results.append(benchmark(BernoulliNB(alpha=.01))) #.731
BNBparams={'alpha':[.001,.01,.1,1]}
results.append(benchmark(BernoulliNB(),BNBparams)) #.739 -> .750 kfold
results.append(benchmark(MultinomialNB(),BNBparams)) #.604
results.append(benchmark(LinearSVC())) #.741
SVCparams={'C':[.01,.1,1,10]}
results.append(benchmark(LinearSVC(),SVCparams)) #.772
results.append(benchmark(LogisticRegression(),SVCparams)) #.729 vs .595 no opt
