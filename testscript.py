from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics

import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, isdir, join
import numpy
import re
import sys
import getopt
import codecs
import time

def read_bagofwords_dat(myfile, numofemails=10000):
    bagofwords = numpy.fromfile(myfile, dtype=numpy.uint8, count=-1, sep="")
    bagofwords = numpy.reshape(bagofwords,(numofemails,-1))
    return bagofwords
print('something')
#def main(argv):
pathTrain = "./trec07p_data/Train/"
dataTrain = read_bagofwords_dat(pathTrain+"train_emails_bag_of_words_200.dat", numofemails=45000)
print('something')
target = []
for i in range(22500):
    target.append(0)
for i in range(22500):
    target.append(1)
#target[0:22499] = 0
#target[25000:44999] = 1
print(target[0], target[22499], target[22500], target[44999])
pathTest = "./trec07p_data/Test/"
dataTest = read_bagofwords_dat(pathTest+"test_emails_bag_of_words_0.dat", numofemails=5000)
testTarget = []
for i in range(2500):
    testTarget.append(0)
for i in range(2500):
    testTarget.append(1)
clf = MultinomialNB(alpha = 1.0)
clf.fit(dataTrain, target)
print(clf.score(dataTest, testTarget))
clf2 = MultinomialNB(alpha = 1.0)
clf2.fit(dataTrain, target)
print(clf2.score(dataTest, testTarget))