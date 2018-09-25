from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
import pydotplus

#Set up the desired working tree here. Pls use a \ to escape characters here.
os.chdir("C:\\Users\\v-vamer\\Desktop\\ML Learning\\MachineLearningForDataAnalysis\\MachineLearningForDataScience\\DecisionTree")
#os.chdir("C:\\Users\\lenovo\\Desktop\\MachineLearningWithDataAnalysis\\DecisionTree")

NesarcData = pd.read_csv('nesarc_pds.csv')
NesarcDataNaN = NesarcData.replace(r'\s+', np.NaN,regex=True)
NesarcData_Clean = NesarcData.dropna()

NesarcData.shape[0]
NesarcData_Clean.shape[0]