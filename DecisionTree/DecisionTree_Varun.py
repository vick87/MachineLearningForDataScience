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
os.chdir("C:\\Users\\lenovo\\Desktop\\MachineLearningWithDataAnalysis\\DecisionTree")

#Reading the input
TreeData = pd.read_csv('tree_addhealth.csv')
TreeData.shape[0]
"""
To drop all the NAs from the dataset. 
Updated dataset size of 4575 records out of 6504 records.
"""
CleanTreeData = TreeData.dropna()
CleanTreeData.shape[0]

PredictDataSet = CleanTreeData[['BIO_SEX','HISPANIC','WHITE','BLACK','NAMERICAN','ASIAN',
'age','ALCEVR1','ALCPROBS1','marever1','cocever1','inhever1','cigavail','DEP1',
'ESTEEM1','VIOL1','PASSIST','DEVIANT1','SCHCONN1','GPA1','EXPEL1','FAMCONCT','PARACTV',
'PARPRES']]

ResponseVariable = CleanTreeData['TREG1']

"""Splitting the trainng and test dataset. 
Test DS = 40% of total and remaining as the Training DS.
"""
prediction_train, prediction_test,target_train,target_test = train_test_split(PredictDataSet,ResponseVariable, test_size=.4)

#Build model on training data
Model_DecisionTree = DecisionTreeClassifier()
Model_DecisionTree = Model_DecisionTree.fit(prediction_train,target_train)

ActualPredictions = Model_DecisionTree.predict(prediction_test)

#sklearn.metrics.confusion_matrix(target_test,ActualPredictions)
sklearn.metrics.accuracy_score(target_test, ActualPredictions)

sklearn.metrics.confusion_matrix(target_test,ActualPredictions)

#Displaying the decision tree
from sklearn import tree
#From StringIO import StringIO
from io import StringIO
#From StringIO import StringIO 
from IPython.display import Image
out = StringIO()
tree.export_graphviz(Model_DecisionTree, out_file=out)
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())