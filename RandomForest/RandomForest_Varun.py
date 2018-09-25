from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
 # Feature Importance
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

#Set up the desired working tree here. Pls use a \ to escape characters here.
os.chdir("C:\\Users\\v-vamer\\Desktop\\ML Learning\\MachineLearningForDataAnalysis\\MachineLearningForDataScience\\RandomForest")
#os.chdir("C:\\Users\\lenovo\\Desktop\\MachineLearningWithDataAnalysis\\DecisionTree")

#Get the input dataset and clean it. Remove all NAs.
TreeData = pd.read_csv("tree_addhealth.csv")
CleanTreeData = TreeData.dropna()

CleanTreeData.dtypes
CleanTreeData.describe()

#Get the Predictors
PredictorVariables = CleanTreeData[['BIO_SEX','HISPANIC','WHITE','BLACK','NAMERICAN','ASIAN','age',
'ALCEVR1','ALCPROBS1','marever1','cocever1','inhever1','cigavail','DEP1','ESTEEM1','VIOL1',
'PASSIST','DEVIANT1','SCHCONN1','GPA1','EXPEL1','FAMCONCT','PARACTV','PARPRES']]

#This is the target variable.
TargetVariable = CleanTreeData.TREG1

#Split into training and testing sets
pred_train, pred_test, tar_train, tar_test  = train_test_split(PredictorVariables, TargetVariable, test_size=.4)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data
classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

"""
The confusion matrix and the accuracy score
This accuracy score will basically serve kind of like a benchmark.
If this score is closer to the score generated after implementing multiple trees 
then we can say that a single Decision Tree would suffice this dataset.
"""
sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)


# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(pred_train,tar_train)
"""
This importance tells is which is the most important predictor variable of the list,
that helps us predict the TargetVariable.
In our case the most important is 'marever1' and least imp is 'ASIAN'.
"""
print(model.feature_importances_)
"""
Running a different number of trees and see the effect
 of that on the accuracy of the prediction
"""
trees=range(25)
accuracy=np.zeros(25)

for idx in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=idx + 1)
   classifier=classifier.fit(pred_train,tar_train)
   predictions=classifier.predict(pred_test)
   accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)
   
plt.cla()
plt.plot(trees, accuracy)