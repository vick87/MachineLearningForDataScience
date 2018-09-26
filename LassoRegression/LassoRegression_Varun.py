import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

TreeData = pd.read_csv('tree_addhealth.csv')
TreeData.columns = map(str.upper,TreeData.columns)

#removing the NAs
TreeDataClean = TreeData.dropna()

#We need to recode 1 for Male and 0 for Female
GenderRecode={1:1,2:0}
TreeDataClean['MALE']= TreeDataClean['BIO_SEX'].map(GenderRecode)

#Load the Predictor variables
#select predictor variables and target variable as separate data sets  
PredictorVariables= TreeDataClean[['MALE','HISPANIC','WHITE','BLACK','NAMERICAN','ASIAN',
'AGE','ALCEVR1','ALCPROBS1','MAREVER1','COCEVER1','INHEVER1','CIGAVAIL','DEP1',
'ESTEEM1','VIOL1','PASSIST','DEVIANT1','GPA1','EXPEL1','FAMCONCT','PARACTV',
'PARPRES']]

# SCHCONN1 = School Connectedness Response Variable
TargetVariable = TreeDataClean.SCHCONN1

FinalPredictors = PredictorVariables.copy()
FinalPredictors['MALE'] = preprocessing.scale(FinalPredictors['MALE'].astype('float64'))
FinalPredictors['HISPANIC'] = preprocessing.scale(FinalPredictors['HISPANIC'].astype('float64'))
FinalPredictors['WHITE'] = preprocessing.scale(FinalPredictors['WHITE'].astype('float64'))
FinalPredictors['BLACK'] = preprocessing.scale(FinalPredictors['BLACK'].astype('float64'))
FinalPredictors['NAMERICAN'] = preprocessing.scale(FinalPredictors['NAMERICAN'].astype('float64'))
FinalPredictors['ASIAN'] = preprocessing.scale(FinalPredictors['ASIAN'].astype('float64'))
FinalPredictors['AGE'] = preprocessing.scale(FinalPredictors['AGE'].astype('float64'))
FinalPredictors['ALCEVR1'] = preprocessing.scale(FinalPredictors['ALCEVR1'].astype('float64'))
FinalPredictors['ALCPROBS1'] = preprocessing.scale(FinalPredictors['ALCPROBS1'].astype('float64'))
FinalPredictors['MAREVER1'] = preprocessing.scale(FinalPredictors['MAREVER1'].astype('float64'))
FinalPredictors['COCEVER1'] = preprocessing.scale(FinalPredictors['COCEVER1'].astype('float64'))
FinalPredictors['INHEVER1'] = preprocessing.scale(FinalPredictors['INHEVER1'].astype('float64'))
FinalPredictors['CIGAVAIL'] = preprocessing.scale(FinalPredictors['CIGAVAIL'].astype('float64'))
FinalPredictors['DEP1'] = preprocessing.scale(FinalPredictors['DEP1'].astype('float64'))
FinalPredictors['ESTEEM1'] = preprocessing.scale(FinalPredictors['ESTEEM1'].astype('float64'))
FinalPredictors['VIOL1'] = preprocessing.scale(FinalPredictors['VIOL1'].astype('float64'))
FinalPredictors['PASSIST'] = preprocessing.scale(FinalPredictors['PASSIST'].astype('float64'))
FinalPredictors['DEVIANT1'] = preprocessing.scale(FinalPredictors['DEVIANT1'].astype('float64'))
FinalPredictors['GPA1'] = preprocessing.scale(FinalPredictors['GPA1'].astype('float64'))
FinalPredictors['EXPEL1'] = preprocessing.scale(FinalPredictors['EXPEL1'].astype('float64'))
FinalPredictors['FAMCONCT'] = preprocessing.scale(FinalPredictors['FAMCONCT'].astype('float64'))
FinalPredictors['PARACTV'] = preprocessing.scale(FinalPredictors['PARACTV'].astype('float64'))
FinalPredictors['PARPRES'] = preprocessing.scale(FinalPredictors['PARPRES'].astype('float64'))

#Splitting the data into test and training data
pred_train,pred_test,tar_train,tar_test = train_test_split(FinalPredictors,TargetVariable,test_size=.3,random_state=123)

#Cretaing the Model
LassoRegModel = LassoLarsCV(cv=10,precompute=False).fit(pred_train,tar_train)

# print variable names and regression coefficients
dict(zip(FinalPredictors.columns, LassoRegModel.coef_))

# plot coefficient progression
m_log_alphas = -np.log10(LassoRegModel.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, LassoRegModel.coef_path_.T)
plt.axvline(-np.log10(LassoRegModel.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')

# Plot mean square error for each fold
m_log_alphascv = -np.log10(LassoRegModel.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, LassoRegModel.cv_mse_path_, ':')
plt.plot(m_log_alphascv, LassoRegModel.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(LassoRegModel.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')

# MSE from training and test data
train_error = mean_squared_error(tar_train, LassoRegModel.predict(pred_train))
test_error = mean_squared_error(tar_test, LassoRegModel.predict(pred_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

# R-square from training and test data
rsquared_train=LassoRegModel.score(pred_train,tar_train)
rsquared_test=LassoRegModel.score(pred_test,tar_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)