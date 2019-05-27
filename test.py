# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:28:38 2018

@author: sourav
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from scipy.stats import mode, zscore
from itertools import product
import matplotlib.pyplot as plt


#reading the data from train.csv and storing as pandas dataframe
input_data = pd.read_csv("testSold.csv",index_col = "Id")



#selecting the training data from input data by excluding the target data and id column
test_data = input_data.loc[:,'MSSubClass':'SaleCondition']

#getting the columns indices whose data type is numerical for preprocessing
numeric_feats = test_data.dtypes[test_data.dtypes != "object"].index



#########################   Data Preprocessing  ########################################################
#  Filling numerical NaN present in each column with the mean of that column 
#  I have filled the NaN values b/c its possible that the other features of the corresponding
#  ...row might be useful 
test_data = test_data.fillna(test_data.mean())



# normalizing the numerical values using l2 norm
test_data[numeric_feats] = preprocessing.normalize(test_data[numeric_feats], norm='l2')
# I had tried to normalize data using min_max_scaler but the result obtained was poor

#getting the columns indices whose data type is categorical for preprocessing
categorical_feats = test_data.dtypes[test_data.dtypes == "object"].index

# Filling categorical NaN present in each column with the mode of that column 
test_data[categorical_feats] = test_data[categorical_feats].fillna(str(test_data[categorical_feats].mode()))



#############################   Feature Selection  ########################################################
# These values have been obtained at later stages after finding them out have been appended here
test_data = test_data.drop(columns=['LowQualFinSF','BsmtFullBath','BsmtHalfBath','Neighborhood','Condition1','Condition2','BldgType','RoofStyle','MSZoning'])

#########################   Data Preprocessing  ########################################################
# Converting categorical variables into dummy variable
test_data = pd.get_dummies(test_data)

#########################   Loading the saved best model  ########################################################
filename = 'finalmodel.pkl'
model = joblib.load(filename)

#########################   Predicting the target values  ############################################### 
predictions = model.predict(test_data)

c=0
y = []  #array to store status
for i in range(predictions.shape[0]):
    dummy = 0
    for j in range (3):
        dummy = dummy + predictions[i,j]
        if (predictions[i,j] == 1.0):
            if (j == 0):
                y.append('NotSold')
            elif (j == 1):
                y.append('SoldFast')
            elif (j == 2):
                y.append('SoldSlow')
    if (dummy == 0):
        y.append(None)

# printing out.csv file
pd.DataFrame({'Id': test_data.index, 'SaleStatus': y}).to_csv("out.csv", index=False)

#reading the data from out.csv and storing as pandas dataframe
out_data = pd.read_csv("out.csv",index_col = "Id")

#reading the data from train.csv and storing as pandas dataframe
gt_data = pd.read_csv("gt.csv",index_col = "Id")

count = 0
for i in range(len(out_data.index)):
    if (out_data.iloc[i,0] == gt_data.iloc[i,0]):
        count = count + 1

print ("Number of rows that were of same value =",count)
print ("Total number of rows ",len(out_data.index)) 
print("Ration =",(count/len(out_data.index)))   
    
        







