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
input_data = pd.read_csv("trainSold.csv")


#selecting the training data from input data by excluding the target data and id column
train_data = input_data.loc[:,'MSSubClass':'SaleCondition']

#getting the columns indices whose data type is numerical for preprocessing
numeric_feats = train_data.dtypes[train_data.dtypes != "object"].index



#########################   Data Preprocessing  ########################################################
#  Filling numerical NaN present in each column with the mean of that column 
#  I have filled the NaN values b/c its possible that the other features of the corresponding
#  ...row might be useful 
train_data = train_data.fillna(train_data.mean())

# removing the outliers by removing the corresponding row if its z-score is >3
train_data = train_data[(np.abs(zscore(train_data[numeric_feats])) < 3).all(axis=1)]

# normalizing the numerical values using l2 norm
train_data[numeric_feats] = preprocessing.normalize(train_data[numeric_feats], norm='l2')
# I had tried to normalize data using min_max_scaler but the result obtained was poor

#getting the columns indices whose data type is categorical for preprocessing
categorical_feats = train_data.dtypes[train_data.dtypes == "object"].index

# Filling categorical NaN present in each column with the mode of that column 
train_data[categorical_feats] = train_data[categorical_feats].fillna(str(train_data[categorical_feats].mode()))



#############################   Feature Selection  ########################################################
# These values have been obtained at later stages after finding them out have been appended here
train_data = train_data.drop(columns=['LowQualFinSF','BsmtFullBath','BsmtHalfBath','Neighborhood','Condition1','Condition2','BldgType','RoofStyle','MSZoning'])

#########################   Data Preprocessing  ########################################################
# Converting categorical variables into dummy variable
train_data = pd.get_dummies(train_data)
target_data= pd.get_dummies(input_data.SaleStatus[train_data.index])


#############################   Feature Selection  ########################################################
#  This part of code is required only once 
# selecting best features by removing # from below lines and finding the important features for each model
#model = DecisionTreeClassifier()
#model = RandomForestClassifier()
model = ExtraTreesClassifier()

# fit the model for important feature selection
model.fit(train_data, target_data)
# get important features
importances = model.feature_importances_
# sorting the feature by importance
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")       
for f in range(train_data.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, train_data.columns[indices[f]], importances[indices[f]])) 

#############################   Applying the models  ########################################################

def classification_model(model,i):
    
    ######## Splitting the data for cross validation  ######
    X_train, X_test, y_train, y_test = train_test_split(train_data, target_data, test_size=0.33, random_state=42)
    
    ######################## hyper-parameter tuning based on the input model  #################################
    if (i == 3):
        parameters = [{'min_samples_leaf': [1, 5, 10,] }]
    else:
        parameters = [{'n_estimators': [1, 5, 10, 15, 20] }]
    
    # Using Grid Search to implement various parameters
    model = GridSearchCV(model, parameters, cv=5)
    model.fit(X_train, y_train)
    # Printing Best parameters
    print("Best parameters set found on development set:")
    print(model.best_params_)
    
    """if (i!=3):
        model_mean_scores = [result.mean_validation_score for result in model.model_scores_]
        plt.figure(i)
        plt.plot(n_estimators, model_mean_scores)"""
        
    ######################## Cross Validating data  ######################################
    # Getting cross validation score on implementing in testing data
    cvs = cross_val_score(model,X_test,y_test,cv=5)
    # Printing cross validation score
    print(" Cross Validation Score")
    print(cvs)
    print("Accuracy: %0.4f (+/- %0.4f)" % (cvs.mean(), cvs.std()*2))
    acc_score = round(model.score(X_test, y_test), 4) #rounding to 4 digits
    print("Accuracy: %0.4f" % (acc_score))
    # Predicting the target values
    predictions = model.predict(X_test) 
    return (round(acc_score,4),model)





#############################   Calling the models  ########################################################
score = 0
selected_model = ExtraTreesClassifier()
for i in range(1,4):
    
    if (i == 1):
        print ("Using RandomForestClassifier for learing: ")
        prec_score,model = classification_model(RandomForestClassifier(),i)
        selected_model = model
        score = prec_score
        print (score)
    elif (i ==2):
        print ("\n Using ExtraTreesClassifier for learing: ")
        prec_score,model = classification_model(ExtraTreesClassifier(),i)
        if (prec_score > score):
            score = prec_score
            print (score)
            selected_model = model
    else:
        print ("\n Using DecisionTreeClassifier for learing: ")
        prec_score,model = classification_model(DecisionTreeClassifier(),i)
        if (prec_score > score):
            score = prec_score
            print (score)
            selected_model = model

filename = 'finalmodel.pkl'
joblib.dump(selected_model, filename)

       

######   plotting #######
"""X = train_data[['OverallQual','GrLivArea']]
Y = target_data
y_send=[]
y=input_data.SaleStatus[train_data.index]
for i in range (train_data.shape[0]):
    if (y[i]=='SoldFast'):
        y_send.append(0)
    elif (y[i]=='SoldSlow'):
        y_send.append(1)
    else:
        y_send.append(2)
m=model.fit(X,y_send)
# Plotting decision regions
x_min, x_max = float(X[['OverallQual']].min() + 1), float(X[['OverallQual']].max() + 5)
y_min, y_max = float(X[['GrLivArea']].min() + 1), float(X[['GrLivArea']].max() + 5)
h = (x_max / x_min)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()]
Z = m.predict(X_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15, 5))
plt.show() """

######################### 



    

    
    
    










                          
