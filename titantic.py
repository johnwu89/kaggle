#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:35:54 2017

@author: John

For classfiying the Titanic data set
"""


import pandas as pd
import numpy as np


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# some exploratory data analysis
import matplotlib.pyplot as plt
import seaborn as sb
sb.set()
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report




from sklearn import metrics # for evaluating metrics 
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from scipy import interp


def data():
    global data 
    global test
    
    train=pd.read_csv("/Users/John/Downloads/trainTitanic.csv")
    print(len(train))
    print("===================================\n Train data \n===================================")
    
    columnNames=train.columns.values.tolist()
    
    print("===================================\n Column names\n===================================")
    print(columnNames)


    print(train.head(5))
    print(train.info()) #checking the data to see missing values, Cabin data
    # has a lot of nulls 

    test=pd.read_csv("/Users/John/Downloads/testTitanic.csv")  
    print("===================================\n Test data \n===================================")
    print(test.tail(5))
      
    # Store our passenger ID for easy access
    PassengerId = test['PassengerId']
    
    
    print("===================================\n data types\n===================================")
    print(train.dtypes)
    print(train.describe())
    
    
    print("===================================\n Null Values?\n===================================")
    
    print(train.isnull().any())# check for each of the columns if there are Null values
    print(test.isnull().any())# check for each of the columns if there are Null values


    '''
        http://pbpython.com/categorical-encoding.html
        5 object data types  Name, Sex, Ticket, Cabin, Embarked
        
        ===================================
         data types
        ===================================
        PassengerId      int64
        Survived         int64
        Pclass           int64
        Name            object <-----
        Sex             object <-----
        Age            float64
        SibSp            int64
        Parch            int64
        Ticket          object <----- ticket number, prob not useful
        Fare           float64
        Cabin           object <-----
        Embarked        object <-----
        dtype: object
    
    '''

    full_data = [train, test]

    # cleaning data and filling NA 
    # 2 types of data needs to be filled: Age, Embarked
    
    # Age
    for dataset in full_data:
        age_avg 	   = dataset['Age'].mean()
        age_std 	   = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()
        
        # randomly generate mean + Std, mean - std,
        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)
        
    train['CategoricalAge'] = pd.cut(train['Age'], 5)
    
    print("===================================\n CategoricalAge \n===================================")
    print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
    
    
    
    #Embarked, fill with most occuring value ( "S" )
    for dataset in full_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
    print("===================================\n Embarked \n===================================")
    print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
    
    
    #Fare, since training data has missing dataset
    
    for dataset in full_data:
        dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
    
    print("===================================\n Categorical Fare, computed \n===================================")

    print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
        
        
    
    #Encoding ceategorical data into numerical data
    '''
        http://pbpython.com/categorical-encoding.html
        5 object data types:
            Name, 
            Sex, 
            Ticket,
            Cabin,  <---- too little data,dropping this column 
            Embarked
    '''

    for dataset in full_data:
        # Mapping Sex
        dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
        
        # Mapping Embarked
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
        
        # Mapping Fare
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
        dataset['Fare'] = dataset['Fare'].astype(int)
        
        # Mapping Age
        dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4
    
    print("===================================\n Cleaned data \n===================================")

    train = train.drop(['Cabin', 'CategoricalAge', 'CategoricalFare'], axis = 1)


    print(train.info()) #checking the data to see missing values, Cabin data



    '''
    Feature selection - 特征提取
    
    4 ways of statistical feature selection
    
    1. Univariate Selection.
    2. Recursive Feature Elimination.
    3. Principle Component Analysis.
    4. Feature Importance.

    '''






    # 1.Univariate Selection
    # X = all the data, which ones should i pick 
    X = train[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
    # Y has the answer
    Y = train['Survived']  # the data label, answer
    
    print(X)
    print(Y)


    
    # feature extraction
    test = SelectKBest(score_func=chi2, k=4)
    fit = test.fit(X, Y)
    
    
    # summarize scores
    np.set_printoptions(precision=3)
    
    print("===================================\n The Fitting scores\n===================================")
    print(fit.scores_)
    features = fit.transform(X)
    # summarize selected features
    print(features)

    '''
    1. univariate's data answer
        
    
    The Fitting scores
    
        ===================================
        [['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]    
        [  3.313  30.874  92.702   1.741   2.582  10.097  64.722  11.353]
        [[3 1 0 0]
         [1 0 3 1]
         [3 0 1 0]
         ..., 
         [3 0 2 0]
         [1 1 2 1]
         [3 1 0 2]]
            
    '''








































def main():
    data()

    
    
if __name__ == "__main__":
    main()

