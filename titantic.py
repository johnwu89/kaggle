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

       # pd.get_dummies(obj_df, columns=["drive_wheels"])

    '''
        http://pbpython.com/categorical-encoding.html
        5 object data types  Name, Sex, Ticket, Cabin, Embarked
        
        ===================================
         data types
        ===================================
        PassengerId      int64
        Survived         int64
        Pclass           int64
        Name            object
        Sex             object
        Age            float64
        SibSp            int64
        Parch            int64
        Ticket          object
        Fare           float64
        Cabin           object
        Embarked        object
        dtype: object
    
    '''

    full_data = [train, test]

   
    # cleaning data and filling NA 
    # 2 types of data needs to be filled: Age, Cabin( 数据太少了，不要 687 个 missing), Embarked
    
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
    print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
    
    
    #Embarked, fill with most occuring value ( "S" )
    for dataset in full_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
    print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
    
    






















def main():
    data()

    
    
if __name__ == "__main__":
    main()

