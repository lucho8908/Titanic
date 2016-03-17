import sys
import time
import csv as csv 
import numpy as np
#Data manipulation library
import pandas
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

# Import data as a data fram on pandas
data = pandas.read_csv("csv/train.csv")

# Convert data frame to array
data1 = np.asarray(data)

number_passengers = np.size(data1[0::,1].astype(np.float))
number_survived = np.sum(data1[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

# This finds where all the elements in the gender column that equals female
women_only_stats = data1[0::,4] == "female"

# This finds where all the elements do not equal female (i.e. male)
men_only_stats = data1[0::,4] != "female"

# Using the index from above we select the females and males separately
women_onboard = data1[women_only_stats,1].astype(np.float)     
men_onboard = data1[men_only_stats,1].astype(np.float)

# Then we finds the proportions of them that survived
proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)  
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard) 

print 'Proportion of women who survived is %s' % proportion_women_survived
print 'Proportion of men who survived is %s' % proportion_men_survived

#------- Data cleaning ---------

#--- Age
data['Age'] = data['Age'].fillna(data['Age'].median())

#--- Sex
data.loc[data["Sex"] == "male", "Sex"] = 0 
data.loc[data["Sex"] == "female", "Sex"] = 1
data["Sex"] = data["Sex"].fillna(0)

#--- Embarked
data["Embarked"] = data["Embarked"].fillna("S")
data.loc[data["Embarked"] == "S", "Embarked"] = 0
data.loc[data["Embarked"] == "C", "Embarked"] = 1
data.loc[data["Embarked"] == "Q", "Embarked"] = 2

# The .apply method generates a new series
data["Name_len"] = data["Name"].apply(lambda x: len(x))

# Generating a familysize column
data["FamilySize"] = data["SibSp"] + data["Parch"]

# Extract only numerical columns
num_data = data[["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare","Name_len","FamilySize"]]

# Import my PCA algorithm
from myPCA import *
for num_comp in np.arange(1,9):
    start_time = time.time() #Starts time coounter
    pca = myPCA(np.asmatrix(num_data),num_comp) #Runs myPCA
    
    TX = pca['TX'] # Data transformed using myPCA
    
    # Index and colmns names to create data frame
    index  = np.arange(np.shape(pca['TX'])[0])
    columns = np.arange(1,np.shape(pca['TX'])[1]+1)
    columns = map(str, columns)
    
    #Data frame of transmormed variables and concatenae the survived column
    T_data = pandas.DataFrame(TX, index=index, columns=columns)
    T_data = pandas.concat([data['Survived'], T_data], axis=1)
    
    #As predictors use all the columns in T_data
    predictors = columns
    
    # Initialize the algorithm class
    alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
    
    #Calculates the scores for the training dataframe and pronts it
    scores = cross_validation.cross_val_score(alg, T_data[predictors], T_data["Survived"], cv=3)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Error de crossvalidation: %s" % scores.mean())
