#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:39:04 2019

@author: william Keilsohn
"""

# a = 0.05


# Import Packages:
## Data management:
import pandas as pd
import numpy as np
## Plotting:
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
## Macing Learning:
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


### Solve "Future Warmning" issue:
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

### ---- Please Read This Note: ---- ###
# So I'm kinda commiting a taboo with tis section of code. #
# It ignores all sklearn warnings (that don't outright break the model) #
# B/c I got an error indicating that in one of my models I had two variables which were correlated to each other. 
def warn(*args, **kwargs): #https://stackoverflow.com/questions/32612180/eliminating-warnings-from-scikit-learn
    pass
import warnings
warnings.warn = warn
# Now Generally speaking, in a perfect world, you would want to eliminate at least #
# one of those variables as you are sorta hedging your bets by having both. #
# https://stats.stackexchange.com/questions/29385/collinear-variables-in-multiclass-lda-training #
# However their appears to be some debate as to how much it matters when you're in the 85-95% range #
# as around there suposidly the "second" variable arguably doesn't do much. #
# Anyway, I'm leaving this code in, but I felt it deserved explaining. 
### ------------------------------------------- ###

# Import data:
### Kaggle has a competition going right now for this data... but there are ways around that...
### https://www.kaggle.com/pavlofesenko/titanic-extended
loc = '/home/william/Documents/Class_Scripts/'
boat_data = pd.read_csv(loc + 'train.csv') # Seemed that calling it "train" data was just wierd...
# Also, I chose the training data b/c it was a larger set and would allow for me to look at more passangers.

# Clean data:
## First maybe see what is in the data?
print(boat_data.columns.values)
### Given this list, probably safe to say that we can't use "Survived" to predict survival.
'''
So funny story, the "Testing Data" doesn't come with answers.
Thus validation will have to be done via splitting the larger dataset.
Y_boat = train_data['Survived']
Y_validation = test_data['Survived']
'''
'''
Now I went here to find the full data: https://www.kaggle.com/c/titanic
According to the competition page, Passanger ID is a unique identifyer for each person,
Pclass is where they were on the ship, SibSp is if they had siblings, Parch indicated parents,
Ticket is their ticket number, Fare is how much they paid (In I think pounds), Cabin is their cabin,
embarked is the prot they got on the ship from, and the rest are fairly self explanitory. 

This noted I would reason that their ID and name didn't play a role in if they survived (ticket number
probably didn't matter either). However, a title like Mr./Mrs./Ms. may make a diff if no sex is provided 
and if the tickets could say what class they were in that may matter. So I'm going to try to get that info first.
'''
## First check what's missing:
print(boat_data.isna().sum()) # https://stackoverflow.com/questions/26266362/how-to-count-the-nan-values-in-a-column-in-pandas-dataframe

'''
So age and cabin numbers are missing the most, followed by lifeboat, body, and age. 
From what I can infer based on the data source:
    https://www.kaggle.com/pavlofesenko/titanic-extended/home
the "wiki" columns are instances where someone has gone back and confirmed the original record. 
Therefore, Age_wiki is the confirmed age of the passangers and Name_wiki is the verified name of passengers. 
Thus, as age is missing more values than Age_wiki,and Name_wiki is missing more values than Name 
(with the info essentially being redundant) I'm going to drop those columns.
'''
boat_data = boat_data.drop(columns = ['Age'])
boat_data = boat_data.drop(columns = ['Name_wiki'])

## Now see if my predition(s) about PassId and Name are right:
'''
I'm going to be honest; I'm not 100% sure this is the right test.
https://stats.stackexchange.com/questions/103801/is-it-meaningful-to-calculate-pearson-or-spearman-correlation-between-two-boolea
Inertet seems to think you can get away with a pearson ranked correlation model when two variables are binary, but the concensus is ify.
The other option is to do a logistic regrestion when only one is binary.
    https://www.researchgate.net/post/What_statistical_test_to_use_dependent_variable_is_binary_and_independent_variable_is_continuous
    https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
    https://code.i-harness.com/en/q/186322a
'''
## Create a general mogel for running the logistic regression: # https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
log_reg = LogisticRegression()
Survived = np.array(boat_data['Survived']).reshape(-1,1) # Fixes and error
ID = np.array(boat_data['PassengerId']).reshape(-1,1) # https://stackoverflow.com/questions/35082140/preprocessing-in-scikit-learn-single-sample-depreciation-warning

print(log_reg.fit(ID, Survived).score(ID, Survived)) # Not Significant.
#### Needs a number so...
boat_data['Name'] = boat_data['Name'].str.len() # http://pandas.pydata.org/pandas-docs/version/0.17/generated/pandas.Series.str.len.html
Name = np.array(boat_data['Name']).reshape(-1,1) # https://stackoverflow.com/questions/35082140/preprocessing-in-scikit-learn-single-sample-depreciation-warning
print(log_reg.fit(Name, Survived).score(Name, Survived))# Not significant.
boat_data = boat_data.drop(columns = ['PassengerId', 'Name'])
del Name 

## Deal with the columns missing the most data:
print(boat_data['Ticket'].head())
'''
First it turns out that the ticket column is made up of two values:
    1) An alpha/numeric string indicating anything from if people are sharing a ticket to what port they got on the boat at.
    2) There actual room number.
    https://www.kaggle.com/c/titanic/discussion/11127
In my opinion, they ticket/room number seems to be the more important portion, so lets clean up the ticket number to just be that number.
'''
boat_data['Ticket'] = boat_data['Ticket'].str.extract(r'(^|\s+)(\d+)')[1] #Creates a pandas dataframe...
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.extract.html
# https://stackoverflow.com/questions/48144206/how-can-i-extract-credit-card-substring-from-a-string-using-python
print(boat_data.isna().sum()) # Woth noting that 4 tckets don't have numbers.
### Due to these missing tickets, we have some bias in the system. 
### Technically there are some stats to get around this, but due to the nature of the beast, I'm just going to drop them,
### and if it turns out tickets don't have a correlation I'll add them back.
boat_data = boat_data.dropna(subset = ['Ticket'])
Survived = np.array(boat_data['Survived']).reshape(-1,1) # Doing it again b/c we cut out values

boat_data['Ticket'] = boat_data['Ticket'].astype(int) # https://stackoverflow.com/questions/15891038/change-data-type-of-columns-in-pandas
Ticket = np.array(boat_data['Ticket']).reshape(-1,1) # https://stackoverflow.com/questions/35082140/preprocessing-in-scikit-learn-single-sample-depreciation-warning
print(log_reg.fit(Ticket, Survived).score(Ticket, Survived)) # Insignificant.
'''
I'm going to make the argument, that even though the ticket is "insignificant" from a statistical point point of view it still may play a role by acting
as a stand in for cabin. I mean, hypothetically people of a certian class should have been grouped by cabin, and this should have been reflected in their ticket.
...also if it really doesn't matter it should get a weight of 0 and be dicredited. Thus no harm, no foul.
'''
## Unforunitly, this doesn't really provide any information on cabin number:
boat_data = boat_data.drop(columns = ['Cabin'])

## Last issue is lifeboat vs. body:
'''
Lifeboat seems to indicate which life boat they got into.
Body seems to be an indication of if a body was recovered.
Both can probably be filled in as binary values.
'''
boat_data['Body'] = boat_data['Body'].notnull().astype('int').fillna(0) # https://stackoverflow.com/questions/37543647/how-to-replace-all-non-nan-entries-of-a-dataframe-with-1-and-all-nan-with-0
boat_data['Lifeboat'] = boat_data['Lifeboat'].notnull().astype('int').fillna(0)


'''
Now we get an issue with values being strings instead of floats of other numbers.
Proposed solutions:
    1) Turn sex into a binary -- 1/0
    2) Turn Embarked into a set of numbers -- 1/2/3
    3) Drop HomeTown, and Boarded
        - Hometown can't be converted to a number easily...and is argubly irelevant with this many passangers.
        - Boarded will be represented as Embarked.
    4) Turn Destination into a set of numbers --
        print(boat_data['Destination'].unique())
        Turns out there are a lot of these. 
        From what I can tell, it seems like the passangers eventual destination (like the ship was part of the longer passage, not where they got off)
        was recorded. 
'''
boat_data = boat_data.drop(columns = ['Hometown', 'Boarded', 'Destination'])
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.replace.html
boat_data['Sex'] = boat_data['Sex'].str.replace('female','1')
boat_data['Sex'] = boat_data['Sex'].str.replace('male', '0')
boat_data['Sex'] = boat_data['Sex'].astype('int')
boat_data['Embarked'] = boat_data['Embarked'].str.replace('S', '1')
boat_data['Embarked'] = boat_data['Embarked'].str.replace('Q', '2')
boat_data['Embarked'] = boat_data['Embarked'].str.replace('C', '3')
boat_data['Embarked'] = boat_data['Embarked'].fillna(0).astype('int')

### So, yes, there are still some missing values
print(boat_data.isna().sum()) # https://stackoverflow.com/questions/26266362/how-to-count-the-nan-values-in-a-column-in-pandas-dataframe
### Unfortunitly, the machine needs a number in every spot or it has trouble compressing the data.
boat_data = boat_data.fillna(0) 
### So, now one could argure that a lack of data is not the same thing as a point with no data, this is the only was things will run.

### Therefore, we can move on to the actual learning.

# Create training and validation subsets: # Ppt
## Also, yes, I'm writingover the above. 
data_array = boat_data.values
X = data_array[:,1:]         # All other data
Y = data_array[:,0]          # Just survival
validation_size = 0.20       # Keep 20% of data for testing
seed = 10
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)


# Run Learning: 
scoring = 'accuracy' # From Ppt
### Internet likes logistic regressions... but lets do ALL THE THINGS!
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Let's just make the final results stand out and look pretty:
print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('\n')
print('Final Results Below:')
print('\n')

results = [] # Ppt
names = []
print("\nShow results for each of the methods\nName\t   Mean\t\t   STD")
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, 
			cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s:\t %f \t(%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
