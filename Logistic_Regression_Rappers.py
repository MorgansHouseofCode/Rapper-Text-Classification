# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:04:16 2020

@author: Morgan
"""



import numpy as np
from sklearn.linear_model import LogisticRegression
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from  TextPreprocessing import artist_dict
import os 
#Load files
os.chdir('C:/Users/Morgan/Documents/rap NLP project/TestTrainData') #setting the directory to a folder for juest train test split
labels_train =pd.read_csv("labels_train.csv")
labels_test = pd.read_csv("labels_test.csv")
feature_X_Train = np.load('features_X_Train.npy')
features_X_Test = np.load('features_X_Test.npy')
os.chdir('C:/Users/Morgan/Documents/rap NLP project')#changing the directory to the general one 

#I cantuse the same data I used in the SVM program because Of how I transformed that data 
#Randomized Search Cross Validation
# C
C = [float(x) for x in np.linspace(start = 0.1, stop = 1, num = 10)]

# multi_class
multi_class = ['multinomial']

# solver
solver = ['newton-cg', 'sag', 'saga', 'lbfgs']
 
# class_weight
class_weight = ['balanced', None]

# penalty
penalty = ['l2']

# Create the random grid
random_grid = {'C': C,
               'multi_class': multi_class,
               'solver': solver,
               'class_weight': class_weight,
               'penalty': penalty}

pprint(random_grid)
lrc = LogisticRegression(random_state=8)

# Definition of the random search
random_search = RandomizedSearchCV(estimator=lrc,
                                   param_distributions=random_grid,
                                   n_iter=50,
                                   scoring='accuracy',
                                   cv=5, 
                                   verbose=1, 
                                   random_state=8)

# Fit the random search model
random_search.fit(feature_X_Train, labels_train)


print("The best hyperparameters from Random Search are: \n" ,random_search.best_params_ )
print("The mean accuracy of a model with these hyperparameters is: \n",random_search.best_score_)



best_lrc = random_search.best_estimator_
best_lrc.fit(feature_X_Train, labels_train)
lrc_pred = best_lrc.predict(features_X_Test)
# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train, best_lrc.predict(feature_X_Train)))
# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test, lrc_pred))
# Classification report
print("Classification report")
print(classification_report(labels_test,lrc_pred))


Artist_inv_map = {v: k for k, v in artist_dict.items()}
LOG_PREDICT_LABELS=list(map(Artist_inv_map.get,lrc_pred))
LOG_TEST_LABELS=list(map(Artist_inv_map.get,labels_test))

labels = [*artist_dict]
cm = confusion_matrix(LOG_PREDICT_LABELS, LOG_TEST_LABELS,labels)
print(cm)
plt.figure(figsize=(12.8,6))
sns.heatmap(cm, 
            annot=True,
            xticklabels=labels, 
            yticklabels=labels,
            cmap="Blues")
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Confusion matrix of Logistic Regression with accuracy of 86.20%')
plt.show()


