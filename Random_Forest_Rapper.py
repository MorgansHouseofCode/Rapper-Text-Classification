# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 18:08:38 2020

@author: Morgan
"""



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  confusion_matrix, accuracy_score
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from  TextPreprocessing import artist_dict
import os 
#Load files
os.chdir('C:/Users/Morgan/Documents/rap NLP project/TestTrainData') #setting the directory to a folder for juest train test split
labels_train =pd.read_csv("labels_train.csv")
labels_test = pd.read_csv("labels_test.csv")
feature_X_Train = np.load('features_X_Train.npy')
features_X_Test = np.load('features_X_Test.npy')
os.chdir('C:/Users/Morgan/Documents/rap NLP project')#changing the directory to the general one 




#See what hyperparameters the model has:
rf_0 = RandomForestClassifier(random_state = 8)
print('Parameters currently in use:\n')
pprint(rf_0.get_params())



# =============================================================================
# n_estimators = number of trees in the forest.
# max_features = max number of features considered for splitting a node
# max_depth = max number of levels in each decision tree
# min_samples_split = min number of data points placed in a node before the node is split
# min_samples_leaf = min number of data points allowed in a leaf node
# bootstrap = method for sampling data points (with or without replacement)
# =============================================================================


# n_estimators
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]

# max_features
max_features = ['auto', 'sqrt']

# max_depth
max_depth = [int(x) for x in np.linspace(20, 100, num = 5)]
max_depth.append(None)

# min_samples_split
min_samples_split = [2, 5, 10]

# min_samples_leaf
min_samples_leaf = [1, 2, 4]

# bootstrap
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)

# First create the base model to tune
random_Forest = RandomForestClassifier(random_state=8)

# Definition of the random search
random_search = RandomizedSearchCV(estimator=random_Forest,
                                   param_distributions=random_grid,
                                   n_iter=50,
                                   scoring='accuracy',
                                   cv=3, 
                                   verbose=1, 
                                   random_state=8)

# Fit the random search model
random_search.fit(feature_X_Train, labels_train)
print("The best hyperparameters from Random Search are:")
print(random_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(random_search.best_score_)


best_random_Forest = random_search.best_estimator_
best_random_Forest.fit(feature_X_Train, labels_train)
random_forest_pred = best_random_Forest.predict(features_X_Test)

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test, random_forest_pred))



#Creating the Confusion matrix
Artist_inv_map = {v: k for k, v in artist_dict.items()}
random_forest_labels=list(map(Artist_inv_map.get,random_forest_pred))
random_forest_test_labels=list(map(Artist_inv_map.get,labels_test))
labels = [*artist_dict]
cm = confusion_matrix(random_forest_labels, random_forest_test_labels,labels)
plt.figure(figsize=(12.8,6))
sns.heatmap(cm, 
            annot=True,
            xticklabels=labels, 
            yticklabels=labels,
            cmap="Blues")
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Confusion matrix of Random Forest with accuracy: 78.20%')
plt.show()





