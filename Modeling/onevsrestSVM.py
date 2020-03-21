# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 21:31:57 2020

@author: Morgan
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from  TextPreprocessing import artist_dict
#Load files
os.chdir('C:/Users/Morgan/Documents/rap NLP project/TestTrainData') #setting the directory to a folder for juest train test split
labels_train =pd.read_csv("labels_train.csv")
labels_test = pd.read_csv("labels_test.csv")
feature_X_Train = np.load('features_X_Train.npy')
features_X_Test = np.load('features_X_Test.npy')
os.chdir('C:/Users/Morgan/Documents/rap NLP project')#changing the directory to the general one 

##############################
#Data preperation for SVM classifier 
#Attaching artist codes to the numpy array, moving columns around and organizing the data so one hot encoding can be done 
def AttachingLabels(feature_array,labels):
    feature_array = pd.DataFrame(feature_array)
    labels=labels.reset_index(drop=True)
    feature_arraypd = pd.concat([feature_array,labels],axis =1)
    cols = feature_arraypd.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    feature_arraypd=feature_arraypd[cols]
    feature_arraypd = feature_arraypd.sort_values('artist_code', ascending=True)

    return feature_arraypd

#Doing one hot encoding
def onehot(y):
    ret = np.zeros((len(y),11))
    for i in range(len(y)):
        num = y[i]
        ret[i,int(num)] = 1.0
    return ret
############################################################################




Train = AttachingLabels(feature_X_Train,labels_train)
Test = AttachingLabels(features_X_Test,labels_test)


XTrain_OVR = np.array(Train.iloc[:, 1:])
XTrainTest_OVR = onehot(np.array(Train['artist_code'].values))

Test_OVR = np.array(Test.iloc[:, 1:])
Test_OVR_Labels = np.array(Test['artist_code'].values)

##############################################################################



#creating the classfiers and training the model
classifiers = [SVC(probability=True, C=1.5) for _ in range(11)]
print('Training...')
for i in range(11):
    print(i)
    classifiers[i].fit(XTrain_OVR,XTrainTest_OVR[:,i]) #the i takes it go through the 12 artsit 


#testing the model 
print('Predicting test...')
predict_test = [0.0 for _ in range(Test_OVR_Labels.shape[0])]
for j in range(Test_OVR_Labels.shape[0]):
    pred = [cla.predict_proba([Test_OVR[j,:]])[0,1] for cla in classifiers] #
    predict_test[j] = np.argmax(pred)

print(sum(predict_test == Test_OVR_Labels) / float(len(Test_OVR_Labels)))



##############################################################################

#mapping the artist codes to the artist's name for confusion matrix
Artist_inv_map = {v: k for k, v in artist_dict.items()}
OVR_Predict_Mapped=list(map(Artist_inv_map.get,predict_test))
OVR_TEST_LABELS=list(map(Artist_inv_map.get,Test_OVR_Labels))
labels = [*artist_dict]

#creating the confusion matrix
cm = confusion_matrix(OVR_Predict_Mapped, OVR_TEST_LABELS,labels)
plt.figure(figsize=(12.8,6))
sns.heatmap(cm, 
            annot=True,
            xticklabels=labels, 
            yticklabels=labels,
            cmap="Blues")
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Confusion matrix of OneVRestSVC with Accuracy of 87.24')
plt.show()
