# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:39:59 2020

@author: Morgan
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')


os.chdir('C:/Users/Morgan/Documents/rap NLP project')#changing the directory to the general one 

RapperDf= pd.read_csv("outputMacMiller.csv")
RapperDf = RapperDf.dropna(axis=0, subset=['lyrics'])
RapperDf_Just_Important_Parts = RapperDf[['artist','album','title','lyrics']]
RapperDf_Just_Important_Parts.artist.value_counts() #number of songs per artist




#text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n"," ",text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    

    text = text.strip(' ')
    return text

RapperDf_Just_Important_Parts.loc[:,'lyrics'] = RapperDf_Just_Important_Parts.loc[:,'lyrics'].apply(lambda com : clean_text(com))

RapperDf_Just_Important_Parts.loc[50,'lyrics']


def extract_artist_map(df):
    artist_map = {}
    artists = df["artist"].unique()
    for i, artist in enumerate(artists):
        artist_map[artist] = i
    return artist_map


artist_dict=extract_artist_map(RapperDf_Just_Important_Parts) # gives each artist a encoding number to give ot the text
RapperDf_Just_Important_Parts.loc[:,'artist_code'] = RapperDf_Just_Important_Parts.loc[:,'artist'].map(artist_dict) #cretes the column I need 


RapperDf_Just_Important_Parts= RapperDf_Just_Important_Parts[["artist","lyrics","artist_code"]]

#####################################################################################
#lemenize words
wordnet_lemmatizer = WordNetLemmatizer()

nrows = len(RapperDf_Just_Important_Parts)
lemmatized_text_list = []

for i in range(0, nrows):
    
    # Create an empty list containing lemmatized words
    lemmatized_list = []
    
    # Save the text and its words into an object
    text=(RapperDf_Just_Important_Parts.iloc[i,1])
    text_words = text.split(" ")

    # Iterate through every word to lemmatize
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        
    # Join the list
    lemmatized_text = " ".join(lemmatized_list)
    
    # Append to the list containing the texts
    lemmatized_text_list.append(lemmatized_text)
RapperDf_Just_Important_Parts["parsed_Lyrics"] = lemmatized_text_list
########################################################################
#Stop words now 
stop_words = list(stopwords.words('english'))
newStopWords = ['niggas',"nigga","uh","yeah","bitch","oh","ya","yo","ah","lil","woah","ayy"] #adding words to stop word dictioanry
stop_words.extend(newStopWords)
for stop_word in stop_words:
    regex_stopword = r"\b" + stop_word + r"\b"
    RapperDf_Just_Important_Parts.loc['parsed_Lyrics'] = RapperDf_Just_Important_Parts['parsed_Lyrics'].str.replace(regex_stopword, '')
################################################################
RapperDf_Just_Important_Parts = RapperDf_Just_Important_Parts.dropna(axis=0, subset=['lyrics'])
#making training val split
X_train, X_test, Y_train, Y_test= train_test_split(RapperDf_Just_Important_Parts['parsed_Lyrics'],
                                                    RapperDf_Just_Important_Parts['artist_code'],test_size=0.2, random_state=36)


#making test split
############################################################################################
    #Seeing how the break down of the artist test and train is 
X_test.value_counts() #number of songs per artist
Y_train.value_counts() #number of songs per artist
X_train.shape
#################################################################################################    
# Parameter election
ngram_range = (1,2)
min_df = 20
max_df = 1000
max_features = 300
################################################################
#TFIDF Creation
tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        norm='l2',
                        sublinear_tf=True)


#################################
#save the data frames to a folder so I can sue them in other scripts
os.chdir('C:/Users/Morgan/Documents/rap NLP project/TestTrainData') #setting the directory to a folder for juest train test split
RapperDf_Just_Important_Parts.to_csv("RapperDf_Just_Important_Parts.csv", index = False)
#creating the tf IDf matrix
features_X_Train = tfidf.fit_transform(X_train).toarray()
features_X_Test = tfidf.transform(X_test).toarray()
#saving nummpy arrays
np.save('features_X_Train.npy',features_X_Train)
np.save('features_X_Test.npy',features_X_Test)
labels_train = Y_train
labels_test = Y_test
#saving the labels
labels_train.to_csv("labels_train.csv", index=False)
labels_test.to_csv("labels_test.csv", index=False)
os.chdir('C:/Users/Morgan/Documents/rap NLP project')#changing the directory to the general one 


