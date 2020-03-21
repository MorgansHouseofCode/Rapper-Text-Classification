# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 03:13:26 2020

@author: Morgan
"""

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


RapperDf= pd.read_csv("outputMacMiller.csv")
RapperDf_Cleaned= pd.read_csv("RapperDf_Cleaned.csv", encoding = "latin-1")

def clean_text(text):
    text = text.lower()
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

Rappers_and_Lyrics = RapperDf[['artist','title','lyrics']]
Rappers_and_Lyrics.loc[:,'lyrics'] = Rappers_and_Lyrics.loc[:,'lyrics'].apply(lambda com : clean_text(com))

stop_words = list(stopwords.words('english'))
newStopWords = ['niggas',"nigga","uh","yeah","bitch","oh","ya","yo","ah","lil","woah","ayy"] #adding words to stop word dictioanry
stop_words.extend(newStopWords)

Rappers_grouped = Rappers_and_Lyrics.groupby(["artist"])["title","lyrics"].agg(lambda x: ' '.join(x.astype(str)))
Rappers_grouped = Rappers_grouped[['lyrics']]
Rappers_grouped.index
Rappers_and_Lyrics['artist'].unique()

tfidf = TfidfVectorizer(encoding='utf-8',
                        stop_words=stop_words,
                        lowercase=False,
                        sublinear_tf=True)
sparse_matrix = tfidf.fit_transform(Rappers_grouped['lyrics'])
doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix, 
                  columns=tfidf.get_feature_names(),
                  index= Rappers_grouped.index)

from sklearn.metrics.pairwise import cosine_similarity
CosineDF = pd.DataFrame(cosine_similarity(df, df),index= Rappers_grouped.index, columns = Rappers_grouped.index)
CosineDF.to_csv("CosineDF.csv", index=True)








