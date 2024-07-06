import pandas as pd
import numpy as np
import warnings as warnings

data=pd.read_csv(r"C:\Users\krish\Downloads\Youtube01-Psy (1).csv")

print(data)

data.drop(["COMMENT_ID", "AUTHOR", "DATE"], axis=1, inplace=True)

print(data)

from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test=train_test_split(data["CONTENT"],data["CLASS"])

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect=TfidfVectorizer(use_idf=True, lowercase=True)
x_train_tfidf= tfidf_vect.fit_transform(x_train)
x_train_tfidf.shape

from sklearn.naive_bayes import MultinomialNB

model= MultinomialNB()
model.fit(x_train_tfidf, y_train)

from sklearn.metrics import confusion_matrix, classification_report

x_test_tfidf=tfidf_vect.transform(x_test)
predictions= model.predict(x_test_tfidf)
predictions

confusion_matrix(y_test, predictions)

print(classification_report(y_test, predictions))

model.score(x_test_tfidf, y_test)

import pickle

with open("model.pkl", "wb") as model_file:
    pickle.dump(model , model_file)
with open("tfidf-vect.pkl","wb") as tfidf_vect_file:
    pickle.dump(tfidf_vect, tfidf_vect_file)