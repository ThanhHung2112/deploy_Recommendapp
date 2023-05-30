import streamlit as st
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import calendar

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import f1_score
import nltk

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words("portuguese")
import re

def main_reason(label, data):

    df_ans = data
    df_ans = df_ans[['review_score','review_comment_message','Sentiments']]

    sentiments_thich = df_ans[df_ans['Sentiments'] == label]['review_comment_message']
    
    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(sentiments_thich)

    # Áp dụng K-means clustering
    k = 3  # Số cụm 
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(vectors)

    cluster_labels = kmeans.labels_

    cluster_centers = kmeans.cluster_centers_

    num_samples = 3
    similar_sentences = []
    for i in range(k):
        cluster_indices = np.where(cluster_labels == i)[0]
        distances = np.linalg.norm(vectors[cluster_indices] - cluster_centers[i], axis=1)
        closest_indices = np.argsort(distances)[:num_samples]
        closest_sentences = [sentiments_thich.iloc[idx] for idx in cluster_indices[closest_indices]]
        similar_sentences.extend(closest_sentences)

    # Hiển thị các câu gần với tâm của từng cụm
    st.write("Các câu gần với tâm của từng cụm:")
    for sentence in similar_sentences:
        st.write(sentence)

    return

st.title("Vũ cute")
uploaded_file = st.file_uploader('Chooseafile')

if uploaded_file is not None:

    df=pd.read_csv(uploaded_file, encoding='utf-8')

    st.write(df)

input = st.text_input('Positive - negative')

b2 = st.button('RESULT')
if b2:
    result = main_reason(input, df)