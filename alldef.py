import streamlit as st
import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
# stopwords = stopwords.words("portuguese")
import spacy.cli
spacy.cli.download("en_core_web_sm")    
import en_core_web_sm

spc_en = en_core_web_sm.load()

import lifetime
from datetime import datetime
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter
from lifetimes.utils import calibration_and_holdout_data
from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases
from sklearn.preprocessing import StandardScaler
from googletrans import Translator
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Định nghĩa hàm main_reason(label, data, stopwords)


def predict(data):
    # Model/svm_fixed_plane.sav
    loaded_model = pickle.load(open('Model/svm_plane.sav', 'rb'))
    vector = pickle.load(open('Model/vector_conv.sav', 'rb'))

    # Chuẩn bị dữ liệu cho việc tìm lý do chính
    df_ans = data.copy()
    df_ans['Sentiments'] = '' 
    df_ans=df_ans.dropna(subset=['review_comment_message']) # Thêm cột 'Sentiments' rỗng
    review_comment_messages = df_ans['review_comment_message'].tolist()
    vectors = vector.transform(review_comment_messages)

    # Dự đoán nhãn 'Positive' hay 'Negative'
    predicted_labels = loaded_model.predict(vectors)
    df_ans['Sentiments'] = predicted_labels

    return df_ans

def translate_text(text, target_language = "vietnamese"):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

def main_reason(label, data , stopwords):

    # Load mô hình đã huấn luyện
    df_ans = data.copy()

    # Tìm lý do chính
    sentiments = df_ans[df_ans['Sentiments'] == label]['review_comment_message']
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentiments)

    # Hiện bảng
    df_label = df_ans[df_ans['Sentiments'] == label]
    st.header('Reviews with ' + str(label)+ ' sentiments:')
    st.write(df_label)

    # Áp dụng K-means clustering
    k = 3 # Số cụm
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(vectors)

    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

        # Clustering
    wcss = []

    for i in range(1, 20):
        kmeans = KMeans(n_clusters=i, max_iter=1000, random_state=0)
        kmeans.fit(vectors)
        wcss.append(kmeans.inertia_)

    # ebove(wcss)

    num_samples = 10
    similar_sentences = []
    sentences = []

    for i in range(k):
        cluster_indices = np.where(cluster_labels == i)[0]
        distances = np.linalg.norm(vectors[cluster_indices] - cluster_centers[i], axis=1)
        closest_indices = np.argsort(distances)[:num_samples]
        closest_sentences = [sentiments.iloc[idx] for idx in cluster_indices[closest_indices]]
        
        similar_sentences.extend(closest_sentences)
        for i, sentence in enumerate(similar_sentences):
            st.write(sentence)
            if i ==10: break
            # sentences.append(sentence)
        # n-gram models
        most_common_ngrams5, max_count = build_ngram_model(similar_sentences, 5)
        most_common_ngrams4, max_count = build_ngram_model(similar_sentences, 4)
        most_common_ngrams3, max_count = build_ngram_model(similar_sentences, 3)
        most_common_ngrams2, max_count = build_ngram_model(similar_sentences, 2)
        most_common_ngrams, max_count = build_ngram_model(similar_sentences, 1)


        # most_common_ngrams, max_count = build_ngram_model(similar_sentences, 3)
        for ngram in most_common_ngrams:
                sentences.append(" ".join(ngram))
        for ngram in most_common_ngrams2:
                sentences.append(" ".join(ngram))
        for ngram in most_common_ngrams3:
                sentences.append(" ".join(ngram))
        for ngram in most_common_ngrams4:
                sentences.append(" ".join(ngram))
        # for ngram in most_common_ngrams5:
        #         sentences.append(" ".join(ngram))
                # st.write(" ".join(ngram))
        conclution = find_most_similarities(sentences)
            
        st.write(f">> Main: ",conclution)

        st.write('-------')

        similar_sentences = []
        sentences = []


    # Hiển thị các câu gần với tâm của từng cụm
    # st.write("Display sentences that are close to the meaning of each cluster:")
    # for i, sentence in enumerate(similar_sentences):
    #     st.write(sentence)
    #     if i in [2,5,8]:
    #         st.write('-------')

####################################################################
####################################################################    

def find_most_similarities (sentences):
    
    vectorizer = CountVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    cosine_similarities = cosine_similarity(sentence_vectors)
    # Tìm câu có độ tương đồng cao nhất với các câu khác
    most_similar_index = -1
    max_similarity_score = -1
    
    for i in range(len(sentences)):
        similarity_sum = 0
        
        for j in range(len(sentences)):
            if i != j:
                similarity_sum += cosine_similarities[i, j]
        
        average_similarity = similarity_sum / (len(sentences) - 1)
        
        if average_similarity > max_similarity_score:
            max_similarity_score = average_similarity
            most_similar_index = i

    most_similar_sentence = sentences[most_similar_index]
    # sentences3 = [sentence for sentence in sentences if len(sentence.split()) == 3]
    # sentences4 = [sentence for sentence in sentences if len(sentence.split()) == 4]
    result = most_similar_sentence  
    # result = [sentence for sentence in sentences3 if most_similar_sentence in sentence]
    # result = [sentence for sentence in sentences4 if most_similar_sentence in sentence]
    # if len(result) ==1:
         
    return result
    # else: return result
    # print("Câu có độ tương đồng cao nhất:", most_similar_sentence)
    
    

####################################################################
####################################################################

def preprocess_text(text):

    stopwords_eng = stopwords.words("english")
    text = text.lower()
    text = text.replace(",", "").replace(".", "").replace("!", "").replace("?", "")

    text = re.sub(r"[\W\d_]+", " ", text)

    text = [pal for pal in text.split() if pal not in stopwords_eng]

    spc_text = spc_en(" ".join(text))
    tokens = [word.lemma_ if word.lemma_ != "-PRON-" else word.lower_ for word in spc_text]
  
    return " ".join(tokens)
    

####################################################################
####################################################################

def build_ngram_model(sentences, n):
    # generate counters for each
    ngram_counts = defaultdict(int)

    for sentence in sentences:
        sentence = preprocess_text(sentence)
        words = sentence.split()

        # generate bigrams
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngram_counts[ngram] += 1

    # highest probability
    max_count = max(ngram_counts.values())
    most_common_ngrams = [ngram for ngram, count in ngram_counts.items() if count >= max_count]

    return most_common_ngrams, max_count

####################################################################
####################################################################

def ebove(wcss):
    plt.plot(range(1, 20), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    st.pyplot(plt)

####################################################################
####################################################################

def draw_nlp_plot(data, label):

    df = data.copy()
    # Lọc chỉ những hàng có Sentiments label
    positive_reviews = df[df['Sentiments'] == label]

    # Đếm số lần xuất hiện của từng từ
    word_counts = positive_reviews['review_comment_message'].str.split(expand=True).stack().value_counts()

    # Lấy top 10 từ xuất hiện nhiều nhất
    top_10_words = word_counts.head(10)

    # Vẽ biểu đồ cột
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_10_words.values, y=top_10_words.index, palette='viridis')
    plt.xlabel('Số lượng')
    plt.ylabel('Từ')
    plt.title('Top 10 most frequently occurring words in the reviews '+str(label))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Hiển thị biểu đồ trên Streamlit
    st.pyplot(plt)

####################################################################
####################################################################

def associate (data,sp):

    cus_items = data.copy()
    list_item = cus_items.groupby('customer_unique_id')['product_id'].agg(list)
    # item_set = pd.DataFrame({'product_id': list_item})
    item_set =  pd.DataFrame(list_item)
    pd.set_option('max_colwidth', 190)
    item_set.sort_values(by="product_id", key=lambda x: x.str.len(), ascending=False)

    # Xoá các dòng chỉ có 1 sản phẩm
    item_set = item_set.assign(length=item_set['product_id'].apply(len)).loc[lambda x: x['length'] > 1].drop('length', axis=1)
    item_set.sort_values(by="product_id", key=lambda x: x.str.len(), ascending=False)

    item_set['product_id'] = item_set['product_id'].apply(lambda x: ','.join(x))

    data = list(item_set['product_id'].apply(lambda x:x.split(",") ))

    a = TransactionEncoder()
    a_data = a.fit(data).transform(data)
    df = pd.DataFrame(a_data,columns=a.columns_)
    
    from mlxtend.frequent_patterns import apriori, association_rules
    #set a threshold value for the support value and calculate the support value.
    apriori = apriori(df, min_support = sp/len(df), use_colnames = True, verbose = 1)

    #Let's view our interpretation values using the Associan rule function.
    df_ar = association_rules(apriori, metric = "confidence", min_threshold = 0.6)
    st.write(df_ar)

    return

####################################################################
####################################################################

def Cus_life_time(data, week):

    df = data.copy()
    # tạo bảng rfm từ thư viên
    rfm = summary_data_from_transaction_data(df, customer_id_col='customer_unique_id', datetime_col='order_purchase_timestamp', 
                                    monetary_value_col ='payment_value', observation_period_end='2018-08-29', 
                                    datetime_format='%Y-%m-%d', freq='W')
    # dùng phân phối
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(rfm['frequency'], rfm['recency'], rfm['T'], verbose=True)
    print(bgf)
    
    t = 4 # Weeks for a future transaction 
    rfm['expected_'+str(t)+'week'] = round(bgf.conditional_expected_number_of_purchases_up_to_time(t, rfm['frequency'], rfm['recency'], rfm['T']), 2)
    rfm.sort_values(by='expected_'+str(t)+'week', ascending=False)
    rfm['expected_8week'] = round(bgf.predict(8, rfm['frequency'], rfm['recency'], rfm['T']), 2)
    rfm['expected_12week'] = round(bgf.predict(12, rfm['frequency'], rfm['recency'], rfm['T']), 2)
    rfm['expected_'+str(week) + 'week'] = round(bgf.predict(week, rfm['frequency'], rfm['recency'], rfm['T']), 2)  
    rfm.sort_values(by='expected_'+str(week) + 'week', ascending=False)

    

    rfm_val = calibration_and_holdout_data(df, customer_id_col='customer_unique_id', datetime_col='order_purchase_timestamp', 
                                        monetary_value_col ='payment_value', calibration_period_end='2018-05-29',
                                        observation_period_end='2018-08-29', datetime_format='%Y-%m-%d', freq='W')


    bgf_val = BetaGeoFitter(penalizer_coef=0.001)
    bgf_val.fit(rfm_val['frequency_cal'], rfm_val['recency_cal'], rfm_val['T_cal'], verbose=True)

    # ploting 
    from lifetimes.plotting import plot_period_transactions

    fig, axes = plt.subplots(2, 1, figsize=(12, 16))

    # Biểu đồ 1 - plot_period_transactions
    ax1 = axes[0]
    plot_period_transactions(bgf_val, ax=ax1)
    ax1.set_yscale('log')
    ax1.set_title('Period Transactions')

    # Biểu đồ 2 - plot_calibration_purchases_vs_holdout_purchases
    ax2 = axes[1]
    plot_calibration_purchases_vs_holdout_purchases(model=bgf_val, calibration_holdout_matrix=rfm_val, ax=ax2)
    ax2.set_title('Calibration vs Holdout Purchases')

    st.pyplot(fig)

    # Mo Hình gamma-gamma    
    rfm_gg = rfm[rfm['frequency'] > 0]

    from lifetimes import GammaGammaFitter

    ggf = GammaGammaFitter(penalizer_coef = 0.0)
    ggf.fit(rfm_gg['frequency'], rfm_gg['monetary_value'])

    rfm['avg_transaction'] = round(ggf.conditional_expected_average_profit(rfm_gg['frequency'],
                                                     rfm_gg['monetary_value']), 2)

    rfm['avg_transaction'] = rfm['avg_transaction'].fillna(0)
    rfm.sort_values(by='avg_transaction', ascending=False)

    rfm['CLV'] = round(ggf.customer_lifetime_value(bgf, rfm['frequency'],
                    rfm['recency'], rfm['T'], rfm['monetary_value'],
                    time=week, discount_rate=0.01, freq='W'))
    dp = rfm.sort_values(by='CLV', ascending=False)
    # Display the Table 
    st.write('Customer lifetime')
    st.write(dp)

    # Clustering
    clusters = rfm.drop(rfm.iloc[:, 0:4], axis=1)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(clusters)
    model = KMeans(n_clusters = 3, max_iter = 1000)
    model.fit(scaled)
    labels = model.labels_

    clusters['cluster'] = labels
    clusters.groupby('cluster').agg(['max','min'])['CLV']
    clusters['cluster'].replace(to_replace=[0,1,2], value = ['Non-Profitable', 'Profitable', 'Very Profitable'], inplace=True)
    dp2 = clusters.sort_values(by='CLV', ascending=False)
    st.write('Table After Clustering')
    st.write(dp2)
