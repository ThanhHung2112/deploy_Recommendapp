import streamlit as st
import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words("portuguese")

import lifetime
from datetime import datetime
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter
from lifetimes.utils import calibration_and_holdout_data
from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases
from sklearn.preprocessing import StandardScaler
# Định nghĩa hàm main_reason(label, data, stopwords)


def predict(data):

    loaded_model = pickle.load(open('trained_model.sav', 'rb'))
    vector = pickle.load(open('vector_conv.sav', 'rb'))

    # Chuẩn bị dữ liệu cho việc tìm lý do chính
    df_ans = data.copy()
    df_ans['Sentiments'] = ''  # Thêm cột 'Sentiments' rỗng
    review_comment_messages = df_ans['review_comment_message'].tolist()
    vectors = vector.transform(review_comment_messages)

    # Dự đoán nhãn 'Positive' hay 'Negative'
    predicted_labels = loaded_model.predict(vectors)
    df_ans['Sentiments'] = predicted_labels

    return df_ans

def main_reason(label, data , stopwords):

    # Load mô hình đã huấn luyện
    df_ans = data.copy()

    # Tìm lý do chính
    sentiments = df_ans[df_ans['Sentiments'] == label]['review_comment_message']
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentiments)

    # Hiện bảng
    df_label = df_ans[df_ans['Sentiments'] == label]
    st.header('Reviews with' + str(label)+ 'sentiments:')
    st.write(df_label)

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
        closest_sentences = [sentiments.iloc[idx] for idx in cluster_indices[closest_indices]]
        similar_sentences.extend(closest_sentences)

    # Hiển thị các câu gần với tâm của từng cụm
    st.write("Display sentences that are close to the meaning of each cluster:")
    for i, sentence in enumerate(similar_sentences):
        st.write(sentence)

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

def Cus_life_time(data):

    df = data.copy()

    rfm = summary_data_from_transaction_data(df, customer_id_col='customer_unique_id', datetime_col='order_purchase_timestamp', 
                                    monetary_value_col ='payment_value', observation_period_end='2018-08-29', 
                                    datetime_format='%Y-%m-%d', freq='W')
    
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(rfm['frequency'], rfm['recency'], rfm['T'], verbose=True)
    print(bgf)

    t = 4 # Weeks for a future transaction 
    rfm['expected_'+str(t)+'week'] = round(bgf.conditional_expected_number_of_purchases_up_to_time(t, rfm['frequency'], rfm['recency'], rfm['T']), 2)
    rfm.sort_values(by='expected_'+str(t)+'week', ascending=False)
    rfm['expected_8week'] = round(bgf.predict(8, rfm['frequency'], rfm['recency'], rfm['T']), 2)
    rfm['expected_12week'] = round(bgf.predict(12, rfm['frequency'], rfm['recency'], rfm['T']), 2)  
    rfm.sort_values(by='expected_12week', ascending=False)

    

    rfm_val = calibration_and_holdout_data(df, customer_id_col='customer_unique_id', datetime_col='order_purchase_timestamp', 
                                        monetary_value_col ='payment_value', calibration_period_end='2018-05-29',
                                        observation_period_end='2018-08-29', datetime_format='%Y-%m-%d', freq='W')


    st.write(rfm_val.head(5))

    bgf_val = BetaGeoFitter(penalizer_coef=0.001)
    bgf_val.fit(rfm_val['frequency_cal'], rfm_val['recency_cal'], rfm_val['T_cal'], verbose=True)

    st.write(bgf_val)

    fig = plt.figure(figsize=(12,8))
    plot_calibration_purchases_vs_holdout_purchases(model=bgf_val, calibration_holdout_matrix=rfm_val)
    st.pyplot(plt)
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
                    time=26, discount_rate=0.01, freq='W'))
    dp = rfm.sort_values(by='CLV', ascending=False)
    # Display the Table 
    st.write(dp)

    # Clustering
    clusters = rfm.drop(rfm.iloc[:, 0:4], axis=1)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(clusters)
    scaled
    model = KMeans(n_clusters = 3, max_iter = 1000)
    model.fit(scaled)
    labels = model.labels_

    clusters['cluster'] = labels
    clusters.groupby('cluster').agg(['max','min'])['CLV']
    clusters['cluster'].replace(to_replace=[0,1,2], value = ['Non-Profitable', 'Very Profitable', 'Profitable'], inplace=True)
    dp2 = clusters.sort_values(by='CLV', ascending=False)

    st.write(dp2)