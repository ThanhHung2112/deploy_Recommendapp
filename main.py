import streamlit as st
import pandas as pd
import numpy as np

from alldef import main_reason, predict, draw_nlp_plot, associate, Cus_life_time


option = st.sidebar.selectbox("Select an Option", ["Reviews Comment Analyst", "Association Rule", "Customer Life time"])
file = st.sidebar.file_uploader("Upload a file")
slider_value = st.sidebar.slider("Slider", 0, 100, 50)

st.title("Recommend App")

if file is not None:
    df = pd.read_csv(file, index_col=False)
    st.write("Running Analyst with uploaded file:", file.name)
    st.write(df)
else: st.write("Upload file to start analysis...")

# Xử lý khi nhấn nút Run
if st.sidebar.button("Run"):
    if option == "Reviews Comment Analyst":
        # Thực hiện hàm tương ứng với Reviews Analyst
        if file is not None:
            # Xử lý tệp tin đã tải lên
            df = predict(df)
            # Chạy hàm main_reason
            draw_nlp_plot(df,'Positive' )
            main_reason('Positive', df, stopwords)
            draw_nlp_plot(df,'Negative' )
            main_reason('Negative', df, stopwords)
            
        else:
            st.write("Please upload a file for Reviews Analyst")
    elif option == "Association Rule":
        # Thực hiện hàm tương ứng với Association Rule
        
        if file is not None:
            # Xử lý tệp tin đã tải lên
            st.write("Running Association Rule with uploaded file:", file.name)

            associate (df, slider_value)
        else:
            st.write("Please upload a file for Association Rule")
    elif option == "Customer Life time":
        # Thực hiện hàm tương ứng với Customer Life time
        if file is not None:
            # Xử lý tệp tin đã tải lên
            st.write("Running Customer Life time with uploaded file:", file.name)

            Cus_life_time(df)
        else:
            st.write("Please upload a file for Customer Life time")


