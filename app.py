import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

vector = pickle.load(open('vector_conv.sav', 'rb'))

def predict_function(input):
    #change to vect
    rev = []
    content = str(input)
    rev.append(content)
    rev_vec = vector.transform(rev)
    result = loaded_model.predict(rev_vec)
    
    return result

def main():
    st.title("Vũ cute")
    input = st.text_input('Give text')
    if st.button('RESULT'):
        result = predict_function(input)
    st.success(result)



if __name__ == '__main__':
    main()