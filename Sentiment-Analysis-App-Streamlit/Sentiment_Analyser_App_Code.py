import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

st.title('Sentiment Analyser App')
st.write('This app uses Logistic Regression and Support Vector Machine Algorithm to analyze the sentiment of reviews')

st.write('Select the algorithm -')
algo = st.selectbox('Algorithm', ('Logistic Regression', 'SVM'))

form = st.form(key='sentiment-form')
user_input = form.text_area('Enter your text')

submit = form.form_submit_button('Submit')

if submit:

    user_input = [user_input]

    vectorizer = CountVectorizer(max_features=2000)
    user_input_to_model = vectorizer.fit_transform(user_input)

    user_input_to_model = user_input_to_model.toarray()

    user_input_to_model.resize(len(user_input_to_model), 2000, refcheck=False)

    if algo == 'Logistic Regression':
        with open('sa_lr_model.pkl', 'rb') as f:
            clf = pickle.load(f)
    elif algo == 'SVM':
        with open('sa_svc_model.pkl', 'rb') as f:
            clf = pickle.load(f)

    result = clf.predict(user_input_to_model)

    if result == 0:
        st.success('Negative Sentiment')
    elif result == 1:
        st.error('Positive Sentiment')
