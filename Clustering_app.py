import streamlit as st
import pickle
import pandas as pd
import numpy as np

df = pd.DataFrame()


st.markdown("<h1 style='text-align: center; color: red;'>Discover Your Shopper Identity</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>What kind of customer are you?</h3>", unsafe_allow_html=True)

with open('scaler.pk', 'rb') as file:
    scaler = pickle.load(file)

with open('encoder.pk', 'rb') as file:
    encoder = pickle.load(file)

with open('clusters_kmean.pk', 'rb') as file:
    clusters_names_kmean = pickle.load(file)

with open('kmeans.pk', 'rb') as file:
    model = pickle.load(file)

mapper = clusters_names_kmean
st.write("--"*2)

st.markdown("<h4 style='text-align: left; color: black;'>Select the features that describe you:</h4>", unsafe_allow_html=True)
df['Gender'] = [st.selectbox(label='Gender', options=['Male', 'Female'])]

df['Age'] = [st.number_input("Age", min_value=18, max_value=70)]

df['Annual Income (k$)'] = st.number_input("What is your annual Income in Thousand dollars: Range between '15K - 120K'",
                                           min_value=15.0, max_value=120.0, step=1.0)
df['Spending Score (1-100)'] = st.select_slider("If you have $100 in your pocket, how much of it would you spend at once?",
                                                options=(np.arange(1, 101, 1)))
st.write("--"*2)

button = st.button("Predict your Shopping group", type='secondary')
if button:
    num_cols = df.select_dtypes('number').columns
    df[num_cols] = scaler.transform(df[num_cols])

    df['Gender'] = encoder.transform(df['Gender'])

    st.write(mapper[model.predict(df)[0]])
