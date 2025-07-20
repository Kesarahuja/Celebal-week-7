# app.py
import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title
st.title("🌼 Iris Flower Classifier")
st.write("Enter flower measurements to predict the species.")

# Sidebar for user input
st.sidebar.header("Input Features")
def get_user_input():
    sepal_length = st.sidebar.slider("Sepal length (cm)", 4.0, 8.0, 5.8)
    sepal_width = st.sidebar.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider("Petal length (cm)", 1.0, 7.0, 4.3)
    petal_width = st.sidebar.slider("Petal width (cm)", 0.1, 2.5, 1.3)

    data = {
        "sepal length (cm)": sepal_length,
        "sepal width (cm)": sepal_width,
        "petal length (cm)": petal_length,
        "petal width (cm)": petal_width
    }
    return pd.DataFrame(data, index=[0])

# Get input
input_df = get_user_input()

# Display input
st.subheader("Input Parameters")
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Show results
species = ['Setosa', 'Versicolor', 'Virginica']
st.subheader("Prediction")
st.write(f"Predicted Class: **{species[prediction[0]]}**")

st.subheader("Prediction Probabilities")
st.bar_chart(prediction_proba[0])
