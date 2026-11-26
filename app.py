
import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="centered"
)

# Test message
st.write("Streamlit is working!")

# App title
st.title("🏡 House Price Prediction")
st.write("Enter the house details below to predict its price.")

# Load the trained model
model_path = os.path.join("models", "house_price_model.pkl")

if not os.path.exists(model_path):
    st.error("Model file not found. Please check the models folder.")
    st.stop()

model = joblib.load(model_path)

# Input fields
st.subheader("House Details")

transaction_date = st.number_input(
    "Transaction Date",
    min_value=2012.0,
    max_value=2014.0,
    value=2013.0
)

house_age = st.number_input(
    "House Age (years)",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)

mrt_distance = st.number_input(
    "Distance to Nearest MRT Station",
    min_value=0.0,
    value=500.0
)

convenience_stores = st.number_input(
    "Number of Convenience Stores",
    min_value=0,
    max_value=20,
    value=5
)

latitude = st.number_input(
    "Latitude",
    value=24.98
)

longitude = st.number_input(
    "Longitude",
    value=121.54
)

# Prediction
if st.button("Predict House Price"):
    input_data = pd.DataFrame([{
        "X1 transaction date": transaction_date,
        "X2 house age": house_age,
        "X3 distance to the nearest MRT station": mrt_distance,
        "X4 number of convenience stores": convenience_stores,
        "X5 latitude": latitude,
        "X6 longitude": longitude
    }])

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted House Price per Unit Area: {prediction:.2f}"
    )
