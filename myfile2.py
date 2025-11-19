import pickle
import streamlit as st


model2 = pickle.load(open("Houseprediction.pkl", "rb"))

def mydeploy():
    st.title("House Price Prediction")
    area = st.number_input("Enter Area in square feet:")
    bedrooms = st.number_input("Enter Number of Bedrooms:")     
    age = st.number_input("Enter Age of the House (in years):")
    pred = st.button("Predict Price")
    if pred:
        X = model2.predict([[area, bedrooms, age]])
        st.write("Predicted price of the house is: ₹", X[0])

    
mydeploy()