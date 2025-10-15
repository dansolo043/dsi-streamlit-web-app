
#importing important packages

import streamlit as st
import pandas as pd
import joblib

# load our model pipeline object

model = joblib.load("model.joblib")

# Add a title and instructions 

st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit for liklihood to purchase")

# age input

age = st.number_input(label="01. Enter the customer's age", min_value=18,
                      max_value=120, value=35)


# gender input

gender = st.radio(label="02. Enter the customer's gender", options=["M", "F"])

# credit score input

credit_score = st.number_input(label="03. Enter the customer's credit score", min_value=0,
                      max_value=1000, value=500)

# submitting the given information

if st.button("Submit For Prediction"):
    #store the data in a data frame for prediction
    
    new_data = pd.DataFrame({"age":[age], "gender":[gender], "credit_score":[credit_score]})
    
    # apply the model pipeline to the input data and extract the probability prediction
    
    pred_proba = model.predict_proba(new_data)[0][1]
    
    # output prediction (so that we get a whole number we use .0)
    
    st.subheader(f"Based on these customer attributes, out model predicts a purchase \
                 probability of {pred_proba:.0%}")

    
