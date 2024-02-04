import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the pre-trained model and scaler
loaded_model = pickle.load(open('bank_churn_model.pkl', 'rb'))
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define feature names
feature_names = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
                 'Geography_Germany', 'Geography_Spain', 'Gender_Male', 'HasCrCard_1', 'IsActiveMember_1']

# Function to preprocess input data
def preprocess_input_data(input_data, scaler):
    # Create a DataFrame with feature names
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Use the loaded scaler to transform the input data
    input_data_scaled = scaler.transform(input_df.values)

    return input_data_scaled

# Function to predict churn
def predict_churn(input_data):
    prediction = loaded_model.predict(input_data)
    return prediction

# Set up the Streamlit app
st.set_page_config(layout="wide")
st.title('Bank Churn Prediction 💸')

# User input for prediction
with st.form(key='input_form'):
    st.header('Enter Customer Information for Churn Prediction')
    col1, col2 = st.columns([3, 2])  # Adjusted the width ratio here

    with col1:
        credit_score = st.slider('Credit Score', 300, 850, step=1)
        age = st.slider('Age', 18, 100, step=1)
        tenure = st.slider('Tenure', 0, 20, step=1)
        balance = st.slider('Balance', 0, 250000, step=1)
        num_of_products = st.slider('Number of Products', 1, 4, step=1)
        estimated_salary = st.slider('Estimated Salary', 0, 250000, step=1)

    with col2:
        geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
        gender = st.selectbox('Gender', ['Male', 'Female'])
        has_cr_card = st.checkbox('Has Credit Card')
        is_active_member = st.checkbox('Is Active Member')

    submit_button = st.form_submit_button(label='Predict Churn')

# Prediction and result display
if submit_button:
    input_data = [credit_score, age, tenure, balance, num_of_products, estimated_salary]

    if geography == 'France':
        input_data.extend([1, 0])
    elif geography == 'Spain':
        input_data.extend([0, 1])
    else:
        input_data.extend([0, 0])

    if gender == 'Male':
        input_data.append(1)
    else:
        input_data.append(0)

    if has_cr_card:
        input_data.append(1)
    else:
        input_data.append(0)

    if is_active_member:
        input_data.append(1)
    else:
        input_data.append(0)

    # Preprocess input data using the loaded scaler
    input_data_processed = preprocess_input_data(input_data, loaded_scaler)

    # Display processed input data in a table
    st.subheader('Processed Input Data')
    processed_input_df = pd.DataFrame([input_data_processed.flatten()], columns=feature_names)
    st.write(processed_input_df)

    # Predict churn
    prediction = predict_churn(input_data_processed)

    # Display result
    if prediction[0] == 0:
        result = 'Customer will not churn.'
        result_color = 'green'
    else:
        result = 'Customer will churn.'
        result_color = 'red'

    # Display the final result with faded blue texture
    with st.spinner('Predicting...'):
        st.subheader('Prediction Result:')
        result_html = f'<div style="background-color: #D3D3D3; padding: 10px; border-radius: 5px;"><span style="color:{result_color}; font-size: larger; font-weight: bold;">{result}</span></div>'
        st.markdown(result_html, unsafe_allow_html=True)

# Add some descriptions and instructions
st.markdown('---')
st.subheader('Instructions')
st.markdown('1. Adjust the sliders and select boxes to enter customer information.')
st.markdown('2. Click the "Predict Churn" button to see the prediction.')

st.markdown('---')
st.subheader('About')
st.markdown('This web app predicts whether a customer will churn from the bank based on their information.')