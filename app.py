#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Function to preprocess input data
def preprocess_input(data):
    # Example preprocessing: scaling numeric inputs
    scaler = StandardScaler()
    numeric_cols = ['site_size', 'num_doctors', 'num_nurses', 'avg_patient_age',
                    'avg_patient_income', 'trial_success_rate', 'patient_enrollment_rate',
                    'previous_experience', 'site_rating', 'accessibility_score',
                    'bed_count', 'years_operational', 'training_quality', 'site_capacity']
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

# Function to predict success rate
def predict_success_rate(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Load the trained model
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    return joblib.load('best_model.pkl')

# Main function to run the app
def main():
    st.title('Predicting Clinical Site Success Rate')
    st.write('Enter the details below to predict the success rate of a clinical site:')
    
    # Collect inputs from user
    site_id = st.text_input('Site ID')
    region = st.selectbox('Region', ['North', 'South', 'East', 'West'])
    site_size = st.number_input('Site Size')
    num_doctors = st.number_input('Number of Doctors')
    num_nurses = st.number_input('Number of Nurses')
    avg_patient_age = st.number_input('Average Patient Age')
    avg_patient_income = st.number_input('Average Patient Income')
    trial_success_rate = st.number_input('Trial Success Rate (%)')
    patient_enrollment_rate = st.number_input('Patient Enrollment Rate (%)')
    previous_experience = st.number_input('Previous Experience (years)')
    site_rating = st.number_input('Site Rating')
    accessibility_score = st.number_input('Accessibility Score')
    site_type = st.selectbox('Site Type', ['Type A', 'Type B', 'Type C'])
    bed_count = st.number_input('Bed Count')
    years_operational = st.number_input('Years Operational')
    training_quality = st.number_input('Training Quality')
    site_capacity = st.number_input('Site Capacity')
    
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'site_size': [site_size],
        'num_doctors': [num_doctors],
        'num_nurses': [num_nurses],
        'avg_patient_age': [avg_patient_age],
        'avg_patient_income': [avg_patient_income],
        'trial_success_rate': [trial_success_rate],
        'patient_enrollment_rate': [patient_enrollment_rate],
        'previous_experience': [previous_experience],
        'site_rating': [site_rating],
        'accessibility_score': [accessibility_score],
        'bed_count': [bed_count],
        'years_operational': [years_operational],
        'training_quality': [training_quality],
        'site_capacity': [site_capacity]
    })
    
    # Preprocess input data
    input_data = preprocess_input(input_data)
    
    # Load the model
    model = load_model()
    
    # Make prediction
    if st.button('Predict Success Rate'):
        prediction = predict_success_rate(model, input_data)
        st.success(f'Predicted Success Rate: {prediction[0]}')
    
if __name__ == '__main__':
    main()


# In[ ]:




