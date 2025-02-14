import streamlit as st
import joblib
import scipy.sparse as sp
import numpy as np

# Load the trained model and vectorizers
model = joblib.load("logistic_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Function to safely transform categorical inputs
def safe_transform(label_encoder, value):
    if value in label_encoder.classes_:
        return label_encoder.transform([value])[0]
    else:
        return label_encoder.transform(["Unknown"])[0]  # Fallback to "Unknown" if unseen

# Streamlit UI
st.title("Fake Job Posting Detection")

# User inputs
title = st.text_input("Job Title")
company_profile = st.text_area("Company Profile")
description = st.text_area("Job Description")
requirements = st.text_area("Job Requirements")
benefits = st.text_area("Job Benefits")
telecommuting = st.selectbox("Is it a remote job?", [0, 1])
has_company_logo = st.selectbox("Does it have a company logo?", [0, 1])
has_questions = st.selectbox("Does it ask additional questions?", [0, 1])
employment_type = st.selectbox("Employment Type", label_encoders['employment_type'].classes_)
required_experience = st.selectbox("Required Experience", label_encoders['required_experience'].classes_)
required_education = st.selectbox("Required Education", label_encoders['required_education'].classes_)
industry = st.selectbox("Industry", label_encoders['industry'].classes_)
function = st.selectbox("Function", label_encoders['function'].classes_)

# Process input
text_input = " ".join([title, company_profile, description, requirements, benefits])
text_vector = tfidf.transform([text_input])  # Transform text input using TF-IDF

# Convert categorical values using the safe_transform function
categorical_values = [
    safe_transform(label_encoders['employment_type'], employment_type),
    safe_transform(label_encoders['required_experience'], required_experience),
    safe_transform(label_encoders['required_education'], required_education),
    safe_transform(label_encoders['industry'], industry),
    safe_transform(label_encoders['function'], function)
]

# Combine all inputs
numerical_input = np.array([[telecommuting, has_company_logo, has_questions] + categorical_values])
numerical_sparse = sp.csr_matrix(numerical_input)  # Convert to sparse matrix

# Combine text and numerical features
X_input = sp.hstack([text_vector, numerical_sparse])

# Predict
prediction = model.predict(X_input)[0]

# Output result
if prediction == 1:
    st.error("⚠️ This job posting is likely **fraudulent**.")
else:
    st.success("✅ This job posting seems **legitimate**.")
