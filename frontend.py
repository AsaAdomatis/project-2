import streamlit as st
import pandas as pd
import joblib
import scipy.sparse as sp
import numpy as np

# ✅ Load trained model, vectorizer, and label encoders properly
model = joblib.load("logistic_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
label_encoders = joblib.load("label_encoders.pkl")  # This now contains class labels!

# Load the dataset to extract unique values for dropdowns
file_path = "fake_job_postings.csv"  # Ensure the correct file path
df = pd.read_csv(file_path)

# Extract unique values for dropdowns
job_titles = df['title'].dropna().unique().tolist()
company_profiles = df['company_profile'].dropna().unique().tolist()
descriptions = df['description'].dropna().unique().tolist()
requirements_list = df['requirements'].dropna().unique().tolist()
benefits_list = df['benefits'].dropna().unique().tolist()

# ✅ Correct way to extract category names from saved LabelEncoders
employment_types = label_encoders['employment_type']['classes']
required_experiences = label_encoders['required_experience']['classes']
required_educations = label_encoders['required_education']['classes']
industries = label_encoders['industry']['classes']
functions = label_encoders['function']['classes']

# Function to safely transform categorical inputs
def safe_transform(label_encoder_data, value):
    encoder = label_encoder_data["encoder"]
    if value in label_encoder_data["classes"]:
        return encoder.transform([value])[0]
    else:
        return encoder.transform(["Unknown"])[0]  # Fallback to "Unknown" if unseen

# Streamlit UI
st.title("Fake Job Posting Detection")

# Job Title Dropdown
job_title = st.selectbox("Select a Job Title", job_titles)

# Other inputs as dropdowns
company_profile = st.selectbox("Select Company Profile", company_profiles)
description = st.selectbox("Select Job Description", descriptions)
requirements = st.selectbox("Select Job Requirements", requirements_list)
benefits = st.selectbox("Select Job Benefits", benefits_list)

# Convert Yes/No to 0/1
yes_no_mapping = {"Yes": 1, "No": 0}

telecommuting = yes_no_mapping[st.selectbox("Is it a remote job?", ["Yes", "No"])]
has_company_logo = yes_no_mapping[st.selectbox("Does it have a company logo?", ["Yes", "No"])]
has_questions = yes_no_mapping[st.selectbox("Does it ask additional questions?", ["Yes", "No"])]

# ✅ FIX: Extract proper values for Employment Type dropdown
employment_type = st.selectbox("Employment Type", employment_types)
required_experience = st.selectbox("Required Experience", required_experiences)
required_education = st.selectbox("Required Education", required_educations)
industry = st.selectbox("Industry", industries)
function = st.selectbox("Function", functions)

# Process input
text_input = " ".join([job_title, company_profile, description, requirements, benefits])
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
