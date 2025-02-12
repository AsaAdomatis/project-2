import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
file_path = "/Users/swapna/Desktop/project-2/fake_job_postings.csv"
df = pd.read_csv(file_path)

# Prepare the data
features = ["location", "department", "salary_range", "employment_type", "required_experience", "required_education", "industry", "function"]
df = df[["title"] + features + ["fraudulent"]].dropna()

# Encode categorical variables manually (excluding title)
mapping = {}
for col in features:
    df[col] = df[col].astype(str)
    unique_values = df[col].unique()
    mapping[col] = {val: idx for idx, val in enumerate(unique_values)}
    df[col] = df[col].map(mapping[col])

X = df[features]
y = df["fraudulent"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a basic model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.title("Jupyter Notebook Interface - Fake Job Prediction")

#param = st.number_input("Enter parameter", min_value=1, max_value=100, value=10)
selected_job = st.selectbox("Select a Job Listing", df["title"].unique().tolist()[:20])

if st.button("Predict Fake Job Posting"):
    with st.spinner("Processing..."):
        try:
            # Retrieve job features from dataframe
            job_features = df[df["title"] == selected_job][features]
            if not job_features.empty:
                prediction = model.predict(job_features)[0]
                result = "Fake Job Posting" if prediction == 1 else "Real Job Posting"
                st.success(f"Prediction: {result}")
            else:
                st.warning("Selected job title is not in training data, unable to predict.")
        except Exception as e:
            st.error(f"Error running prediction: {e}")
