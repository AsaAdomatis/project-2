import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

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

# Normalize the features
scaler = MinMaxScaler()
X = scaler.fit_transform(df[features])
y = df["fraudulent"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

st.title("Jupyter Notebook Interface - Fake Job Prediction using Linear Regression")

selected_job = st.selectbox("Select a Job Listing", df["title"].unique().tolist()[:20])

if st.button("Predict Fake Job Posting"):
    with st.spinner("Processing..."):
        try:
            # Retrieve job features from dataframe
            job_features = df[df["title"] == selected_job][features]
            if not job_features.empty:
                job_features_scaled = scaler.transform(job_features)
                prediction = model.predict(job_features_scaled)[0]
                result = f"Predicted Fraud Score: {prediction:.2f} (Closer to 1 = More Fake, Closer to 0 = Real)"
                st.success(result)
            else:
                st.warning("Selected job title is not in training data, unable to predict.")
        except Exception as e:
            st.error(f"Error running prediction: {e}")
