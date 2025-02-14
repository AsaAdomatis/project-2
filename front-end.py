import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# Load dataset
file_path = "/Users/swapna/Desktop/project-2/fake_job_postings.csv"
df = pd.read_csv(file_path)

# Prepare the data
features = ["location", "department", "salary_range", "employment_type", "required_experience", "required_education", "industry", "function"]
df = df[["title"] + features + ["fraudulent"]].dropna()

# Encode categorical variables manually
for col in features:
    df[col] = df[col].astype(str).factorize()[0]

# Normalize the features
scaler = MinMaxScaler()
X = scaler.fit_transform(df[features])
y = df["fraudulent"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
model_options = {
    'Logistic Regression': LogisticRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestClassifier()
}
selected_models = st.multiselect("Select Models", list(model_options.keys()), default=list(model_options.keys()))
selected_models_instances = {name: model_options[name] for name in selected_models}

for model in selected_models_instances.values():
    model.fit(X_train, y_train)

def predict_fake_job(selected_job):
    predictions = {}
    job_features = df[df["title"] == selected_job][features]
    if not job_features.empty:
        job_features_scaled = scaler.transform(job_features)
        for name, model in selected_models_instances.items():
            predictions[name] = model.predict(job_features_scaled)[0]
        return predictions
    return {"Error": "Selected job title is not in training data, unable to predict."}

titles = df["title"].unique().tolist()[:20]
st.title("Fake Job Posting Prediction")

selected_job = st.selectbox("Select a Job Listing", titles)

if st.button("Predict Fake Job Posting"):
    with st.spinner("Processing..."):
        result = predict_fake_job(selected_job)
    
    col1, col2 = st.columns(2)
    with col1:
        for model_name, pred in result.items():
            st.subheader(model_name)
            st.write(f"Prediction: {pred} (1 = Fake, 0 = Real or Score based)")

st.write("---")
st.write("This app allows users to compare different models for detecting fake job postings.")
