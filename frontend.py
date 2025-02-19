import streamlit as st
import pandas as pd
import joblib
import scipy.sparse as sp
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Load available models dynamically
model_files = [f for f in os.listdir() if f.endswith("_model.pkl")]
selected_model_file = st.selectbox("Select a Model", model_files)
model = joblib.load(selected_model_file)

# Handle GridSearchCV model if necessary
if hasattr(model, 'best_estimator_'):
    model = model.best_estimator_

# Load corresponding accuracy file if exists
accuracy_file = selected_model_file.replace("_model.pkl", "_accuracy.pkl")
if os.path.exists(accuracy_file):
    model_accuracy = joblib.load(accuracy_file)
else:
    model_accuracy = None

# Load vectorizer and ensure it is fitted
tfidf = joblib.load("sara_tfidf_vectorizer.pkl")
if not hasattr(tfidf, 'idf_'):
    st.warning("TF-IDF vectorizer not fitted. Fitting now...")
    df = pd.read_csv("fake_job_postings.csv")
    corpus = (
        df['title'].fillna('') + ' ' +
        df['company_profile'].fillna('') + ' ' +
        df['description'].fillna('') + ' ' +
        df['requirements'].fillna('')
    ).tolist()
    tfidf.fit(corpus)
    joblib.dump(tfidf, "sara_tfidf_vectorizer.pkl")  # Save fitted vectorizer
    st.success("TF-IDF vectorizer successfully fitted and saved.")

# Streamlit UI
st.title("Fake Job Posting Detection")

# Display team pictures
st.header("Meet Our Team")
#st.image("team_picture_1.jpg", caption="Team Member 1", use_column_width=True)
#st.image("team_picture_2.jpg", caption="Team Member 2", use_column_width=True)
#t.image("team_picture_3.jpg", caption="Team Member 3", use_column_width=True)
#st.image("team_picture_4.jpg", caption="Team Member 4", use_column_width=True)
#st.image("team_picture_5.jpg", caption="Team Member 5", use_column_width=True)

# Four required inputs as text inputs
job_title = st.text_input("Enter the Job Title")
company_profile = st.text_input("Enter Company Profile")
description = st.text_input("Enter Job Description")
requirements = st.text_input("Enter Job Requirements")

# Show prediction only if all fields are filled
if all([job_title.strip(), company_profile.strip(), description.strip(), requirements.strip()]):
    # Process input without cleaning
    raw_text = " ".join([job_title, company_profile, description, requirements])
    st.text(f"Raw Text Preview: {raw_text[:200]}...")  # Debug output

    try:
        text_vector = tfidf.transform([raw_text])  # Transform raw text input using TF-IDF
        st.text(f"Transformed Vector Shape: {text_vector.shape}")  # Debug output
    except ValueError as e:
        st.error(f"⚠️ TF-IDF transformation failed: {e}")
        text_vector = None

    if text_vector is not None:
        # ✅ Fix feature mismatch: Add zero-filled features if needed
        try:
            expected_features = model.n_features_in_
        except AttributeError:
            st.error("⚠️ The model does not have 'n_features_in_' attribute. Check if it is compatible.")
            expected_features = text_vector.shape[1]

        current_features = text_vector.shape[1]

        if current_features < expected_features:
            missing_features = expected_features - current_features
            padding = sp.csr_matrix(np.zeros((1, missing_features)))
            X_input = sp.hstack([text_vector, padding])
            st.info(f"ℹ️ Added {missing_features} zero-filled features to match model input size.")
        elif current_features > expected_features:
            X_input = text_vector[:, :expected_features]
            st.info(f"ℹ️ Trimmed extra features to match model input size.")
        else:
            X_input = text_vector

        # Predict
        try:
            prediction = model.predict(X_input)[0]

            # Output result
            if prediction == 1:
                st.error("⚠️ This job posting is likely **fraudulent**.")
            else:
                st.success("✅ This job posting seems **legitimate**.")

            # Show model accuracy if available
            if model_accuracy is not None:
                st.info(f"📊 The selected model's accuracy rate is: {model_accuracy * 100:.2f}%")
            else:
                st.warning("⚠️ Accuracy rate for the selected model is not available.")

        except ValueError as e:
            st.error(f"⚠️ Prediction failed due to feature mismatch: {e}")
        except AttributeError as e:
            st.error(f"⚠️ Prediction failed due to attribute error: {e}")
else:
    st.info("ℹ️ Please fill in all details to see the prediction result.")
