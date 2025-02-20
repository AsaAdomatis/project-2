import streamlit as st
import pandas as pd
import joblib
import scipy.sparse as sp
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer

import preprocessing as prep

# Streamlit UI
st.title("Fake Job Posting Detection")

# ✅ Load available models dynamically
model_files = [f for f in os.listdir("pickles") if f.endswith("_model.pkl") or f.endswith("_grid_search.pkl")]
selected_model_file = st.selectbox("Select a Model", model_files)
model = joblib.load("pickles/" + selected_model_file)

# Handle GridSearchCV model if necessary
if hasattr(model, 'best_estimator_'):
    model = model.best_estimator_

# Load corresponding accuracy file if exists
accuracy_file = "pickles/" + selected_model_file.replace("_model.pkl", "_accuracy.pkl")
gs_accuracy_file = "pickles/" + selected_model_file.replace("_grid_search.pkl", "_accuracy.pkl")
if "_grid_search" not in selected_model_file and os.path.exists(accuracy_file):
    model_accuracy = joblib.load(accuracy_file)
elif os.path.exists(gs_accuracy_file):
    model_accuracy = joblib.load(gs_accuracy_file)
else:
    model_accuracy = None

# Loading corresponding f1 score file if exists
f1_file = "pickles/" + selected_model_file.replace("_model.pkl", "_f1_score.pkl")
gs_f1_file = "pickles/" + selected_model_file.replace("_grid_search.pkl", "_f1_score.pkl")
if "_grid_search" not in selected_model_file and os.path.exists(f1_file):
    model_f1 = joblib.load(f1_file)
elif os.path.exists(gs_f1_file):
    model_f1 = joblib.load(gs_f1_file)
else:
    model_f1 = None

# Loading default vectorizer
if not ("bayes" in selected_model_file or "svm" in selected_model_file):
    # trying to load default pickle
    tfidf = joblib.load("pickles/default_vectorizer.pkl")

    # fitting vectorizer if unfitted
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
        joblib.dump(tfidf, "pickles/default_vectorizer.pkl")  # Save fitted vectorizer
        st.success("TF-IDF vectorizer successfully fitted and saved.")

# Four required inputs as text inputs
job_title = st.text_input("Enter the Job Title")
company_profile = st.text_input("Enter Company Profile")
description = st.text_input("Enter Job Description")
requirements = st.text_input("Enter Job Requirements")

# Show prediction only if all fields are filled
if all([job_title.strip(), company_profile.strip(), description.strip(), requirements.strip()]):
    # Process input without cleaning
    raw_text = " ".join([job_title, company_profile, description, requirements])
    # st.text(f"Raw Text Preview: {raw_text[:200]}...")  # Debug output 

    # preprocessing (only for bayes and svm)
    if "bayes" in selected_model_file or "svm" in selected_model_file:
        if "bayes" in selected_model_file:
            prep.vectorizer = joblib.load("pickles/bayes_vectorizer.pkl")
        elif "svm" in selected_model_file:
            prep.vectorizer = joblib.load("pickles/bayes_vectorizer.pkl")
        X_input = prep.preprocess(text=raw_text)

    # preprocessing for rfc model
    elif "rfc" in selected_model_file:
        cleaned_text = prep.clean_text(raw_text)
        vectorizer = joblib.load("pickles/rfc_vectorizer.pkl")
        X_input = vectorizer.transform([cleaned_text]).toarray()

    # preprocessing for defaults (logistic model)
    else:
        try:
            text_vector = tfidf.transform([raw_text])  # Transform raw text input using TF-IDF
            # st.text(f"Transformed Vector Shape: {text_vector.shape}")  # Debug output # FIX: Remove when done
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
                #st.info(f"ℹ️ Added {missing_features} zero-filled features to match model input size.")
            elif current_features > expected_features:
                X_input = text_vector[:, :expected_features]
                st.info(f"ℹ️ Trimmed extra features to match model input size.")
            else:
                X_input = text_vector

    # Trying and displaying predictions
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

        if model_f1 is not None:
            st.info(f"📊 The selected model's f1 score for fraudulent data is: {model_f1 * 100:.2f}%")
        else:
            st.warning("⚠️ f1 scoring for the selected model is not available.")

    except ValueError as e:
        st.error(f"⚠️ Prediction failed due to feature mismatch: {e}")
    except AttributeError as e:
        st.error(f"⚠️ Prediction failed due to attribute error: {e}")
else:
    st.info("ℹ️ Please fill in all details to see the prediction result.")
