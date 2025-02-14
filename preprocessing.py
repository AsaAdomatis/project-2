from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import re

from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

import nltk
from nltk.corpus import stopwords

# cleaning text data
nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))

def clean_text(text:str):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)  
        cleaned_text = " ".join(word for word in text.split() if word not in stop_words)
        return cleaned_text

def preprocess(data:pd.DataFrame, vectorizer_type:str="tfidf", num_words:int=5000, 
               add_qualitative:bool=False, convert_nulls:bool=False) -> pd.DataFrame:
    # cleaning texts
    data["text"] = data["title"] + " " + data["company_profile"] + " " + data["description"] + " " + data["requirements"]
    data["text"].fillna("", inplace=True)
    data["text_cleaned"] = data["text"].apply(clean_text)

    # vectorizing the data
    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(max_features=num_words)
        X = vectorizer.fit_transform(data["text_cleaned"])
    elif vectorizer_type == "count":
        vectorizer = CountVectorizer(max_features=num_words)
        X = vectorizer.fit_transform(data["text_cleaned"])
    else:
        raise ValueError(f"Invalid vectorizer: {vectorizer_type}. Choose from 'tdif' or 'count'!")
    X = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())


    # adding in qualitative data with dummies
    if add_qualitative:
        X = pd.merge(right=data[["employment_type", "required_experience", "required_education"]],
                left=X, left_index=True, right_index=True)
        X = pd.get_dummies(X, columns=["employment_type", "required_experience", "required_education"], dtype=int)

    # convert nulls
    if convert_nulls:
        has_columns = ["location", "company_profile", "salary_range", "benefits"]
        for col in has_columns:
            X[f"has_{col}"] = data[col].notnull().astype(int)

    return X


def train_test_split(balancing:str = None, vectorizer_type:str="tfidf"):
    # cleaning initial data
    data = pd.read_csv("fake_job_postings.csv")

    try:
        data.drop(inplace=True, columns="Unnamed: 0", axis=1)
    except KeyError:
        pass #oopsy!

    X = data.drop(columns="fraudulent", axis=1)
    y = data["fraudulent"]

    X = preprocess(X, vectorizer_type=vectorizer_type)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # balancing the data set
    resampler = None
    if balancing is None:
        pass
    elif balancing == "SMOTEEN":
        resampler = SMOTEENN(random_state=42)
    elif balancing == "RandomUnder":
        resampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    elif balancing == "RandomOver":
        resampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    if resampler is not None:
        X_train, y_train = resampler.fit_resample(X_train, y_train)

    return  X_train, X_test, y_train, y_test