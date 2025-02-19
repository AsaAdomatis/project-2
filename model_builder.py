import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV

# model imports
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

import preprocessing as prep



def train_models(models:dict=None, verbose:bool=True):
    """
    I want to build a table of the best models
    """
    if models is None:
        models = {
            "complement_bayes": {
                "model": ComplementNB,
                "hyperparameters": {
                    "alpha": [0.01, 0.1, 1.0],
                    "norm": [True, False]
                },
                "add_qualitative": False,
                "balancing": None
            },
            "multinomial_bayes": {
                "model": MultinomialNB,
                "hyperparameters": {
                    "alpha": [0.01, 0.1, 1.0]
                },
                "add_qualitative": False,
                "balancing": "SMOTEEN"
            },
            "SVM_rbf": {
                "model": SVC,
                "hyperparameters": {
                    'C': [1, 10, 100], # adjusts level of overfitting. higher overfits, lower, under
                    'gamma': ['scale', 'auto'] # adjusts complexity of boundaries, high overfits, lower underfits
                },
                "add_qualitative": True,
                "balancing": "SMOTEEN"
            },
            "SVM_poly": {
                "model": SVC,
                "hyperparameters": {
                    'coef0': [0, 1],
                    'degree': [2, 3, 5],
                    'C': [1, 10, 100, 500, 1000], # adjusts level of overfitting. higher overfits, lower, under
                    'gamma': ['scale', 'auto'] # adjusts complexity of boundaries, high overfits, lower underfits
                },
                "add_qualitative": True,
                "balancing": "SMOTEEN"
            },
            "SVM_linear": {
                "model": SVC,
                "hyperparameters": {
                    'C': [1, 10, 100], # adjusts level of overfitting. higher overfits, lower, under
                    'gamma': ['scale', 'auto'], # adjusts complexity of boundaries, high overfits, lower underfits
                    'kernel': ['linear']
                },
                "add_qualitative": True,
                "balancing": "SMOTEEN"
            }
        }

    model_dicts = []
    for name, value in models.items():
        if verbose:
            print(f"Training {name} for hyperparameters: {value["hyperparameters"]}")

        # get train and test
        X_train, X_test, y_train, y_test = prep.custom_train_test_split(balancing=value["balancing"],
                                                                        add_qualitative=value["add_qualitative"])
        
        # definding custom scorer for f1 of fraudelent jobs
        f1_1_scorer = make_scorer(f1_score, average='binary', pos_label=1)
        # use grid search to find the best parameters
        grid_search = GridSearchCV(value["model"](), value["hyperparameters"], scoring=f1_1_scorer)
        grid_search.fit(X_train, y_train)

        # training the better model
        model = value["model"](**grid_search.best_params_)
        model.fit(X_train, y_train)

        # storing the final report
        y_pred = model.predict(X_test)
        model_dict = classification_report(y_test, y_pred, output_dict=True)
        model_dict["name"] = name
        model_dict["best_params"] = grid_search.best_params_
        model_dicts.append(model_dict)

        if verbose:
            print(f"Output: {model_dict}")

    return pd.DataFrame(model_dicts)

if __name__ == "__main__":
    df = train_models()
    df.to_csv("models_final.csv")



