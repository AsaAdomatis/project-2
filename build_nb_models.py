from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

import model_builder as mb

if __name__ == "__main__":
    models = {
        "complement_bayes": {
            "model": ComplementNB,
            "hyperparameters": {
                "alpha": [0.01, 0.1, 1.0],
                "norm": [True, False]
            },
            "add_qualitative": False,
            "balancing": "SMOTEEN"
        },
        "multinomial_bayes": {
            "model": MultinomialNB,
            "hyperparameters": {
                "alpha": [0.01, 0.1, 1.0]
            },
            "add_qualitative": False,
            "balancing": "SMOTEEN"
        }
    }
    df = mb.train_models(models=models, num_words=5000)
    df.to_csv("model_specifications/models_bayes.csv")