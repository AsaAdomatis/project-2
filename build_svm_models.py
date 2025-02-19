import pandas as pd
import model_builder as mb
from sklearn.svm import SVC

import threading

def build_svm_rbf():
    svm_rbf = {
        "SVM_rbf": {
            "model": SVC,
            "hyperparameters": {
                'C': [1, 10, 100], # adjusts level of overfitting. higher overfits, lower, under
                'gamma': ['scale', 'auto'], # adjusts complexity of boundaries, high overfits, lower underfits
                'kernel': ['rbf']
            },
            "add_qualitative": True,
            "balancing": "SMOTEEN"
        }
    }
    svm_rbf_df = mb.train_models(models=svm_rbf)
    svm_rbf_df.to_csv("models_svm_rbf.csv")
    print("finished building svm rbf")

def build_svm_poly():
    svm_poly = {
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
        }
    }
    svm_poly_df = mb.train_models(models=svm_poly)
    svm_poly_df.to_csv("models_svm_poly.csv")
    print("finished building svm poly")

def build_svm_linear():
    svm_linear = {
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
    svm_linear_df = mb.train_models(models=svm_linear)
    svm_linear_df.to_csv("models_svm_linear.csv")
    print("finished building svm linear")

if __name__ == "__main__":
    thread1 = threading.Thread(target=build_svm_poly)
    thread2 = threading.Thread(target=build_svm_linear)
    thread3 = threading.Thread(target=build_svm_rbf)

    # Start the threads
    thread1.start()
    thread2.start()
    thread3.start()

    # Optionally, wait for all threads to complete
    thread1.join()
    thread2.join()
    thread3.join()

    print("All functions have completed execution.")