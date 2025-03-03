# Project-2
![Image](https://github.com/user-attachments/assets/1d216964-044f-43e2-a8d7-cb44aef67635)

Team Members: Asa Adomatus, Kevin Miller, Sara Moujahed, Xavier Figueroa, Laxmi Atluri, Pablo Romero

[Presentation Link](https://docs.google.com/presentation/d/13SDhB6MA-34hK0nuFP5HCgxqOKQDp2U8RFowtQUadM4/edit#slide=id.g3339bb8fa76_0_24)

## Overview:
This repository is for Team10 Project 2\. 

In today's digital age, online job platforms have become a vital resource for job seekers and employers alike. However, with the convenience of these platforms comes the growing problem of fraudulent job postings. These fake listings not only waste time for job seekers but can also lead to scams, identity theft, or financial loss. Detecting fake job postings is a critical task for ensuring the safety and trustworthiness of job platforms.

The purpose of our project aims to build a machine learning (ML) model that can automatically detect fraudulent job postings by analyzing patterns in the text and metadata of job listings. Our model will classify job postings as either legitimate or fake.

## **Table of Contents:**

1. Installation  
2. Usage  
3. Data  
4. Models  
5. Evaluation  
6. Results
7. Demo / UI Frontend Application
8. Project Requirements.

## **Installation**

You can close the following repository that will help you to open our code in your favorite Dev Tools

1. Clone this repository::  
   

| Git clone https://github.com/AsaAdomatis/project-2.git |
| :---- |

2. Install the required dependencies

## **Usage**

Our models have been training in a large dataset of fake job postings that has recordings of fake and real postings on the internet.

For anyone interested in running our model on your local computer, you can follow the instructions above to clone the repository, install the appropriate dependencies and import all the important dependencies a shown below:

![Image](https://github.com/user-attachments/assets/ef3e8037-77db-476f-b6c9-7cc3699611ea)

![][image1]

## **Data**

The dataset used for this project comes from the Kaggle Faje\_job\_postings dataset. Due to an imbalanced dataset we have worked through preprocessing and cleaning out data, many values were transformed using Vectorizer. It was preprocessed by removing missing values, encoding categorical features and scaling Numerical features.

URL: [https://www.kaggle.com/datasets/srisaisuhassanisetty/fake-job-postings](https://www.kaggle.com/datasets/srisaisuhassanisetty/fake-job-postings)

## **Models**

Based on the objective of our project we have selected the following Classification models:

* Support Vector Machine (SVM)  
* Naive Bayes  
* Logistic Regression  
* DecisionTree  
* Random Forest


All models were trained and tested against each other to find the best outcome and our premium model that will help us to achieve our goal for predicting fake job postings. 

## **Evaluation**

The models were evaluated using  accuracy, precision, recall and F1-score. The model outperformed others is Logistic Regression with a F1-score for fraudulent jobs of 0.99% and a weighted avg of 99%

![Image](https://github.com/user-attachments/assets/c0d1665d-208a-41db-8b48-b20298db392a)

## **Results**

After training and evaluating several models, our Logistic Regression model has the best performance when identifying real vs fake job postings as it's accurate by 0,99% using F1-score with a weighted avarage of 99%
The model was successfully able to predict a fake job posting with high accuracy.

## **Demo / UI Frontend Application**

Out Team has developed an UI frontend demo using streamlit that shows all our coding in action, it take different parameters as input that our model analyzes and gives an accurate prediction about the job posting either legitimate or fake.

## **Project Requirements**

Requirements Data Model Implementation (25 points) There is a Jupyter notebook that thoroughly describes the data extraction, cleaning, and transformation process, and the cleaned data is exported as CSV files for the machine learning model. (10 points)

A Python script initializes, trains, and evaluates a model or loads a pretrained model. (10 points)

The model demonstrates meaningful predictive power at least 75% classification accuracy or 0.80 R-squared. (5 points)

Data Model Optimization (25 points) The model optimization and evaluation process showing iterative changes made to the model and the resulting changes in model performance is documented in either a CSV/Excel table or in the Python script itself. (15 points)

Overall model performance is printed or displayed at the end of the script. (10 points)

GitHub Documentation (25 points) GitHub repository is free of unnecessary files and folders and has an appropriate .gitignore in use. (10 points)

The README is customized as a polished presentation of the content of the project. (15 points)

Presentation Requirements (25 points) Your presentation should cover the following:

An executive summary or overview of the project and project goals. (5 points)

An overview of the data collection, cleanup, and exploration processes. Include a description of how you evaluated the trained model(s) using testing data. (5 points)

The approach that your group took in achieving the project goals. (5 points)

Any additional questions that surfaced, what your group might research next if more time was available, or share a plan for future development. (3 points)

The results and conclusions of the application or analysis. (3 points)

Slides effectively demonstrate the project. (2 points)

Slides are visually clean and professional. (2 points)  


[image1]: <data:image/png;base64,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>
