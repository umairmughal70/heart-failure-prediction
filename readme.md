Heart Disease Prediction
This repository contains a machine learning project that aims to predict heart disease using a dataset with eleven different features, including conventional clinical signs such as high blood pressure, high cholesterol, chest discomfort, and more. We have employed supervised machine learning classification methods to build a reliable prediction model. The following classifiers have been used:

Random Forest
Logistic Regression
XGBoost
K-Nearest Neighbors
Support Vector Machine (SVM)
Bernoulli Naive Bayes
Dataset
The dataset used in this project consists of the following columns:

Age
Sex
ChestPainType
RestingBP (Resting Blood Pressure)
Cholesterol
FastingBS (Fasting Blood Sugar)
RestingECG (Resting Electrocardiographic Results)
MaxHR (Maximum Heart Rate Achieved)
ExerciseAngina
Oldpeak
ST_Slope
HeartDisease (Target Variable)
Data Preprocessing
Before training the machine learning models, we performed some data preprocessing steps:

We loaded the dataset from a CSV file.
Removed any duplicate rows from the dataset.
Standardized numerical columns (RestingBP, Cholesterol, MaxHR) to have zero mean and unit variance using StandardScaler.
Encoded categorical features (Sex, ChestPainType, RestingECG, ST_Slope, ExerciseAngina) using LabelEncoder.
Model Building
We split the dataset into training and testing sets using train_test_split and used the following classifiers to build the models:

Random Forest
Logistic Regression
XGBoost
K-Nearest Neighbors
Support Vector Machine (SVM)
Bernoulli Naive Bayes
Evaluation
We evaluated the models using the following metrics:

Accuracy
Precision
Recall
F1-score
Matthews Correlation Coefficient (MCC)
Confusion Matrix
Results
After evaluating the models, we found that the Random Forest classifier achieved the highest accuracy and is the most effective model for predicting heart disease.

Receiver Operating Characteristic (ROC) Curve
We also plotted the ROC curve to visualize the model's performance in terms of True Positive Rate (TPR) and False Positive Rate (FPR) and calculated the Area Under the Curve (AUC).

The results and code provided in this repository offer insights into predicting heart disease using machine learning. Feel free to explore the code and the Jupyter Notebook provided to gain a deeper understanding of the project.