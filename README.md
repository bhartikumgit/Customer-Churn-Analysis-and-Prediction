# Customer-Churn-Analysis-and-Prediction

Overview

This project focuses on analyzing customer churn and retention patterns using historical customer data and deploying a machine learning–based churn prediction system. The goal is to identify high-risk customers, understand the drivers of churn, and enable data-driven retention strategies.
The project covers the full pipeline: data analysis → model training → evaluation → deployment via a Flask web application.

#Problem Statement-
##Customer churn directly impacts revenue and growth. The objective of this project is to:

Analyze customer behavior and engagement data

Identify key factors contributing to churn

Predict whether a customer is likely to churn

Provide both individual and batch-level churn analysis through a web interface

##Key Features

Exploratory data analysis on customer and churn-related variables

Feature engineering and preprocessing

Random Forest–based churn prediction model

Model evaluation using ROC curve, confusion matrix, and feature importance

Flask web app for:

Individual customer churn prediction

Batch churn analysis via file upload

Modular and reproducible project structure

##Tech Stack

Programming & Analysis

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Modeling

Random Forest Classifier

Label encoding and feature validation

Deployment

Flask

HTML (Jinja templates)

##Project Structure
churn-notebook/
│
├── app.py                  # Main Flask application
├── run_app.py              # App runner
├── start_app.bat           # Windows startup script
│
├── train_model.py          # Model training pipeline
├── check_features.py       # Feature consistency checker
│
├── rf_model.pkl            # Trained Random Forest model
├── label_encoders.pkl      # Saved encoders
│
├── templates/              # HTML templates
│   ├── home.html
│   ├── upload.html
│   ├── analysis_choice.html
│   ├── batch_results.html
│   ├── individual_select.html
│   ├── result.html
│   ├── index.html
│   └── analytics.html
│
├── churn analysis.xlsx     # Dataset
├── requirements.txt
├── README.md


