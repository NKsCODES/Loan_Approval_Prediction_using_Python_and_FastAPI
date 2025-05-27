# loan_approval_prediction

# Project Overview
The objective of this project is to build a classification model that can predict whether a loan application will be approved ('Y') or not approved ('N'). This is a common problem in the financial industry, where accurate predictions can help lenders make informed decisions and mitigate risks.

# Dataset
The project utilizes a dataset named loan_data.csv. This dataset contains various features related to loan applicants and their loan details, including:
Loan_ID: Unique Loan ID
Gender: Male/Female
Married: Applicant married (Yes/No)
Dependents: Number of dependents
Education: Applicant Education (Graduate/Not Graduate)
Self_Employed: Self-employed (Yes/No)
ApplicantIncome: Applicant income
CoapplicantIncome: Co-applicant income
LoanAmount: Loan amount in thousands
Loan_Amount_Term: Term of loan in months
Credit_History: Credit history meets guidelines (1/0)
Property_Area: Urban/Semiurban/Rural
Loan_Status: Loan approved (Y/N) - Target Variable

# Methodology
The project follows a standard machine learning pipeline:
  
  1.Data Loading and Initial Inspection
  The dataset is loaded using pandas. Initial steps involve viewing the first few rows (df.head()), checking the dimensions (df.shape), and reviewing data types and non-null counts (df.info()). Missing values are also identified.
  
  2.Handling Missing Values
  Missing values are addressed in several columns:
  Rows with missing values in 'Gender', 'Dependents', and 'Loan_Amount_Term' are dropped.
  
  3.Missing values in 'Self_Employed' and 'Credit_History' are imputed using the mode of their respective columns.
  
  4.Feature Engineering and Encoding
  The Loan_ID column is dropped as it is a unique identifier and not useful for prediction.
  The 'Dependents' column's '3+' category is replaced with '4' to convert it to a numerical representation.
  Categorical features (Gender, Married, Dependents, Education, Self_Employed, Property_Area, Loan_Status) are converted into numerical representations using a custom encoding dictionary.
  
  5.Data Scaling
  Numerical features (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term) are scaled using StandardScaler to normalize their ranges, which helps in improving the performance of certain machine learning algorithms.

# Model Evaluation
  Several classification models are evaluated using accuracy and cross-validation scores:
  Logistic Regression
  Support Vector Classifier (SVC)
  Decision Tree Classifier
  Random Forest Classifier
  Gradient Boosting Classifier
  A function evaluate_model is used to train each model on a split dataset (80% train, 20% test) and report its accuracy and average cross-validation score.

# Hyperparameter Tuning
  RandomizedSearchCV is employed to find the best hyperparameters for selected models:
  Logistic Regression: Tunes C and solver.
  SVC: Tunes C and kernel.
  Random Forest Classifier: Tunes n_estimators, max_features, max_depth, min_samples_split, and min_samples_leaf.
  The best performing model from the tuning process (in this case, RandomForestClassifier) is selected as the final_model.

# Model Persistence
  The trained final_model and the StandardScaler object are saved using joblib for future use, preventing the need to retrain the model every time.
  loan_status_predictor.pkl: Stores the trained machine learning model.
  vector.pkl: Stores the fitted StandardScaler object for consistent data transformation.

# Prediction System
A simple prediction system is implemented to demonstrate how to use the saved model and scaler to predict the loan approval status for new, unseen data. The system takes a sample input, preprocesses it using the loaded scaler, and then uses the loaded model to make a prediction.

# Requirements
To run this project, you will need the following Python libraries:
  pandas
  numpy
  scikit-learn
  joblib  

# You can install them using pip:
  %pip install pandas numpy scikit-learn joblib

# Usage
1. Clone the repository (if applicable):

  % git clone <repository_url>
  % cd <repository_name>

2. Ensure you have the loan_data.csv file in the same directory as the notebook.
3. Run the Jupyter Notebook: Execute all cells in the Loan Approval Prediction.ipynb notebook to preprocess data, train models, tune hyperparameters, and save the final model and scaler.
4. Use the Prediction System: The notebook includes a section demonstrating how to load the saved model and make predictions on new data. You can adapt this code for your own inference needs.
