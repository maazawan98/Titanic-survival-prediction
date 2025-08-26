# titanic-survival-prediction
A machine learning model to predict passenger survival on the Titanic.

Project Overview 

Objective: 

The objective of this project is to develop a machine learning model that can accurately predict whether a passenger survived the Titanic disaster, based on features such as Age, Sex, Pclass, and Fare. 

Key Goals: 
 
Learn the end-to-end machine learning pipeline using a real dataset. - Clean and preprocess data to handle missing values and encode categorical features. 
Apply classification algorithms such as Random Forest to train the model. - Evaluate the model’s performance using accuracy, precision, recall, F1-score, and confusion matrix. 
Visualize the insights from the dataset using Seaborn and Matplotlib. 
 
Dataset Description: 
 
The Titanic dataset consists of information about the passengers aboard the 
Titanic ship. Each row represents a passenger, with features like PassengerId, 
Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, and Embarked. 
For this task, we focused on: Age, Sex, Pclass, and Fare to predict the target variable 'Survived'. 
 
Step 1: Import Required Libraries 

We imported libraries for data handling, visualization, and modeling. 
import pandas as pd import numpy as np import seaborn as sns import matplotlib.pyplot as plt from sklearn.model_selection import train_test_split from sklearn.preprocessing import LabelEncoder from sklearn.impute import SimpleImputer from sklearn.ensemble import RandomForestClassifier from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
 
Step 2: Load the Dataset and Perform Initial Exploration 

We loaded the dataset and performed basic data exploration to understand structure and detect missing values. 
df = pd.read_csv('Titanic-Dataset.csv') print(df.info()) print(df.describe()) print(df.isnull().sum()) 
 
Step 3: Data Preprocessing 
 
We cleaned the dataset by handling missing values in the Age column using median imputation. The Sex column was label-encoded to convert it into numeric format. 
The final features selected were: Age, Sex, Pclass, and Fare. 
df_model = df[['Age', 'Sex', 'Pclass', 'Fare', 'Survived']].copy() imputer = SimpleImputer(strategy='median') df_model['Age'] = imputer.fit_transform(df_model[['Age']]) df_model.dropna(subset=['Fare'], inplace=True) df_model['Sex'] = LabelEncoder().fit_transform(df_model['Sex']) 
 
Step 4: Data Visualization and Insights 
 
We used Seaborn and Matplotlib to analyze patterns and relationships between Age, Sex, Fare, and survival. 
Example plots: 
Histogram of Age distribution. 
Survival rate by gender. 
Fare distribution across survival classes. 
sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', stat='percent', bins=30) plt.title('Age vs Survived') plt.show() 
 
Step 5: Splitting the Data 

We split the data into training and test sets (80/20) to train and evaluate the model. 
X = df_model[['Age', 'Sex', 'Pclass', 'Fare']] y = df_model['Survived'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
Step 6: Training the Model 

We trained a Random Forest Classifier with 100 trees. This algorithm is robust, handles both categorical and numerical data, and avoids overfitting. 
model = RandomForestClassifier(n_estimators=100, random_state=42) model.fit(X_train, y_train) 
 
Step 7: Making Predictions and Evaluating the Model 
 
The model was evaluated using accuracy, precision, recall, F1-score, and confusion matrix. 
These metrics give insight into model performance on unseen data. 
y_pred = model.predict(X_test) accuracy = accuracy_score(y_test, y_pred) report = classification_report(y_test, y_pred) conf_matrix = confusion_matrix(y_test, y_pred) print("Accuracy:", accuracy) print(report) print(conf_matrix) 
 
Step 8: Feature Importance 
 
We visualized feature importance using the built-in attribute of Random 
Forest. This helps understand which features contributed most to the survival prediction. 
importances = model.feature_importances_ features = X.columns sns.barplot(x=importances, y=features) plt.title('Feature Importance') plt.show() 
 
Step 9: Conclusion 
 
The Random Forest classifier achieved nearly 80% accuracy. The most important features were Sex, Pclass, and Age.  
This project demonstrates the full ML pipeline from data preprocessing to model evaluation. 
Future improvements may include hyperparameter tuning, feature engineering (like FamilySize), and trying other models like XGBoost. 
 
THEORY CONCEPTS USED IN TASK 1 
1. Machine Learning Workflow 
 
Problem Definition — Predict survival of passengers. 
Data Collection — Titanic dataset (CSV file). 
Data Exploration — Analyzing structure, summary statistics. 
Data Preprocessing — Handling missing values, encoding, and selecting features. 
Model Training — Random Forest. 
Evaluation — Accuracy, confusion matrix. 
Improvement — Feature importance analysis and visualization. 
 
2. Classification vs Regression 

This task is classification (binary): predicting survival (0 or 1), not regression (continuous prediction). 

3. Model Evaluation Metrics 
 
Accuracy: Correct predictions / total predictions. 
Precision: True positives / predicted positives. 
Recall: True positives / actual positives. 
F1-score: Harmonic mean of precision and recall. 
Confusion Matrix: Visual layout of predicted vs actual values. 
 
4. Label Encoding 

Categorical data like 'Sex' must be converted to numbers using Label Encoding before training the model.

5. Random Forest Classifier 

An ensemble model that combines multiple decision trees for robust, accurate predictions and reduced overfitting.
