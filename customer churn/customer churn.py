import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def customer_churn():
    print("Customer Churn Prediction using Logistic Regression\n")
    df = pd.read_csv('churn_data.csv')
    print(f"{df.info()}\n")
    df.drop("customerID", axis=1, inplace=True)
    print(df.isnull().sum())

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols.remove('Churn')
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    df['Churn'] = df['Churn'].map({"No": 0, "Yes": 1})
    print(df.dtypes)

    x = df.drop('Churn', axis=1)
    y = df['Churn']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    print(f"Accuracy: {accuracy_score(y_test, pred)}\n\n")
    print(f"Confusion matrix: {confusion_matrix(y_test, pred)}\n\n")
    print(f"Classification report: \n{classification_report(y_test, pred)}\n\n")
    
customer_churn()
