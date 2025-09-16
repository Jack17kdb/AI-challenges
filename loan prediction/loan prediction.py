import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def loan_pred():
    print("Loan Prediction using Random Forest\n")
    df = pd.read_csv('loan_data.csv')
    print(f"{df.info()}\n")
    print(f"{df.head()}\n")
    df.drop('customer_id', axis=1, inplace=True)
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols.remove('defaulted')
    df['defaulted'] = df['defaulted'].map({"No": 0, "Yes": 1})
    le = LabelEncoder()
    for cols in cat_cols:
        df[cols] = le.fit_transform(df[cols])
    print(f"Processes data:\n{df.head()}")
    
    x = df.drop('defaulted', axis=1)
    y = df['defaulted']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred)}\n")
    print(f"Classification Report: \n{classification_report(y_test, y_pred)}\n")


loan_pred()
