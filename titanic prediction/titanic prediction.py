import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def titanic_pred():
    print("Titanic Survival Prediction Using Logistic Regression\n")
    df = sns.load_dataset("titanic")
    print(df.info())
    print("\n")
    print(df.head())
    print("\n")
    print(df.isnull().sum())
    print("\n")
    df = df.drop(["alone", "alive", "embark_town", "deck", "who", "class", "adult_male"], axis=1)
    df = df.dropna(subset=["age", "embarked"])
    df['sex'] = df['sex'].map({"male": 0, "female": 1})
    df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    print(f"Formatted infomation: \n\n{df.head()}\n")

    x = df.drop("survived", axis=1)
    y = df['survived']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"Accuracy score: \n{accuracy_score(y_test, y_pred)}")
    print(f"Confusion matrix: \n{confusion_matrix(y_test, y_pred)}\n")
    print(f"Classification report: \n{classification_report(y_test, y_pred)}\n")


titanic_pred()
