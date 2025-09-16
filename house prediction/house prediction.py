import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

def house_pred():
    print("House Price Prediction using Linear Regression\n")
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['MedHouseVal'] = data.target
    print(df.head())
    print("\n")
    print(df.info())
    print("\n")

    x = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Root Mean Squared Error: {rmse}\n")
    print(f"R² Score: {r2}")
    coeff_df = pd.DataFrame(model.coef_, x.columns, columns=["Coefficient"])
    print(coeff_df)
    
house_pred()
