import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

x, y = make_classification(
    n_samples=5000, n_features=10, n_informative=5, n_redundant=2,
    n_classes=2, weights=[0.99, 0.01], flip_y=0, random_state=42
)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
