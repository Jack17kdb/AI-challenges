import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import random

def main():
    random.seed(42)
    np.random.seed(42)
    
    print("Sentiment Analysis with Scikit-Learn")
    print("=" * 40)
    
    positive_reviews = [
        "This movie was absolutely fantastic! Great acting and storyline.",
        "I loved every minute of it. Highly recommend to everyone.",
        "Outstanding performance by all actors. A masterpiece!",
        "Brilliant cinematography and excellent direction. Must watch!",
        "One of the best movies I've ever seen. Perfect in every way."
    ]
    
    negative_reviews = [
        "Terrible movie with poor acting. Complete waste of time.",
        "Boring storyline and bad direction. Very disappointing.",
        "Awful performances and confusing plot. Not recommended.",
        "Poor quality film with terrible acting. Avoid at all costs.",
        "Disappointing movie with weak characters. Very bad."
    ]
    
    texts = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    test_examples = [
        "This movie was amazing and I loved it!",
        "Terrible film, complete waste of time."
    ]
    
    print("\nTesting on new examples:")
    for example in test_examples:
        example_tfidf = vectorizer.transform([example])
        prediction = model.predict(example_tfidf)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        print(f"'{example}' -> {sentiment}")

if __name__ == "__main__":
    main()
