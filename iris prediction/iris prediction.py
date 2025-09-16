from sklearn.model_selection import train_test_split
fro sklearn.preprocessing import StandardScaler
from sklearn.neighbours import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def flower_pred():
    print("Predicting Iris Flower Species using K-Nearest Neighbors (KNN)")
    iris = load_iris()
    x = iris.data
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    
    print(f"Accuracy score: \n{accuracy_score(y_test, y_pred)}")
    print(f"Classification report: \n{classification_report(y_test, y_pred)}\n")

    sample = [[5.1, 3.5, 1.4, 0.2]]
    scaled = scaler.transform(sample)
    pred = model.predict(scaled)
    print("\nPrediction for sample {}: {}".format(sample, iris.target_names[pred[0]]))


flower_pred()
