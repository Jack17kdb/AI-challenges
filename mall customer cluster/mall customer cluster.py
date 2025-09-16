import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

def mall_customer_cluster():
    df = pd.read_csv('mall_customers.csv')
    print(df.head(), "\n")
    print(df.info(), "\n")

    x = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(x)

    center = kmeans.cluster_centers_
    print(center)

    cluster_profiles = pd.DataFrame(center, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
    print(cluster_profiles)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x['Age'], x['Annual Income (k$)'], x['Spending Score (1-100)'],
           c=df['Cluster'], cmap='rainbow', s=50)

    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income (k$)")
    ax.set_zlabel("Spending Score (1-100)")
    plt.title("Customer Segments in 3D")
    plt.show()

mall_customer_cluster()
