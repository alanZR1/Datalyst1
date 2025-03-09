import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def train_kmeans(filepath, k, n_init, x_col, y_col):
    df = pd.read_csv(filepath)
    df_numeric = df.select_dtypes(include=["number"])

    kmeans = KMeans(n_clusters=k, n_init=n_init)
    kmeans.fit(df_numeric)

    fig, ax = plt.subplots()
    ax.scatter(df_numeric.iloc[:, int(x_col)], df_numeric.iloc[:, int(y_col)], c=kmeans.labels_, cmap='viridis')
    ax.scatter(kmeans.cluster_centers_[:, int(x_col)], kmeans.cluster_centers_[:, int(y_col)], c='red', marker='X')
    ax.set_xlabel(f"Característica {x_col}")
    ax.set_ylabel(f"Característica {y_col}")
    ax.set_title("K-Means Clustering")

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    img_base64 = base64.b64encode(buf.getvalue()).decode()
    return img_base64, kmeans

def calculate_silhouette(filepath, kmeans):
    df = pd.read_csv(filepath).select_dtypes(include=["number"])
    return silhouette_score(df, kmeans.labels_)
