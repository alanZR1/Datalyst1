import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def calculate_optimal_k(data, method="silhouette", max_k=8):

    if method == "silhouette":
        optimal_k = calculate_with_silhouette(data)
    else: 
        optimal_k = calculate_with_jambu(data)
        
    return optimal_k


def calculate_with_silhouette(data, max_k = 8):
    best_k = 2
    best_score = -1
    for k in range (2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=5)
        labels = kmeans.fit_predict(data)
        score = calculate_silhouette(data, labels)
        if score > best_score:
            best_score = score
            best_k = k
            
    return best_k


def calculate_with_jambu(data, max_k = 8):
    inertias = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=5)
        KMeans.fit(data)
        inertias.append(kmeans.inertia_)
        
    deltas = np.diff(inertias,2)
    return np.argmax(deltas) + 2  # +2 porque empezamos en k=2
    



def train_kmeans(df, k, n_init, x_col, y_col):
    """
    Entrena un modelo K-Means y genera un gráfico de los clusters.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        k (int): Número de clusters.
        n_init (int): Número de inicializaciones.
        x_col (str): Nombre de la columna para el eje X.
        y_col (str): Nombre de la columna para el eje Y.

    Retorna:
        img_base64 (str): Imagen del gráfico en base64.
        kmeans (KMeans): Modelo entrenado.
    """
    # Selecciona solo columnas numéricas
    df_numeric = df.select_dtypes(include=["number"])

    # Verifica que las columnas existan
    if x_col not in df_numeric.columns or y_col not in df_numeric.columns:
        raise ValueError(f"Las columnas {x_col} o {y_col} no existen en el DataFrame.")

    # Entrenamiento ....
    kmeans = KMeans(n_clusters=k, n_init=n_init)
    kmeans.fit(df_numeric)

    fig, ax = plt.subplots()
    ax.scatter(df_numeric[x_col], df_numeric[y_col], c=kmeans.labels_, cmap="viridis")
    ax.scatter(kmeans.cluster_centers_[:, df_numeric.columns.get_loc(x_col)], 
               kmeans.cluster_centers_[:, df_numeric.columns.get_loc(y_col)], 
               c="red", marker="X")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("K-Means Clustering")

    # Convierte el gráfico a base64
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return img_base64, kmeans

def calculate_silhouette(df, kmeans):
    """
    Calcula el índice de silueta para el modelo K-Means.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        kmeans (KMeans): Modelo K-Means entrenado.

    Retorna:
        float: Índice de silueta.
    """
    # Selecciona solo columnas numéricas
    df_numeric = df.select_dtypes(include=["number"])

    # Calcula el índice de silueta
    return silhouette_score(df_numeric, kmeans.labels_)

