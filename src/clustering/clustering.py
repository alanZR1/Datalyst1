import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def calculate_optimal_k(df, method = "silhouette", max_k = 10):
    
    if method == "silhouette":
        return silhouette_k(df, max_k)    
    else:
        return jambu_k(df, max_k) 
    
    
    
def jambu_k(df, max_k):
    inertias = []
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters = k, n_init = 5).fit(df)
        inertias.append(kmeans.inertia_)
    
    deltas = np.diff(inertias, 2)
    
    return np.argmax(deltas) + 2  



def silhouette_k(df, max_k):
    best_k, best_score = 2, -1
    
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters = k, n_init = 5).fit_predict(df)
        score = silhouette_score(df, labels)
    
        if score > best_score:
            best_k, best_score = k, score
    
    return best_k
    

def train_kmeans(df, k, n_init, x_col, y_col):
 
    df_numeric = df.select_dtypes(include = ["number"])

    if x_col not in df_numeric.columns or y_col not in df_numeric.columns:
        raise ValueError(f"Las columnas {x_col} o {y_col} no existen en el DataFrame.")


    kmeans = KMeans(n_clusters = k, n_init = n_init)
    kmeans.fit(df_numeric)

    fig, ax = plt.subplots()
    
    ax.scatter(df_numeric[x_col], df_numeric[y_col], c = kmeans.labels_, cmap = "viridis")
    ax.scatter(kmeans.cluster_centers_[:, df_numeric.columns.get_loc(x_col)], 
               kmeans.cluster_centers_[:, df_numeric.columns.get_loc(y_col)], 
               c = "red", marker = "X")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("K-Means Clustering")

    buf = BytesIO()
    
    plt.savefig(buf, format = "png")
    plt.close()
    
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return img_base64, kmeans



def calculate_silhouette(df, kmeans):
    df_numeric = df.select_dtypes(include = ["number"])

    return silhouette_score(df_numeric, kmeans.labels_)

