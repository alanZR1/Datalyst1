import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_csv(filepath):
    """
    Carga un archivo CSV solo lectura
    
    Parámetros:
        filepath (str): Ruta del archivo CSV.
    
    Retorna:
        pd.DataFrame: DataFrame con los datos limpios.
    Lanza:
        FileNotFoundError: Si el archivo no existe.
        pd.errors.EmptyDataError: Si el archivo está vacío.
        Exception: Si ocurre un error inesperado.
    """
    try:
        df = pd.read_csv(filepath)
        
        if df.empty:
            raise ValueError("El DataFrame está vacío después de la limpieza.")
        return df
    
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {str(e)}")
        raise

def clean_data(df, remove_outliers = True, fill_na = "none",
               remove_duplicates = False,
               normalize = False):
    """
    Aplica limpieza de datos según los parámetros especificados.
    
    Parámetros:
        df (pd.DataFrame): DataFrame a limpiar.
        remove_outliers (bool): Si True, elimina outliers usando el rango intercuartílico.
        fill_na (str): Método para rellenar nulos ("none", "mean", "median", "mode").
        remove_duplicates (bool): Si True, elimina filas duplicadas.
        normalize (bool): Si True, normaliza las columnas numéricas.
    
    Retorna:
        pd.DataFrame: DataFrame limpio.
    """
    #crea una copia del dataframe original
    df_clean = df.copy()
    # 1.-  manejo de valores nulos
    if fill_na != "none":
        numeric_cols = df_clean.select_dtypes(include = [np.number]).columns
        df_clean.fillna(df_clean.mean(), inplace = True)
        if fill_na == "mean":
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif fill_na == "median":
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif fill_na == "mode":
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mode().iloc[0])
        else:
            #eliminar filas con nulos sino se especifica el metodo
            df_clean.dropna(inplace = True)
            
    # 2.- eliminar duplicados
    if remove_duplicates:
        df_clean.drop_duplicates(inplace = True)
    
    # 3.- eliminar outliers
    if remove_outliers:
        numeric_cols = df_clean.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    # 4.- normalizar datos
    if normalize:
        numeric_cols = df_clean.select_dtypes(include=["number"]).columns
        scaler = MinMaxScaler()
        df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

    # Conversión final de tipos numéricos
    for col in df_clean.select_dtypes(include=["number"]).columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    
    # Eliminar filas con valores no numéricos en columnas numéricas
    df_clean = df_clean.dropna(subset=df_clean.select_dtypes(include=["number"]).columns)
    
    return df_clean