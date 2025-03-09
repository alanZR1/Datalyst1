import pandas as pd

def load_csv(filepath):
    """
    Carga un archivo CSV y aplica limpieza de datos.

    Parámetros:
    filepath (str): Ruta del archivo CSV.

    Retorna:
    pd.DataFrame: DataFrame limpio.
    """
    df = pd.read_csv(filepath)
    df = clean_data(df)  # Aplica la limpieza antes de retornarlo
    return df


def clean_data(df):
    """
    Realiza limpieza de datos en un DataFrame.

    Parámetros:
    df (pd.DataFrame): DataFrame a limpiar.

    Retorna:
    pd.DataFrame: DataFrame limpio.
    """
    # Elimina filas con valores nulos
    df = df.dropna()

    # Elimina duplicados
    df = df.drop_duplicates()

    # Convierte todas las columnas numéricas a tipo float
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df