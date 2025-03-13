import pandas as pd

def load_csv(filepath):
    """
    Carga un archivo CSV y aplica limpieza de datos.
    
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
        df = clean_data(df)
        
        if df.empty:
            raise ValueError("El DataFrame está vacío después de la limpieza.")
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(f"El archivo '{filepath}' no existe.")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("El archivo CSV está vacío.")
    except Exception as ex:
        raise Exception(f"Error al cargar el archivo CSV: {str(ex)}")

def clean_data(df):
    # Elimina filas con valores nulos
    df = df.dropna()
    
    # Opcional: Rellenar valores nulos con 0 (si prefieres no eliminarlos)
    # df = df.fillna(0)
    
    # Convierte todas las columnas numéricas a tipo float
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Elimina filas con valores no numéricos en columnas numéricas
    df = df.dropna(subset=df.select_dtypes(include=["number"]).columns)
    
    return df