import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler



def load_csv(filepath):
    
    try:
        df = pd.read_csv(filepath)
        
        if df.empty:
            raise ValueError("El DataFrame está vacío después de la limpieza.")
        return df
    
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {str(e)}")
        raise


def clean_data(df, 
               remove_outliers = True,
               fill_na = "none",
               remove_duplicates = False,
               normalize = False):

    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include = ['int64', 'float64']).columns
    
    if fill_na != "none":
        if fill_na == "mean":
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif fill_na == "median":
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif fill_na == "mode":
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mode().iloc[0])
        else:
            df_clean.dropna(subset = numeric_cols, inplace = True)
            
    
    if remove_duplicates:
        df_clean.drop_duplicates(inplace = True)
    
    
    if remove_outliers:
        for col in numeric_cols:
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    
    if normalize:
        scaler = MinMaxScaler()
        df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])


    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors = "coerce")
    
    df_clean = df_clean.dropna(subset = numeric_cols)
    
    return df_clean