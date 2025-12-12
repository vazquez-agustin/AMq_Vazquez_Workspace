"""
preprocessing.py - Módulo de preprocesamiento de datos para el proyecto Rain in Australia.
Incluye funciones para cargar datos, limpiar valores faltantes, codificar variables categóricas y escalar características.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DataFrame de entrada realizando:
      - Eliminación de columnas con muchos valores faltantes.
      - Eliminación de filas con valores faltantes en la variable objetivo o en campos críticos.
      - Imputación de valores faltantes en variables numéricas con la mediana.
      - Eliminación de filas con valores faltantes en variables categóricas (no imputamos categóricos).
      - Conversión de la columna RainTomorrow a numérica (0/1) y RainToday a 0/1.
      - Eliminación de columnas no utilizadas (por ejemplo, 'Date').
    Devuelve un DataFrame limpio, listo para transformación adicional.
    """
    df_clean = df.copy()
    
    # Eliminar columna de fecha (no la usaremos como predictor)
    if 'Date' in df_clean.columns:
        df_clean = df_clean.drop('Date', axis=1)
    
    # Mapear la variable objetivo a 0/1 (para facilitar cálculos)
    if 'RainTomorrow' in df_clean.columns:
        df_clean['RainTomorrow'] = df_clean['RainTomorrow'].map({'No': 0, 'Yes': 1})
    # También mapear RainToday a 0/1
    if 'RainToday' in df_clean.columns:
        df_clean['RainToday'] = df_clean['RainToday'].map({'No': 0, 'Yes': 1})
    
    # 1. Eliminar columnas con más del 10% de datos faltantes
    # (Por análisis previo, sabemos que Sunshine, Evaporation, Cloud9am, Cloud3pm caen en este caso):contentReference[oaicite:24]{index=24}
    # Calculamos porcentaje de nulos por columna:
    nulos_porcentaje = df_clean.isnull().mean()
    cols_to_drop = nulos_porcentaje[nulos_porcentaje > 0.10].index
    df_clean = df_clean.drop(columns=cols_to_drop)
    
    # 2. Eliminar filas donde RainTomorrow es nulo (no se puede entrenar sin objetivo):contentReference[oaicite:25]{index=25}
    if 'RainTomorrow' in df_clean.columns:
        df_clean = df_clean[df_clean['RainTomorrow'].notna()]
    
    # 3. Eliminar filas con nulos en columnas críticas (>1% nulos):contentReference[oaicite:26]{index=26}.
    # Recalcular porcentaje de nulos tras haber eliminado columnas en el paso 1
    nulos_porcentaje = df_clean.isnull().mean()
    cols_criticas = nulos_porcentaje[(nulos_porcentaje > 0.01) & (nulos_porcentaje <= 0.10)].index
    # Eliminar filas que tengan nulos en esas columnas críticas
    if len(cols_criticas) > 0:
        df_clean = df_clean.dropna(subset=cols_criticas)
    
    # 4. Imputar valores faltantes en variables numéricas (columnas restantes con muy pocos nulos, <=1%)
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
    
    # 5. Eliminar filas con valores faltantes en variables categóricas restantes (si quedara alguna)
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        df_clean = df_clean.dropna(subset=cat_cols)
    
    # A estas alturas, df_clean no debería tener valores faltantes
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean

def transformar_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Aplica transformaciones a las características:
      - One-hot encoding a variables categóricas (dummies).
      - Escalado Min-Max a características numéricas.
    Recibe X_train y X_test originales (tras limpieza) y devuelve nuevas versiones transformadas (pandas.DataFrame).
    """

    # 1) One-hot en TRAIN
    cat_cols_train = X_train.select_dtypes(include=["object"]).columns.tolist()
    X_train_enc = pd.get_dummies(X_train, columns=cat_cols_train, drop_first=True)

    # 2) One-hot en TEST (puede generar columnas distintas)
    cat_cols_test = X_test.select_dtypes(include=["object"]).columns.tolist()
    X_test_enc = pd.get_dummies(X_test, columns=cat_cols_test, drop_first=True)

    # 3) Alinear columnas: el TEST queda con las columnas del TRAIN
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    # 4) Escalado MinMax (fit en TRAIN, transform en TEST)
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_enc),
        columns=X_train_enc.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_enc),
        columns=X_train_enc.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled

