"""
preprocessing.py

Preprocesamiento estructural consistente para el proyecto Rain in Australia:

* Convierte 'Date' a datetime.
* Extrae Month y Year.
* Aplica codificación cíclica del mes (Month_sin / Month_cos).
* Elimina filas sin valor en la variable objetivo (RainTomorrow).
* Elimina columnas 'Date', 'Month' y 'Year' (ya reemplazadas).

El resto del preprocesamiento (imputación, OneHotEncoder, escalado selectivo)
se hace dentro de los Pipelines de scikit-learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def preparar_dataframe(
    df: pd.DataFrame,
    target_col: str = "RainTomorrow",
) -> pd.DataFrame:
    """
    Aplica el preprocesamiento estructural sobre el DataFrame original.

    Pasos:
    - Convierte la columna 'Date' a datetime (si existe).
    - Extrae Month desde 'Date'.
    - Crea Month_sin y Month_cos como codificación cíclica del mes.
    - Elimina columnas 'Date' y 'Month' (si existen).
    - Elimina filas donde la variable objetivo (target_col) es NaN.
    - NO elimina otras filas por nulos: la imputación se hace luego en el Pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original (por ejemplo, weatherAUS).
    target_col : str, opcional
        Nombre de la variable objetivo. Por defecto "RainTomorrow".

    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame limpio y con columnas de fecha transformadas,
        listo para hacer train_test_split y armar Pipelines.
    """
    df_clean = df.copy()

    # 1) Manejo de la fecha
    if "Date" in df_clean.columns:
        # Convertir a datetime
        df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")
        df_clean["Month"] = df_clean["Date"].dt.month

        # Codificación cíclica del mes
        df_clean["Month_sin"] = np.sin(2 * np.pi * (df_clean["Month"] - 1) / 12)
        df_clean["Month_cos"] = np.cos(2 * np.pi * (df_clean["Month"] - 1) / 12)

        df_clean = df_clean.drop(columns=["Date", "Month"], errors="ignore")


    # 2) Eliminar filas sin target
    if target_col in df_clean.columns:
        df_clean = df_clean.dropna(subset=[target_col])

    # No hacemos más drops ni imputaciones acá
    df_clean = df_clean.reset_index(drop=True)
    return df_clean
