"""
calibrate.py - Funciones para calibrar probabilidades de clasificadores usando Platt scaling o isotonic regression.
"""
from sklearn.calibration import CalibratedClassifierCV

def calibrate_platt_cv(X_train, y_train, estimator):
    """
    Aplica calibración de Platt (sigmoid) a un estimador base usando validación cruzada en los datos de entrenamiento.
    - X_train, y_train: conjunto de entrenamiento.
    - base_estimator: modelo ya entrenado o instancia del modelo base (sklearn) a calibrar.
    Retorna un modelo calibrado (CalibratedClassifierCV) ajustado.
    """
    # Si el base_estimator ya está entrenado en X_train, se usará cv='prefit'.
    # En caso contrario, calibrator hará CV internamente. Aquí asumimos que ya está entrenado.
    calibrator = CalibratedClassifierCV(estimator=estimator, cv=5, method='sigmoid')
    calibrator.fit(X_train, y_train)
    return calibrator

def calibrate_isotonic_cv(X_train, y_train, base_estimator):
    """
    Aplica calibración isotónica a un estimador base usando validación cruzada en los datos de entrenamiento.
    Retorna el modelo calibrado (CalibratedClassifierCV) entrenado.
    """
    calibrator = CalibratedClassifierCV(estimator=estimator, cv=5, method='isotonic')
    calibrator.fit(X_train, y_train)
    return calibrator