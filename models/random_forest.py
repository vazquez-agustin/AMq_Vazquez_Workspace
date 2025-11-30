"""
random_forest.py - Entrenamiento y predicción con modelo Random Forest.
"""
from sklearn.ensemble import RandomForestClassifier

def train_and_predict(X_train, y_train, X_test, n_estimators=100, max_depth=None, random_state=42):
    """
    Entrena un modelo Random Forest con los parámetros especificados (por defecto 100 árboles) y predice sobre X_test.
    Retorna (modelo_entrenado, y_predicciones, y_probabilidades).
    """
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return model, y_pred, y_prob