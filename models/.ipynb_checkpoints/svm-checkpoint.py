"""
svm.py - Entrenamiento y predicción con modelo Support Vector Machine (SVM) con kernel RBF.
"""
from sklearn.svm import SVC

def train_and_predict(X_train, y_train, X_test):
    """
    Entrena un modelo SVM (clasificación de margen máximo con kernel RBF) y predice sobre X_test.
    Retorna (modelo_entrenado, y_predicciones, y_probabilidades).
    """
    # probability=True permite obtener predict_proba (realiza internamente Platt scaling sobre el SVM):contentReference[oaicite:38]{index=38}
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return model, y_pred, y_prob