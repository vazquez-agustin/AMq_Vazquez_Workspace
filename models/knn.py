"""
knn.py - Entrenamiento y predicción con modelo K-Nearest Neighbors.
"""
from sklearn.neighbors import KNeighborsClassifier

def train_and_predict(X_train, y_train, X_test, n_neighbors=5):
    """
    Entrena un modelo KNN (K-Nearest Neighbors) con el número de vecinos especificado y predice sobre X_test.
    Retorna (modelo_entrenado, y_predicciones, y_probabilidades).
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # En KNN, predict_proba está disponible si n_neighbors > 1
    y_prob = model.predict_proba(X_test)[:, 1]
    return model, y_pred, y_prob