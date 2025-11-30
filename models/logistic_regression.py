"""
logistic_regression.py - Entrenamiento y predicción con modelo de Regresión Logística.
"""
from sklearn.linear_model import LogisticRegression

def train_and_predict(X_train, y_train, X_test):
    """
    Entrena un modelo de Regresión Logística con los datos de entrenamiento y realiza predicciones sobre X_test.
    Retorna una tupla: (modelo_entrenado, y_predicciones, y_probabilidades)
    """
    # Configuramos el modelo logístico. Usamos solver LBFGS con un número mayor de iteraciones por si converge lento.
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    # Predicción de clases y probabilidades en el conjunto de prueba
    y_pred = model.predict(X_test)
    # predict_proba devuelve 2 columnas (probabilidad de clase 0 y 1); tomamos la columna de clase positiva (1)
    y_prob = model.predict_proba(X_test)[:, 1]
    return model, y_pred, y_prob