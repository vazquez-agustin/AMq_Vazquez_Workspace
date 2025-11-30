"""
naive_bayes.py - Entrenamiento y predicción con modelo Naive Bayes Gaussiano.
"""
from sklearn.naive_bayes import GaussianNB

def train_and_predict(X_train, y_train, X_test):
    """
    Entrena un modelo Gaussian Naive Bayes con los datos de entrenamiento y realiza predicciones sobre X_test.
    Retorna (modelo_entrenado, y_predicciones, y_probabilidades).
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return model, y_pred, y_prob