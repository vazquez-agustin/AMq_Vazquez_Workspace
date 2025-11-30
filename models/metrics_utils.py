"""
metrics_utils.py - Cálculo de métricas de evaluación y funciones de visualización (matriz de confusión, ROC, calibración).
"""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, brier_score_loss, roc_curve, auc
import matplotlib.pyplot as plt

def calcular_metricas_basicas(y_true, y_pred, y_prob=None):
    """
    Calcula métricas básicas de clasificación binaria dadas las etiquetas verdaderas y las predicciones:
      - Accuracy, Precision, Recall, F1-score.
    Si se proporcionan probabilidades (y_prob del positivo), también calcula Brier score.
    Retorna un diccionario con los valores de métricas.
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    if y_prob is not None:
        # Brier score: medida de calibración (menor es mejor)
        metrics['brier_score'] = brier_score_loss(y_true, y_prob, pos_label=1)
    return metrics

def brier_score(y_true, y_prob):
    """
    Calcula y devuelve el Brier score (pérdida cuadrática de probabilidad) dado el vector de etiquetas verdaderas y las probabilidades pronosticadas para la clase positiva.
    """
    return brier_score_loss(y_true, y_prob, pos_label=1)

def plot_confusion_matrix(y_true, y_pred, title="Matriz de Confusión"):
    """
    Dibuja la matriz de confusión para las etiquetas verdaderas vs. predichas.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(shrink=0.8)
    # Etiquetas en los ejes
    classes = ['No', 'Yes']  # asumiendo 0 = No, 1 = Yes
    tick_marks = [0, 1]
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicción')
    # Colocar los números en las celdas
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

def plot_roc_curve(y_true, y_prob, label=None):
    """
    Traza la curva ROC para valores verdaderos y probabilidades predichas.
    Si se proporciona un 'label', se usa como etiqueta en la leyenda.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})' if label else f'AUC = {roc_auc:.2f}')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")

def plot_calibration_curve(y_true, y_prob, n_bins=10, label=None):
    """
    Calcula y dibuja la curva de calibración (reliability diagram) para un modelo dado.
    Divide las predicciones en 'n_bins' buckets y compara la probabilidad promedio pronosticada vs. la fracción positiva real en cada bucket.
    """
    # Binning de probabilidades
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1  # id del bin al que pertenece cada prob
    bin_true_prob = []
    bin_pred_mean = []
    for b in range(n_bins):
        # seleccionar índices de predicciones que caen en el bin b
        idx = np.where(binids == b)[0]
        if len(idx) > 0:
            bin_true = np.mean(y_true[idx])  # fracción real de positivos en el bin
            bin_pred = np.mean(y_prob[idx])  # probabilidad promedio pronosticada en el bin
        else:
            bin_true = None
            bin_pred = None
        bin_true_prob.append(bin_true)
        bin_pred_mean.append(bin_pred)
    # Filtrar bins vacíos
    bin_true_prob = np.array([t for t in bin_true_prob if t is not None])
    bin_pred_mean = np.array([p for p in bin_pred_mean if p is not None])
    # Graficar
    plt.plot(bin_pred_mean, bin_true_prob, marker='o', label=label if label else 'Modelo')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # línea ideal
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel('Probabilidad pronosticada')
    plt.ylabel('Fracción de positivos observada')
    # Nota: No llamamos plt.show() aquí para permitir overlay de múltiples curvas en la misma figura.