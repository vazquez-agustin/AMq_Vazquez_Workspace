# Repositorio de Aprendizaje de Máquina I

- **Profesoras:** Facundo Adrián Lucianna & María Carina Roldán

- **Estudiantes:**  
	* Paola Andrea Blanco    (a2303)  
	* Facundo Manuel Quiroga (a2305)  
	* Juan Manuel Fajardo    (a2310)  
	* Victor Gabriel Peralta (a2322)  
	* Agustín Jesús Vazquez  (e2301)  

## Descripción del repositorio

Este repositorio contiene el **Trabajo Práctico Integrador** de la materia **Aprendizaje de Máquina I**, correspondiente a la Carrera de Especialización en Inteligencia Artificial (23Co2025).

El objetivo del trabajo es predecir la ocurrencia de lluvia al día siguiente en distintas localidades de Australia, utilizando técnicas de **aprendizaje automático supervisado** y datos meteorológicos históricos.

---

## 📌 Objetivo
Predecir la variable binaria `RainTomorrow` (Sí / No) a partir de variables meteorológicas del día actual, priorizando métricas robustas frente al desbalance de clases y probabilidades bien calibradas para toma de decisiones.

## 📂 Dataset
- **Fuente:** Rain in Australia (Kaggle)
- **Observaciones:** ~145.000
- **Variables:** 23 (originales)
- **Target:** `RainTomorrow`
- **Distribución de clases:** ~77% No / 23% Sí

## 🔎 Metodología
- Análisis exploratorio de datos (EDA)
- Limpieza de datos y tratamiento de valores faltantes
- Manejo de outliers (discretización y capping)
- One-Hot Encoding de variables categóricas
- Escalado de variables numéricas
- Split temporal de los datos (train / test)
- Entrenamiento y evaluación de múltiples modelos
- Optimización de hiperparámetros con Optuna
- Calibración de probabilidades (Isotonic Regression y Platt Scaling)

## 🤖 Modelos Evaluados
- Regresión Logística
- Naive Bayes Gaussiano
- K-Nearest Neighbors (KNN)
- Random Forest
- XGBoost

## 🏆 Modelo Final Seleccionado
**XGBoost optimizado con calibración isotónica**

**Justificación:**
- Mayor ROC-AUC
- Mejor calibración de probabilidades (Brier Score más bajo)
- Mejor balance entre Precision y Recall
- Umbral de decisión configurable según necesidades del negocio

## 📈 Métricas Utilizadas
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Brier Score
- Curvas ROC y Precision–Recall

## 📁 Estructura del Repositorio
├── dataset/
│ ├── weatherAUS.csv
│ └── weatherAUS_preprocessed.csv
├── models/
│ ├── preprocessing.py
│ ├── logistic_regression.py
│ ├── naive_bayes.py
│ ├── knn.py
│ ├── random_forest.py
│ ├── xgboost.py
│ ├── metrics_utils.py
│ └── calibratecopy1.py
├── notebooks/
│ └── TP_Integrador_AM1.ipynb
├── trained_models/ # opcional
└── README.md

## ▶️ Ejecución
1. Clonar el repositorio
2. Instalar las dependencias
3. Ejecutar el notebook principal: AMq_Trabajo_Final.ipynb

