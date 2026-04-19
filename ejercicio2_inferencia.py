from pathlib import Path
import ejercicio1_descriptivo as ej1

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def preprocesamiento(df):
    """
    Devuelve los datos de las variables predictoras (X) y las variable objetivo (y) devididas
    en proproción 80% entrenamiento - 20% test.

    En esta función se elimina la columna Unnamed (id) puesto que no proporciona información
    """
    # Eliminamos columnas irrelevantes (id)
    #df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.drop(columns=["Unnamed: 0", "x", "y", "z"], errors="ignore")

    # One-Hot Encoding de variables categóricas
    df = pd.get_dummies(df, drop_first=True) # drop_first=true evita multicolinealidad dentro de cada variable categórica (redundancias en la regresión)

    # Separamos las variables
    X = df.drop(columns=["price"]) # campos predictorios
    df["log_price"] = np.log(df["price"])
    y = df["log_price"] # variable objetivo a predecir

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    ) # 80% para entrenamiento y 20% para test

    # Escalado de las variables (los rangos numéricos son muy distantes entre variables)
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train) # Ajustamos el scaler unicamente sobre los datos de entrenamiento
    X_test = scaler.transform(X_test) # Aplicamos la misma escala sobre los datos de test

    return X, y, X_train, X_test, y_train, y_test

def train_and_predict(X_train, X_test, y_train):
    """
    Entrena el modelo con los datos de entrenamiento y devuelve una predicción
    de la variable objetivo sobre las variables predictivas de test
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return model, y_pred

def calcuar_metricas(y_test, y_pred, output):
    """
    Calcula las métricas de evaluación del modelo (MAE, RMSE y R2).
    Genera un archivo de texto con los datos calculados.
    """
    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Guardamos las métricas
    with open(output, "w") as f:
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"R2: {r2:.4f}\n")

def grafico_residuos(y_test, y_pred, output):
    """
    Genera una grafico de residuos de las predicciones realizadas por el modelo
    de regresión lineal
    """
    # Residuos
    residuos = y_test - y_pred

    plt.figure(figsize=(8,6))
    plt.scatter(y_pred, residuos)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicciones")
    plt.ylabel("Residuos")
    plt.title("Gráfico de residuos")

    plt.savefig(output)
    plt.close()

    return residuos

def top_variables_influyentes(model, X):
    """
    Devuelve un dataframe con las variables predictoras y su coeficiente ordenados
    por el coeficiente absoluto de manera descendiente
    """
    coef_df = pd.DataFrame({
        "feature": X.columns,
        "coef": model.coef_
    }).sort_values(by="coef", key=abs, ascending=False)

    return coef_df

def main():
    ROOT = Path(__file__).resolve().parent # Definimos el root para el enrutado

    # Reciclando codigo leemos y creamos el dataframe a partir del csv
    df = ej1.get_diamond_dataset()

    # Categorizamos las variables
    ej1.categorize_colums(df)

    # Eliminamos los valores inválidos
    df = ej1.clean_invalid_values(df)

    # Prepocesamiento
    X, y, X_train, X_test, y_train, y_test = preprocesamiento(df)

    # Entrenamiento y predicción
    model, y_pred = train_and_predict(X_train, X_test, y_train)

    # Calcular métricas
    OUTPUT_PATH_METRICAS = ROOT / "output" / "ej2_metricas_regresion.txt"
    calcuar_metricas(y_test, y_pred, OUTPUT_PATH_METRICAS)

    # Gráfico de residuos
    OUTPUT_PATH_RESIDUOS = ROOT / "output" / "ej2_residuos.png"
    grafico_residuos(y_test, y_pred, OUTPUT_PATH_RESIDUOS)

    # Variables más influyentes
    #print(top_variables_influyentes(model, X))


if __name__ == "__main__":
    main()