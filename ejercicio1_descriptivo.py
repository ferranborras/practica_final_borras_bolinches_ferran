from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_diamond_dataset():
    """
    Devuelve un dataframe a partir del Diamond dataset
    """
    # Usamos la libreria pathlib para acceder a los datos mediante rutas relativas
    ROOT = Path(__file__).resolve().parent
    DATA_PATH = ROOT / "data" / "diamonds.csv"
    df = pd.read_csv(DATA_PATH) # Leemos el dataset
    return df

def resumen_estructural(df):
    """
    Imprime en pantalla información sobre el dataframe como su dimensión,
    tamaño, tipos de datos y cantidad de nulos por columna
    """
    #print("\n=== RESUMEN ESTRUCTURAL ===\n")
    #df.info()

def categorize_colums(df):
    """
    Convierte los datos de tipo string en categorias para reducir el tamaño del dataframe
    """
    cat_cols = ["cut", "color", "clarity"]
    for col in cat_cols:
        df[col] = df[col].astype("category")

def estadisticos_descriptivos(df, output):
    """
    Genera un csv con un análisis descriptivo de las variables numéricas
    y muestra el IQR del target (price)
    """
    #print("\n=== ESTADÍSTICOS DESCRIPTIVOS ===\n")
    num_cols = df.select_dtypes(include=np.number).columns # Seleccionar las columnas de tipo numérico

    # .describe proporciona algunos valores como count, mean, min, max y cuartiles
    desc = df[num_cols].describe().T # Usamos la traspuesta para que sea mas sencillo definir los campos faltantes
    desc["median"] = df[num_cols].median()
    desc["var"] = df[num_cols].var() # Varianza
    desc["mode"] = df[num_cols].mode().iloc[0]

    desc["skewness"] = df[num_cols].skew() # Asimetría (0 -> simetrica | > 0 -> sesgo derecha | < 0 -> sesgo izquierda)
    desc["kurtosis"] = df[num_cols].kurt() # Forma de la distribución (0 -> normal | > 0 -> pico alto | < 0 más plana)

    # guardar CSV
    desc.to_csv(output)

    # IQR de target (price)
    Q1 = df["price"].quantile(0.25)
    Q3 = df["price"].quantile(0.75)
    IQR = Q3 - Q1
    #print("IQR price:", IQR)

def histogramas(df, output):
    """
    Genera una imagen con los histogramas de las variables numéricas del dataframe.
    Representa la media y la mediana para una mejor comprensión de la simetría.
    """
    num_cols = df.select_dtypes(include=np.number).columns
    num_cols = num_cols[1:] # Omitimos la primera columna (id)

    # Configuración de la imagen
    n_cols = 3
    n_rows = int(np.ceil(len(num_cols) / n_cols))
    plt.figure(figsize=(15, 5 * n_rows))

    for i, col in enumerate(num_cols):
        plt.subplot(n_rows, n_cols, i + 1)

        # Histograma
        sns.histplot(df[col], bins=30)

        # Media y mediana
        mean = df[col].mean()
        median = df[col].median()

        # Visualizar la media y la mediana
        plt.axvline(mean, color='red', linestyle='--', label='Media')
        plt.axvline(median, color='green', linestyle='-', label='Mediana')

        plt.title(col)
        plt.legend()

    plt.tight_layout()
    plt.savefig(output)
    plt.close()

def boxplots_target(df, output):
    """
    Genera una imagen con boxplots del target segmentados por las variables categóricas
    """
    cat_cols = df.select_dtypes(include="category").columns

    fig, axes = plt.subplots(len(cat_cols), 1, figsize=(10, 5 * len(cat_cols)))

    if len(cat_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cat_cols):
        sns.boxplot(x=df[col], y=df["price"], ax=ax)
        ax.set_title(f"Price vs {col}")
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output)
    plt.close()

def get_column_outliers(df, col):
    """
    Devuelve una lista con los outliers del dataframe correspondientes
    a la columna especificada
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)] # Guarda los datos que no se encuentren entre lower y upper (outliers)

    return outliers

def detect_outliers(df):
    """
    Muestra el número de outliers que tiene cada campo del dataframe
    """
    num_cols = df.select_dtypes(include=np.number).columns
    outliers_count = {}
    for col in num_cols:
        outliers_count[col] = len(get_column_outliers(df, col))
    
    #print("\nOutliers count:\n", outliers_count)

def clean_invalid_values(df):
    """
    Devuelve el dataset eliminando las filas con dimensiones invalidas (x|y|z <= 0)
    """
    return df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]

def analizar_categoricas(df):
    """
    Devuelve un diccionario donde cada campo almacena un dataframe con
    la frecuencia absoluta y relativa de los registros de la columna
    {
        "cut": DataFrame,
        "color": DataFrame,
        "clarity": DataFrame
    }
    """
    cat_cols = df.select_dtypes(include="category").columns

    resultados = {}
    for col in cat_cols:
        abs_freq = df[col].value_counts()
        rel_freq = df[col].value_counts(normalize=True)

        # Organiza las frecuencias absolutas y relativas de los datos de cada columna
        # categórica en un dataframe (más fácil de visualizar)
        resultados[col] = pd.DataFrame({
            "frecuencia_absoluta": abs_freq,
            "frecuencia_relativa": rel_freq
        })
        #print(f"\n{resultados[col]}")

    return resultados

def graficos_categoricas(df, output):
    """
    Genera una imagen con los graficos de barras que muestran la frecuencia absoluta
    de los registros de cada variable categórica del dataframe
    """
    cat_cols = df.select_dtypes(include="category").columns # columnas categóricas

    plt.figure(figsize=(15, 5 * len(cat_cols)))

    for i, col in enumerate(cat_cols):
        plt.subplot(len(cat_cols), 1, i + 1)

        counts = df[col].value_counts() # Recuento de las veces que aparece cada registro
        sns.barplot(x=counts.index, y=counts.values)

        plt.title(f"Distribución de {col}")
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output)
    plt.close()

def heatmap_correlaciones(df, output):
    """
    Genera una imagen de un heatmap de correlación entre las variables númericas del dataframe
    """
    num_cols = df.select_dtypes(include=np.number).columns
    num_cols = num_cols[1:] # Omitimos la primera columna (id)

    corr = df[num_cols].corr(method="pearson")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")

    plt.title("Matriz de correlaciones (Pearson)")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

    return corr

def top_3_correlaciones(corr, target="price"):
    """
    Devuelve las 3 variables numéricas con mayor correlación absoluta sobre la variable indicada (por defecto "price")
    """
    corr_target = corr[target].drop(target) # Omitimos la correlacion del target consigo mismo (siempre será 1)

    top3 = corr_target.abs().sort_values(ascending=False).head(3) # Ordenamos de manera descendiente y seleccionamos los 3 primeros
    # Como la correlación es absoluta ignoramos el signo de R con .abs()

    # print("\nTop 3 variables más correlacionadas con price:")
    # print(top3)

    return top3

def detectar_multicolinealidad(corr, threshold=0.9):
    """
    Devuelve una lista de tuplas de formato (col1, col2, R) con los pares de
    variables cuya correlación supere el límite indicado (por defecto 0.9)
    """
    pares = []

    cols = corr.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr.iloc[i, j]) > threshold: # Si la coeficiente absoluto sobrepasa el límite se añade la pareja
                pares.append((cols[i], cols[j], corr.iloc[i, j]))

    return pares

def main():
    ROOT = Path(__file__).resolve().parent # Definimos el root para el enrutado

    # Leemos y creamos el dataframe a partir del csv
    df = get_diamond_dataset()

    # A) Resumen estructural
    #resumen_estructural(df)
    categorize_colums(df) # Reducir el tamaño del dataframe
    resumen_estructural(df) # Verificación de la ejecución anterior

    # B) Estadísticos descriptivos de variables numéricas
    OUTPUT_PATH = ROOT / "output"
    estadisticos_descriptivos(df, OUTPUT_PATH / "ej1_descriptivo.csv")

    # C) Distribuciones
    histogramas(df, OUTPUT_PATH / "ej1_histogramas.png")
    boxplots_target(df, OUTPUT_PATH / "ej1_boxplots.png")
    detect_outliers(df)
    # print(df[(df["x"] == 0) | (df["y"] == 0) | (df["z"] == 0)]) # Mostramos las filas con dimensiones anómalas
    df = clean_invalid_values(df) # Limpiamos el dataset eliminando los registros erróneos

    # D) Variables categóricas
    analizar_categoricas(df)
    graficos_categoricas(df, OUTPUT_PATH / "ej1_categoricas.png")

    # E) Correlaciones
    correlaciones = heatmap_correlaciones(df, OUTPUT_PATH / "ej1_heatmap_correlacion.png")
    top_3_correlaciones(correlaciones)
    detectar_multicolinealidad(correlaciones)


if __name__ == "__main__":
    main()