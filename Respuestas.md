# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo

---

> ### A) Resumen estructural
>
> Mediante `.info()` obtenemos las dimensiones del dataset con 53940 filas y 11 columnas (3.4 MB).
> Inicialmente el dataset es más pesado pero se ha reducido su tamaño categorizando los campos `string`.
> | carat | cut | color | clarity | depth | table | price | x | y | z |
> | :-----: | :------: | :------: | :------: | :-----: | :-----: | :---: | :-----: | :-----: | :-----: |
> | float64 | category | category | category | float64 | float64 | int64 | float64 | float64 | float64 |
>
> Este dataset no contiene valores nulos por lo que el porcentaje de nulos en cada uno de los campos es 0%.
>
> ### B) Estadísticos descriptivos de variables numéricas
>
> Mediante `.descrbe()` y otras funciones adicionales (`.median()`, `.var()`, `.mode()`, `.skew()` y `.kurt()`) obtenemos la información correspondfiente a la media, mediana, moda, desviación típica, varianza, mínimo, máximo y cuartiles de las variables numéricas.
>
> Además calculamos el rango intercuartílico (IQR) de la variable objetivo calculando **Q1** y **Q3** mediante `.quantile(0.25)` y `.quantile(0.75)`.
> $$IQR = Q3 - Q1$$
> **IQR de _price_ (variable objetivo):** 4374.25
>
> ### C) Distribuciones
>
> Se han creado las imágenes de los histogramas mostrando también la media y la mediana para representar la asimetría de las vaiables numéricas. Se observa una fuerte asimetría en la distribución del campo `price`, mientras que la dimensión `x` muestra una distribución practicamente normal un poco aplanada (kurtosis < -0.5) y ligeramente sesgada a la derecha (skewness entre 0 y 0.5).
>
> ![ej1_histogramas.png](/output/ej1_histogramas.png)
>
> También se han generado los gráficos boxplot de la variable objetivo, `target`, segmentados por cada variable categórica.
> Se puede observar una gran cantidad de outliers por encima del precio máximo dentro del IQR. Además, la mediana en las diferentes gráficas nos muestra una tendencia en el precio rondando los 2500 $.
>
> ![ej1_boxplots.png](/output/ej1_boxplots.png)
>
> Por último, como muchas de las variables muestran distribuciones asimétricas se ha optado por utilizar el método de rango intercuartílico (IQR) para la detección de outliers, por su robustez frente a distribuciones no normales y valores extremos.
>
> Durante el análisis se ha detectado registros erróneos en los campos correspondientes a las dimensiones del diamante (`x`, `y`, `z`) por lo que se ha decidido eliminar estos valores para limpiar el dataset.
>
> ### D) Variables categóricas
>
> Se ha analizado la distribución de las variables categóricas >mediante sus frecuencias relativas.
>
> La variable _`cut`_ presenta una ligera dominancia de la categoría >_`Ideal`_ (~40%), aunque el resto de categorías mantienen una >representación significativa, excepto por _`Good`_ y _`Fair`_ que no >alcanzan el 0.1%.
>
> La variable _`color`_ muestra una distribución bastante uniforme >entre sus categorías, sin que ninguna destaque de forma dominante..
>
> Por último, la variable _`clarity`_ presenta un mayor desbalance, >con categorías como _`SI1`_ y _`VS2`_ representando el 47% de los >registros, mientras que otras como _`IF`_ o _`I1`_ están poco >representadas sin alcanzar el 0.05% de los datos.
>
> ### E) Correlaciones
>
> Generamos un heatmap de las variables numéricas y observamos como los campos `price`, `carat` (peso) y las dimensiones (`x`, `y`, `z`) estan fuertemente correlacionadas (r entre 0.8 y 1.0) lo cual es lógico ya que el peso depende de las dimensiones en el mismo material, y el precio parece depender en gran medida del peso y, por ende, de las dimensiones.
>
> ![ej1_heatmap_correlacion.png](/output/ej1_heatmap_correlacion.png)
>
> Por último, se ha detectado la presencia de multicolinealidad entre algunas variables predictoras, especialmente entre x, y y z, que presentan correlaciones superiores a 0.9 en valor absoluto. Esto es esperable, ya que todas ellas miden dimensiones del mismo objeto.

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset [Diamond dataset](https://www.kaggle.com/datasets/lovishbansal123/diamond-dataset) se puede encontrar en Kaggle.
>
> He tomado el campo _`price`_ como variable objetivo o _target_ ya que es un campo que depende fuertemente de otras variables como _`carat`_, sus dimensiones (_`x`_, _`y`_, _`z`_) y la calidad (_`cut`_, _`color`_, _`clarity`_). Además determinar el precio de un producto es una práctica lógica y habitual por lo que convierte a price en una variable adecuada para predicción mediante modelos de regresión.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> | column | skewness | kurtosis | nº outliers |                                                                                                                                             |
> | :----: | :------: | :------: | :---------: | :------------------------------------------------------------------------------------------------------------------------------------------ |
> | carat  |   1.12   |   1.26   |    1889     | Distribución es leptocúrtica (kurtosis > 0) muy sesgada a la derecha (skewness > 1)                                                         |
> | depth  |  -0.08   |   5.74   |    2545     | Distribución es leptocúrtica (kurtosis > 0) bastante simétrica (skewness ≈ 0)                                                               |
> | table  |   0.8    |   2.8    |     605     | Distribución es leptocúrtica (kurtosis > 0) moderadamente sesgada a la derecha (skewness entre 0.5 y 1)                                     |
> | price  |   1.62   |   2.18   |    3540     | Distribución es leptocúrtica (kurtosis > 0) muy sesgada a la derecha (skewness > 1)                                                         |
> |   x    |   0.38   |  -0.62   |     32      | Distribución es prácticamente normal pero con tendencia aplanada (kurtosis ≈/< 0) sesgada ligeramente a la derecha (skewness entre 0 y 0.5) |
> |   y    |   2.43   |  91.21   |     29      | Distribución es fuertemente leptocúrtica (kurtosis > 0) fuertemente sesgada a la derecha (skewness > 1)                                     |
> |   z    |   1.52   |  47.09   |     49      | Distribución es fuertemente leptocúrtica (kurtosis > 0) muy sesgada a la derecha (skewness > 1)                                             |
>
> Durante el análisis de los outliers se han detectado registros iguales a 0 en los campos `x`, `y` y `z`, que representan las dimensiones del diamante.
> No considero que estos valores como outliers válidos, sino errores de registro, ya que son mediciones incompatibles con el peso (`carat`) registrado, que resulta físicamente imposible.
>
> En consecuencia, he decidido eliminar estas filas del dataset para limpiar y mejorar la calidad de los datos y del análisis.

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> Observamos que las 3 variables con mayor correlación absoluta sobre el _target_ son `carat` (R = 0.92), `x` (R = 0.89) y `z` (R = 0.87).
> Esto parece tener sentido puesto que las dimensiones y el peso del diamante son propiedades completamente relacionadas y, a su vez, el precio del diamante se determina por estas cualidades.

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> Tras un análisis mediante `.info()` no se han detectado nulos en el dataset, pero al analizar los outliers se han observado mínimos en los campos `x`, `y` y `z` iguales a 0 lo que puede suponer valores nulos reformateados erróneamente. Estos han sido eliminados del dataset ya que son inválidos y tratar de corregirlos con la mediana o la moda introduciría un sesgo no deseado.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---

> ### 2.1 Preprocesamiento
>
> Tras preparar el dataframe reutilizando código del ejercicio 1, he implementado el prepocesamiento donde primeramente se ha eliminado la columna `Unnamed` puesto que no aporta informacion relevante (row id) y las variables `x`, `y` y `z` puesto que muestran una clara multicolinealidad (|r| > 0.9) entre ellas y con el campo `carat`. De lo contrario el modelo puede confundirse ya que no entiende puesto que no es clara la variable responsable de la predicción resultando en un modelo más inestable y con mayor varianza.
>
> A conttinuación se han codificado los valores de las variables categoricas mediante _One-Hot Encoding_ (`pd.get_dummies()`).
>
> Una vez preparado el dataframe se distinguen las variables predictivas (`X`) de la variable objetivo (`y`). Luego dividir aleatoriamente (`seed 42`) sus datos en proporcion 80%-20%. El 80% de los datos serán usados para entrenamiento mientras que el 20% restante se reserva para el test del modelo predictivo.
>
> Finalmente se aplica un escalado (`StandardScaler`) ajustado únicamente sobre los datos de entrenamiento para respetar que "se desconocen" los datos de test. Luego se aplica el escalado sobre los datos de test para mantenerlos en la misma escala.
>
> Las variables resultantes son `X_train`, `X_test`, `y_train` y `y_test`.
>
> ### 2.2 Modelo A — Regresión Lineal (LinearRegression)
>
> Tras generar el grafico de residuos es notable un patron de residuos "enfermos" con una distribucíón en forma de "U". Y aunque R² mostraba un valor muy alto, los errores también eran elevados lo que era una clara señal de que el modelo no estaba funcioncionando correctamente.
>
> La solución implementada ha sido aplicar una escala logarítmica sobre la variable `price`. Con esto se ha conseguido mejorar la linealidad de las relaciones con la variable objetivo, mantener un R² cercano a 1, reducir drasticamente los errores MAE y RMSE y graficar una dsitribución de residuos más lineal.
>
> En cuanto a las variables más influyentes destaca considerablemente la variable `carat` con un coeficiente de 1.04 frente al segunda variable `clarity_VS2` (codificacion del campo `clarity`) con un coeficiente de 0.35 que no es demasiado significativo.
>
> ![ej2_residuos.png](/output/ej2_residuos.png)
>
> El análisis más útil del Ejercicio 1 ha sido el _heatmap_ de correlaciones, ya que me permitió identificar la fuerte relación entre `carat` y `price`, posteriormente confirmada en el modelo de regresión. También fue clave para detectar multicolinealidad entre `carat` y las dimensiones (`x`, `y`, `z`) y mejorar el modelo simplificando el grupo de variables predictoras. Además, el análisis de las distribuciones en los histogrmas ayudó a detectar la solución sobre la transformación logarítmica de la variable objetivo para el correcto funcionamiento del modelo.

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> - **MAE:** 0.27
> - **RMSE:** 0.33
> - **R²:** 0.8914

> Estos resultados indican un buen rendimiento del modelo. Este coeficiente de determinación implica que el modelo es capaz de explicar el 89% de la variabilidad de los precios.
> Además, los errores MAE y RMSE son relativamente bajos lo que supone un error bastante reducirdo por predicción.
>
> El modelo funciona correctamente, especialmente tras aplicar una transformación logarítmica sobre la variable `price`, lo que ha permitido mejorar la linealidad de las relaciones ya que, en comparación, el modelo inicial sin transformación presentaba una mayor dispersión de los residuos y la aparición de predicciones negativas que evidenciaban las limitaciones del modelo para capturar la relación real entre las variables.

---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---

> En este ejercicio he implementado un modelo de regresión lineal múltiple recreando las fórmulas necesarias como **OLS**, cálculo del **MAE**, el **RMSE** y **$R^2$** utilizando únicamente NumPy. El objetivo principal ha sido entender cómo se calculan los coeficientes del modelo sin usar librerías avanzadas como Scikit-Learn.
>
> Los coeficientes ajustados son bastante cercanos a los valores reales de referencia en el enunciado, lo que es buen indicio de que la implementación es correcta y que el modelo es capaz de aproximar bien la relación entre las variables.
>
> En cuanto a las métricas, tanto el MAE como el RMSE están dentro del rango esperado (±0.20). Por otro lado, el valor de $R^2=0.6897$ es algo más bajo que el de referencia (~0.80), pero sigue siendo razonable, ya que los datos incluyen ruido aleatorio que dificulta una predicción perfecta.
>
> Finalmente, la gráfica de valores reales frente a predichos muestra que los puntos se distribuyen alrededor de la recta, lo que indica que el modelo recoge la tendencia general correctamente.
>
> ![ej3_predicciones.png](/output/ej3_predicciones.png)
>
> Este ejercicio me ha servido para comprender y visualizar en mi cabeza el proceso de un modelo de regresion lineal para ser entrenado y crear predicciones sobre nuevos datos. Además, programar manualmente funciones como **OLS** me ha ayduado a comprender el motivo y el objetivo de sus operaciones y transformaciones.

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> Un modelo de regresión lineal se determina por la **Ecuación de la Recta**:
>
> $$Y = \beta_0 + \beta_1 X + \epsilon$$
>
> Donde $\beta_0$ (Intercepto) indica la altura de la recta y $\beta_1$ la pendiente.
> Cuando la regresión es múltiple se añade un nuevo término a la suma por cada variable $X_i$ y su coeficiente $\beta_i$ que indica su influencia sobre el resultado final.
>
> $$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p + \epsilon$$
>
> Pero la funcion de **Mínimos Cuadrados Ordinarios (OLS)**, que calcula los valores de los coeficientes que minimizan el error cuadrático medio, no es capaz de calcular el intercepto $\beta_0$ puesto que esta no acompaña a ninguna variable predictora de $X$.
>
> $$β = (XᵀX)⁻¹ Xᵀy$$
>
> Por ello se introduce una nueva columna en la matriz $X$ con una valor constante como variable predictora cuyo coeficiente sea el intercepto y que no modifique la ecuación (multiplicación por 1).
>
> $$Y = \beta_0·1 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p + \epsilon$$
>
> De este modo, tomando _p_ como el número de variables predictoras, la matriz $X$ se expande hasta (n, _p_+1).

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parámetro | Valor real | Valor ajustado |
| --------- | :--------: | :------------: |
| β₀        |    5.0     |   `4.864995`   |
| β₁        |    2.0     |   `2.063618`   |
| β₂        |    -1.0    |  `-1.117038`   |
| β₃        |    0.5     |   `0.438517`   |

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

| Parámetro | Valor referencia | Valor obtenido |
| --------- | :--------------: | :------------: |
| MAE       |       1.20       |   `1.166462`   |
| RMSE      |       1.50       |   `1.461243`   |
| $R^2$     |       0.80       |   `0.689672`   |

> Sí, los valores obtenidos se encuentran dentro del margen de error (±0.20) sobre los valores de referencia.

---

## Ejercicio 4 — Series Temporales

---

Añade aqui tu descripción y analisis:

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> _Escribe aquí tu respuesta_

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> _Escribe aquí tu respuesta_

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> _Escribe aquí tu respuesta_

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> _Escribe aquí tu respuesta_

---

_Fin del documento de respuestas_
