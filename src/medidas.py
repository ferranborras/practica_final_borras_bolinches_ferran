import numpy as np
import pandas as pd


def media_evolve(lista_datos: list):
    if len(lista_datos) == 0:
        print("No se han proporcionado suficientes datos")
        return None

    suma = 0
    for d in lista_datos:
        suma += d
    return suma / len(lista_datos)


def mediana_evolve(lista_datos: list):
    if len(lista_datos) == 0:
        print("No se han proporcionado suficientes datos")
        return None

    sorted_datos = sorted(lista_datos)
    if len(sorted_datos) % 2 == 1:
        return sorted_datos[len(sorted_datos) // 2]

    return media_evolve([
        sorted_datos[len(sorted_datos) // 2 - 1],
        sorted_datos[len(sorted_datos) // 2],
    ])


def percentil_evolve(lista_datos: list, percentil: int):
    if len(lista_datos) == 0:
        print("No se han proporcionado suficientes datos")
        return None
    
    if not 0 <= percentil <= 100:
        print("Percentil debe estar entre 0 y 100")
        return None

    sorted_datos = sorted(lista_datos)
    return sorted_datos[int((len(sorted_datos)-1) * percentil / 100)]


def varianza_evolve(lista_datos: list):
    if len(lista_datos) <= 1:
        print("No se han proporcionado suficientes datos")
        return None

    suma = 0
    media = media_evolve(lista_datos)
    for d in lista_datos:
        suma += (d - media) ** 2
    return suma / (len(lista_datos)-1)


def desviacion_evolve(lista_datos: list):
    if len(lista_datos) <= 1:
        print("No se han proporcionado suficientes datos")
        return None

    return varianza_evolve(lista_datos)**0.5


def IQR_evolve(lista_datos: list):
    if len(lista_datos) == 0:
        print("No se han proporcionado suficientes datos")
        return None

    Q1 = percentil_evolve(lista_datos, 25)
    Q3 = percentil_evolve(lista_datos, 75)
    IQR = Q3 - Q1

    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    return IQR, lim_inf, lim_sup


if __name__ == "__main__":

    np.random.seed(42)

    edad = list(np.random.randint(20, 60, 100))
    salario =  list(np.random.normal(45000, 15000, 100))
    experiencia = list(np.random.randint(0, 30, 100))

    np.random.seed(42)

    df = pd.DataFrame({
        'edad': np.random.randint(20, 60, 100),
        'salario': np.random.normal(45000, 15000, 100),
        'experiencia': np.random.randint(0, 30, 100)
    })

    print(df.describe())

    print()
    print("Edad")
    print("----------------------------------")
    print(media_evolve(edad))
    print(mediana_evolve(edad))
    print(percentil_evolve(edad, 50))
    print(varianza_evolve(edad))
    print(desviacion_evolve(edad))
    print(IQR_evolve(edad))

    print()
    print("Salario")
    print("----------------------------------")
    print(media_evolve(salario))
    print(mediana_evolve(salario))
    print(percentil_evolve(salario, 50))
    print(varianza_evolve(salario))
    print(desviacion_evolve(salario))
    print(IQR_evolve(salario))

    print()
    print("Experiencia")
    print("----------------------------------")
    print(media_evolve(experiencia))
    print(mediana_evolve(experiencia))
    print(percentil_evolve(experiencia, 50))
    print(varianza_evolve(experiencia))
    print(desviacion_evolve(experiencia))
    print(IQR_evolve(experiencia))