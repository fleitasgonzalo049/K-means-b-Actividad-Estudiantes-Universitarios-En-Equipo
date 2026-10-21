# Importa Pandas para trabajar con tablas (DataFrames)
import pandas as pd

# Importa Matplotlib para generar graficos
import matplotlib.pyplot as plt

# Importa StandardScaler para estandarizar las variables
from sklearn.preprocessing import StandardScaler

# Importa KMeans para realizar el clustering
from sklearn.cluster import KMeans
# Lista de diccionarios con los datos de los estudiantes
estudiantes = [
    {'edad': 42, 'horas_estudio': 2, 'promedio_academico': 8.87},
    {'edad': 26, 'horas_estudio': 8, 'promedio_academico': 9.11},
    {'edad': 56, 'horas_estudio': 5, 'promedio_academico': 7.29},
    {'edad': 29, 'horas_estudio': 14, 'promedio_academico': 7.27},
    {'edad': 43, 'horas_estudio': 8, 'promedio_academico': 8.5},
    {'edad': 34, 'horas_estudio': 10, 'promedio_academico': 4.45},
    {'edad': 51, 'horas_estudio': 3, 'promedio_academico': 6.12},
    {'edad': 22, 'horas_estudio': 12, 'promedio_academico': 4.33},
    {'edad': 47, 'horas_estudio': 6, 'promedio_academico': 7.0},
    {'edad': 38, 'horas_estudio': 9, 'promedio_academico': 4.75},
    {'edad': 60, 'horas_estudio': 2, 'promedio_academico': 9.0},
    {'edad': 31, 'horas_estudio': 11, 'promedio_academico': 2.0},
    {'edad': 45, 'horas_estudio': 4, 'promedio_academico': 8.0},
    {'edad': 27, 'horas_estudio': 13, 'promedio_academico': 3.2},
    {'edad': 50, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 36, 'horas_estudio': 7, 'promedio_academico': 5.0},
    {'edad': 40, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 24, 'horas_estudio': 15, 'promedio_academico': 3.0},
    {'edad': 55, 'horas_estudio': 3, 'promedio_academico': 9.0},
    {'edad': 33, 'horas_estudio': 10, 'promedio_academico': 3.0},
    {'edad': 48, 'horas_estudio': 4, 'promedio_academico': 8.0},
    {'edad': 28, 'horas_estudio': 12, 'promedio_academico': 2.0},
    {'edad': 52, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 30, 'horas_estudio': 11, 'promedio_academico': 2.0},
    {'edad': 46, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 35, 'horas_estudio': 9, 'promedio_academico': 4.0},
    {'edad': 41, 'horas_estudio': 7, 'promedio_academico': 5.0},
    {'edad': 23, 'horas_estudio': 14, 'promedio_academico': 1.0},
    {'edad': 53, 'horas_estudio': 3, 'promedio_academico': 8.0},
    {'edad': 32, 'horas_estudio': 10, 'promedio_academico': 3.0},
    {'edad': 49, 'horas_estudio': 4, 'promedio_academico': 8.0},
    {'edad': 25, 'horas_estudio': 13, 'promedio_academico': 2.0},
    {'edad': 54, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 37, 'horas_estudio': 8, 'promedio_academico': 4.0},
    {'edad': 39, 'horas_estudio': 7, 'promedio_academico': 5.0},
    {'edad': 21, 'horas_estudio': 15, 'promedio_academico': 9.0},
    {'edad': 58, 'horas_estudio': 2, 'promedio_academico': 9.0},
    {'edad': 44, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 29, 'horas_estudio': 12, 'promedio_academico': 2.0},
    {'edad': 50, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 34, 'horas_estudio': 9, 'promedio_academico': 3.0},
    {'edad': 42, 'horas_estudio': 7, 'promedio_academico': 6.0},
    {'edad': 26, 'horas_estudio': 14, 'promedio_academico': 3.0},
    {'edad': 57, 'horas_estudio': 3, 'promedio_academico': 9.0},
    {'edad': 30, 'horas_estudio': 11, 'promedio_academico': 2.0},
    {'edad': 47, 'horas_estudio': 4, 'promedio_academico': 7.0},
    {'edad': 31, 'horas_estudio': 10, 'promedio_academico': 3.0},
    {'edad': 45, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 27, 'horas_estudio': 13, 'promedio_academico': 1.0},
    {'edad': 55, 'horas_estudio': 5, 'promedio_academico': 8.0}
]


# Convierte la lista de diccionarios en un DataFrame
df = pd.DataFrame(estudiantes)

# Muestra las primeras 10 filas para verificar que los datos se cargaron correctamente
print(df.head(10))


# Selecciona las columnas que participaran en el clustering
X = df[["edad", "horas_estudio", "promedio_academico"]]


# Crea el objeto que realizara la estandarizacion
scaler = StandardScaler()

# Calcula los parametros de escalado y transforma los datos
X_escalados = scaler.fit_transform(X)


# Crea el modelo KMeans indicando que queremos 4 clusters
kmeans = KMeans(n_clusters=4)

# Ejecuta el algoritmo y devuelve el cluster asignado a cada estudiante
clusters = kmeans.fit_predict(X_escalados)

# Agrega al DataFrame una nueva columna con el cluster de cada estudiante
df["cluster"] = clusters


# Agrupa los estudiantes por cluster y calcula las medias de cada grupo
df.groupby("cluster").mean()

# Muestra las primeras filas del DataFrame ya con la columna cluster
print(df.head())

# Muestra el resumen estadistico de cada cluster
print(df.groupby("cluster").mean())


# Genera un grafico de dispersion
plt.scatter(
    df["edad"],                   # Eje X
    df["promedio_academico"],     # Eje Y
    c=df["cluster"]               # Color segun el cluster
)

# Muestra una barra de colores para identificar los clusters
plt.colorbar()

# Etiqueta del eje X
plt.xlabel("Edad")

# Etiqueta del eje Y
plt.ylabel("Promedio Academico")

# Titulo del grafico
plt.title("Clusters de Estudiantes")

# Muestra el grafico
plt.show()



