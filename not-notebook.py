#%% md

#%_ Bloque de introducción

#%% md

# Datasets
## Preparación
### Librerías

#%%

import pandas as pd
import seaborn as sns
import numpy as np

# Para las medidas extrínsecas
from sklearn import metrics
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.cluster import contingency_matrix

#%% md

### Funciones de apoyo


#### Función de carga del dataset
#%_Se crea una función para simplificar la carga del dataset, que acepta varios parámetros:
#%_ - dataset_url: cadena con la ruta al recurso desde donde cargar el dataset.
#%_ - separator (opcional): caracter de división en el origen del dataset.
#%_ - class_position (opcional): ubicación en el dataset de la clase.
#%_ - remove (opcional): qué atributos no se van a usar del dataset para eliminarlos al cargar.

#%%

def load_dataset(dataset_url: str, separator: str = '\s+', class_position: int = None, remove: tuple = None,
                 limpiarNA: bool = True, describe: bool = False):
    """Load a dataset from a specified url into a pandas DataFrame.

    :param str dataset_url: an url from archive.ics.uci.edu
    :param int class_position: column index where classes are defined (starts by 0)
       if left empty (None), no prediction class will be used (intrinsic case).
    """
    # Load dataset as a pandas DataFrame from a specified url.
    dataset = pd.read_csv(dataset_url, sep=separator, header=None)

    # Limpieza de datos
    # El dataset de extrínseco tiene un valor no deseado en una instancia, en la potencia, "?"
    dataset.replace(to_replace="?", value=np.nan, inplace=True)
    #  Eliminamos las líneas con valores vacíos
    if limpiarNA:
        dataset.dropna(inplace=True)

    # Antes de recortar el dataset, lo mostramos
    if describe:
        print(dataset.describe())

    # Extrinsic case, dataset comes with its classes.
    if class_position is not None:
        # Extract classes.
        classes = dataset.iloc[:, class_position]
        # Remove classes from the dataset.
        dataset = dataset.drop([class_position, ], axis=1)

    # Remove attributes.
    if remove is not None:
        dataset.drop(remove, axis=1, inplace=True)

    # Intrinsic case, dataset has no classes.
    else:
        classes = None

    return classes, dataset


#%% md

#### Función de visualización
#%_Creamos también una función para simplificar la visualización del dataset. Recibe dos parámetros:
#%_ - atributos: El DataFrame con los atributos a representar
#%_ - clase (opcional): El DataFrame con la clase de cada instancia.

#%%

def plot_dataset(atributos: pd.DataFrame, clase: pd.DataFrame=None) -> None:
    if clase is not None:
        dataset = pd.concat([clase,atributos], axis=1)
    else:
        dataset = atributos
    sns.pairplot(dataset, hue=1)


#%% md

#### Función de cálculo de las medidas extrínsecas
#%_Función que calcula varias medidas cualitativas del agrupamiento.
#%_ - ARI mide la similaridad entre las clases y los predichos
#%_ - Información mutua
#%_ - Homogeneidad (todos los valores predichos son del clúster correcto)
#%_ - Completación (todos los valores de una clase se predicen en el mismo clúster)
#%_ - Medida V (media armónica de homogeneidad y completación). Parámetro beta (por defecto 1) para ponderar
#%_ - Fowlkes-Mallows es la media geométrica de las parejas precision-recall
#%_ - Silhouette
#%_ - Calinski-Harabasz
#%_ - Davies-Bouldin

#%%

def mediciones_extrinsecas(exDs, y_true, y_pred):
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    im = metrics.adjusted_mutual_info_score(y_true, y_pred)
    hom = metrics.homogeneity_score(y_true, y_pred)
    com = metrics.completeness_score(y_true, y_pred)
    vm = metrics.v_measure_score(y_true, y_pred)
    fm = metrics.fowlkes_mallows_score(y_true, y_pred)

    sil = metrics.silhouette_score(exDs, y_true, metric='euclidean')
    ch = metrics.calinski_harabasz_score(exDs, y_true)
    db = davies_bouldin_score(exDs, y_true)
    metricas = {"ari": ari, "im": im, "hom":hom, "com":com, "vm":vm, "fm":fm, "sil":sil, "ch":ch, "db":db}
    return metricas


#%% md

## Selección

### Dataset extrínseco

#### Auto-mpg


#%%

dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
# print("El dataset mpg:") # Si se activa, añadir parámetro describe=True a este load_dataset
extrinsic_classes, extrinsic_dataset = load_dataset(dataset_url, class_position=1, remove=[0, 4, 5, 6, 7])

#%%

plot_dataset(extrinsic_dataset,extrinsic_classes)

#%% md

#%_ El origen de este dataset se remonta a datos usados en 1983 por la <i>American Statistical
#% Association Exposition</i> y que se conservan en la Universidad de Carnegie Mellon, al que le
#% faltan 8 instancias que se eliminaron para homogeneizar el dataset, ya que carecían del campo mpg.

#% El dataset consta de:

#%%

print(extrinsic_dataset.shape)

#%% md

#% - 398 instancias -------------- OJO: PArece que salen 392
#% - 8 atributos, que son:
#%     · mpg (millas por galón de combustible): de tipo continuo.
#%     · cylinders (cilindros): discreto multi evaluado.
#%     · displacement (cilindrada): continuo.
#%     · horsepower (caballos de potencia): continuo.
#%     · weight (peso): continuo
#%     · acceleration (aceleración): continuo
#%     · model-year (año del modelo): discrto multi evaluado.
#%     · origin (origen): discreto multi evaluado.
#%     · car name (nombre del coche): cadena (único para cada instancia)

#%  Para el estudio que nos ocupa vamos a predecir el número de cilindros basándonos en la cilindrada y la potencia.
#%  Se descartan el resto de valores para mantener baja la dimensión del vector descriptor y simplificar así los
#%  cálculos.
#%  Los datos vienen casi listos para trabajar con ellos. No se detectan campos vacíos:

#%%

print(extrinsic_dataset.isnull().any())

#%% md

#%  Sin embargo, en la potencia hay un valor anómalo, un "?" usado donde se desconocía el dato, por lo que se
#%  ha incorporado a la función de carga de datos un filtro para eliminarlo, ajustable por parámetro (limpiarNA)
#%
#%  Vamos a observar la distribución de nuestra clase:

#%%

sns.distplot(extrinsic_classes)

#%% md

#%  Como se adivinaba en las gráficas anteriores, se observa una marcada preponderancia de los
#%  valores de cilindros 4,6 y 8.

#%%

## Dataset intrínseca

#%_ Hemos escogido el dataset *tae.csv*. Este trata de XXX con los atributos siguientes:
#%_ - aaa1
#%_ - aaa2
#%_ - ...

#%_ Cargamos nuestro dataset (*intrinsic_dataset*):

#%%

dataset_url = 'https://raw.githubusercontent.com/Grupo-MasterAI-ANS/actividad/master/datasets/tae.csv'
_, intrinsic_dataset = load_dataset(dataset_url, separator=',')

#%% md

#%_ Podemos ver la relación siguiente entre atríbutos:

#%%

plot_dataset(atributos = intrinsic_dataset)

#%% md

#%_ blablabla

#%% md

# Análisis dataset extrínseca
## Algoritmos

#%% md

### Algoritmo k-means

#%_ Observando los datos es evidente que el número óptimo de clústers para K-means es 3.
#%%

kmeans_model = KMeans(n_clusters=3, random_state=1).fit(extrinsic_dataset)
labels_pred = kmeans_model.labels_
meds_kmeans_ext = mediciones_extrinsecas(extrinsic_dataset, labels_true, labels_pred)
for key,value in meds_kmeans_ext["metricas"].items():
    print(key,":",value)
#%% md

### Algoritmo 2

#%%



#%% md

### Algoritmo 3

#%%



#%% md

### Algoritmo 4

#%%



#%% md

### Algritmo 5

#%%



#%% md

## Comparación algoritmos

#%%



#%% md

# Análisis dataset intrínseca¶
## Algoritmos

#%% md

### Algoritmo k-means

#%%

# Selecciona el mejor Kmeans de entre los posibles números de clústers pasados por parámetro.
# Ejemplo:
#    num_clusters = [2,3,4,5,6]
#    superK = mejor_Kmeans(extrinsic_dataset,extrinsic_classes,num_clusters)
def mejor_Kmeans(atributos: pd.DataFrame, clase: pd.DataFrame, clusters_K_means: tuple = [3]):
    media = 0
    clusters = 0
    modelo = ""
    metricas = {"ari": 0, "im": 0, "hom": 0, "com": 0, "vm": 0, "fm": 0, "sil": 0, "ch": 0, "db": 0}
    for clus in clusters_K_means:
        # Obtenemos el modelo kmeans
        kmeans_model = KMeans(n_clusters=clus, random_state=1).fit(atributos)
        # Las clases predichas
        labels_pred = kmeans_model.labels_
        # Saco las medidas y su media
        mets = mediciones_extrinsecas(extrinsic_dataset, labels_true, labels_pred)

        val = (mets["ari"] + mets["im"] + mets["hom"] + mets["com"] + mets["vm"] + mets["fm"]) / 6
        print("Media de", clus, ": ", val)
        if val > media:
            media = val
            clusters = clus
            modelo = kmeans_model
            metricas = mets
    return {"modelo":modelo,"metricas":metricas,"clusters":clusters,"media":media}

#%% md

### Algoritmo 2

#%%



#%% md

### Algoritmo 3

#%%



#%% md

### Algoritmo 4

#%%



#%% md

### Algoritmo 5

#%%



#%% md

## Comparación algoritmos

#%%



#%% md

# Conclusión
