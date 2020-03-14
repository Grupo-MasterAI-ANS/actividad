#%% md

"""  #    
Bloque de introducción

"""  #

#%% md

# Datasets
## Preparación
### Librerías

#%%

import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster, cut_tree, centroid

#%% md

"""  #    
### Descarga
Preparamos una función genérica de descarga de los datasets y de su preparación.     
Esta nos permite escoger los atributos que usaremos asi como extraer a un variable aparte las clases en caso de estar disponibles.

"""  #


#%%

def load_dataset(dataset_url: str, attributes: dict, separator: str = '\s+', class_position: int = None):
    """Load a dataset from a specified url into a pandas DataFrame.

    :param str dataset_url: an url from a text based dataset
    :param dict attributes: attributes to keep in dictionary form:
        key: attribute position, value: attribute name
    :param str separator: file separator.
    :param int class_position: column index where classes are defined (starts by 0)
        if left empty (None), no prediction class will be used (intrinsic case).
    """
    # Load dataset as a pandas DataFrame from a specified url.
    dataset = pd.read_csv(dataset_url, sep=separator, header=None)

    # Add class index to the indexes to extract.
    if class_position is not None:
        attributes[class_position] = 'classes'

    # Keep only desired attributes and classes.
    dataset = dataset[attributes]

    # Force all values to be numeric.
    for (column, values) in dataset.iteritems():
        # Do not transform classes.
        if column == class_position:
            continue

        # Coerce transforms non-numeric values into NaN.
        dataset[column] = pd.to_numeric(values, errors='coerce')

    # Remove all NaN rows.
    dataset.dropna(inplace=True)

    # Extrinsic case, dataset comes with its classes.
    if class_position is not None:
        # Extract classes.
        classes = dataset[class_position]
        # Remove classes from attributes.
        dataset.drop(class_position, axis=1, inplace=True)

    # Intrinsic case, dataset has no classes.
    else:
        classes = None

    return classes, dataset


#%% md

"""  #    
### Visualización
Usaremos una función común para presentar los datos, tanto como si están clasificados o no.    
También, en caso de usar más de dos atributos del dataset, usaremos el *pairplot* de seaborn para presentar los atributos de dos en dos.

"""  #


#%%

def plot_dataset(dataset: pd.DataFrame, classes: np.array = None) -> None:
    # For bi-dimensional dataset, use a simple plot.
    if len(dataset.columns) == 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=classes)

    # For extra-dimensional dataset compare attributes.
    else:
        if classes is not None:
            # Clone dataset to avoid modifying the original one.
            dataset = dataset.copy()
            dataset['classes'] = classes.astype(str)
            sns.pairplot(dataset, hue='classes')
        else:
            sns.pairplot(dataset)


#%% md

## Métricas
### Métricas extrínsecas

#%%

def matriz_confusion(cat_real, cat_pred):
    cats = np.unique(cat_real)
    clusts = np.unique(cat_pred)
    mat = np.array([[np.sum(np.logical_and(cat_real == cats[i], cat_pred == clusts[j]))
                     for j in np.arange(clusts.size)]
                    for i in np.arange(cats.size)])
    return (mat)


#%%

def medida_error(mat):
    assign = np.sum([np.max(mat[l, :]) for l in np.arange(mat.shape[0])])
    return 1 - assign / float(np.sum(mat))


def medida_precision(mat, l, k):
    return mat[l, k] / sum(mat[:, k])


def medida_recall(mat, l, k):
    return mat[l, k] / sum(mat[l, :])


def medida_pureza(mat):
    totales = np.sum(mat, axis=0) / float(np.sum(mat))
    return np.sum([
        totales[k] * np.max(mat[:, k] / float(np.sum(mat[:, k])))
        for k in np.arange(mat.shape[1])
    ])


#%%

def medida_f1_especifica(mat, l, k):
    prec = medida_precision(mat, l, k)
    rec = medida_recall(mat, l, k)
    if (prec + rec) == 0:
        return 0
    else:
        return 2 * prec * rec / (prec + rec)


def medida_f1(mat):
    totales = np.sum(mat, axis=1) / float(np.sum(mat))
    assign = np.sum([
        totales[l] * np.max([
            medida_f1_especifica(mat, l, k)
            for k in np.arange(mat.shape[1])
        ])
        for l in np.arange(mat.shape[0])
    ])
    return assign


#%%

def medida_entropia(mat):
    totales = np.sum(mat, axis=0) / float(np.sum(mat))
    relMat = mat / np.sum(mat, axis=0)
    logRelMat = relMat.copy()
    logRelMat[logRelMat == 0] = 0.0001  # Evita el logaritmo de 0. Inofensivo pues luego desaparece al multiplicar por 0
    logRelMat = np.log(logRelMat)
    return -np.sum([
        totales[k] * np.sum([
            relMat[l, k] * logRelMat[l, k]
            for l in np.arange(mat.shape[0])
        ])
        for k in np.arange(mat.shape[1])
    ])


#%% md

"""  #     
####  Agrupación métricas extrínsecas
La función a continuación nos permite generar un diccionario con todas las métricas intrínsecas y poder compararlas entre algoritmos.

"""  #


#%%

def calculate_extrinsic_metrics(real_classes, predicted_classes):
    confusion_matrix = matriz_confusion(real_classes, predicted_classes)

    return {
        'Error': medida_error(confusion_matrix),
        'Pureza': medida_pureza(confusion_matrix),
        'F1': medida_f1(confusion_matrix),
        'Entropía': medida_entropia(confusion_matrix),
        'Información mútua': metrics.mutual_info_score(real_classes, predicted_classes)
    }


#%% md

"""  #    
### Métricas intrínsecas
Añadimos las funciones de cálculo de métricas intrínsecas no disponibiles directamente en python o por lo menso en sklearn.

"""  #


#%%

def RMSSTD_score(dataset, prediction, centers):
    # Extract all individual predicted classes.
    labels = np.unique(prediction)

    numerator = np.sum([
        np.sum(np.sum(dataset[prediction == label] - centers[label], axis=1) ** 2)
        for label in labels
    ])

    denominator = dataset.shape[1] * np.sum([
        np.sum(prediction == label) - 1
        for label in labels
    ])

    return np.sqrt(numerator / denominator)


#%%

def r2_score(dataset, prediction, centroids):
    """
    An intrinsic R² score metric, as sklearn one is extrinsic only.
    """
    attributes_mean = np.mean(dataset, axis=0)
    labels = np.sort(np.unique(prediction))
    numerator = np.sum([
        np.sum(np.sum(dataset[prediction == label] - centroids[label], axis=1) ** 2)
        for label in labels
    ])
    denominator = np.sum(np.sum(dataset - attributes_mean, 1) ** 2)

    return 1 - numerator / denominator


#%%

def distancia_euclidiana(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def distance_matrix(X, distancia):
    mD = np.zeros((X.shape[0], X.shape[0]))
    for pair in it.product(np.arange(X.shape[0]), repeat=2):
        mD[pair] = distancia(X[pair[0], :], X[pair[1], :])
    return mD


def medida_I(dataset, prediction, centers, distance_function, p=1):
    attributes_mean = np.mean(dataset, axis=0)
    labels = np.sort(np.unique(prediction))
    distance_max = np.max(distance_matrix(centers, distance_function))

    num = np.sum([distance_function(instance, attributes_mean) for instance in dataset.values])

    den = len(labels) * np.sum([
        np.sum([
            distance_function(dataset.iloc[i], centers[k])
            for i in np.arange(dataset.shape[0])[prediction == labels[k]]
        ])
        for k in np.arange(centers.shape[0])
    ])

    return (num / den * distance_max) ** p


#%% md

"""  #    
####  Agrupación métricas intrínsecas
La función a continuación nos permite generar un diccionario con todas las métricas intrínsecas y poder compararlas entre algoritmos.

"""  #


#%%

def calculate_intrinsic_metrics(dataset, model, distancia):
    return {
        'RMSSTD': RMSSTD_score(dataset, model['prediction'], model['centers']),
        'R²': r2_score(dataset, model['prediction'], model['centers']),
        'Silueta': metrics.silhouette_score(dataset, model['prediction']),
        'Calinski Harabasz': metrics.calinski_harabasz_score(dataset, model['prediction']),
        'Medida I': medida_I(dataset, model['prediction'], model['centers'], distancia),
        'Davies Bouldin': metrics.davies_bouldin_score(dataset, model['prediction'])
    }


#%% md

## Selección

#%% md

"""  #    
### Dataset extrínseca
   This dataset is a slightly modified version of the dataset provided in
   the StatLib library.  In line with the use by Ross Quinlan (1993) in
   predicting the attribute "mpg", 8 of the original instances were removed 
   because they had unknown values for the "mpg" attribute.  The original 
   dataset is available in the file "auto-mpg.data-original".

   "The data concerns city-cycle fuel consumption in miles per gallon,
    to be predicted in terms of 3 multivalued discrete and 5 continuous
    attributes." (Quinlan, 1993)

**Number of Instances:** 398

**Number of Attributes:** 9 including the class attribute

**Attribute Information:**

    1. mpg:           continuous
    2. cylinders:     multi-valued discrete
    3. displacement:  continuous
    4. horsepower:    continuous
    5. weight:        continuous
    6. acceleration:  continuous
    7. model year:    multi-valued discrete
    8. origin:        multi-valued discrete
    9. car name:      string (unique for each instance)

"""  #

#%%

# Cargamos el dataset.
dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
# attributes = {0: 'mpg', 2: 'displacement', 3: 'horsepower', 4: 'weight', 5: 'acceleration'}
attributes = {3: 'horsepower', 5: 'acceleration'}
extrinsic_classes, extrinsic_dataset = load_dataset(dataset_url, attributes, class_position=1)

# Soporte para las métricas
extrinsic_metrics = {}

#%%

plot_dataset(extrinsic_dataset, extrinsic_classes)

#%% md

"""  #    
blablabla

"""  #

#%% md

"""  #    
## Dataset intrínseca
El dataset intrínseca **Aggregations** está generado de manera artificial por: *A. Gionis, H. Mannila, and P. Tsaparas, Clustering aggregation. ACM Transactions on Knowledge Discovery from Data (TKDD), 2007*


Este dataset está compuesto por 788 observaciones de 2 variables que abarcan un amplio rango numérico. En el conjunto de datos existen entre 5 a 7 grupos que se distribuyen en zonas particulares del rango de valores de las variables.


Cargamos nuestro dataset (*intrinsic_dataset*):

"""  #

#%%

# Cargamos el dataset.
dataset_url = 'http://cs.joensuu.fi/sipu/datasets/Aggregation.txt'
attributes = {0: 'dim 1', 1: 'dim 2'}
_, intrinsic_dataset = load_dataset(dataset_url, attributes)

# Soporte para las métricas
intrinsic_metrics = {}

#%% md

"""  #    
Visualizamos el dataset en 2-D.

"""  #

#%%

plot_dataset(intrinsic_dataset)

#%% md

"""  #    
Destacamos que se podría clasificar con 4, 5 o con 7 clusters.

"""  #

#%% md

"""  #    
# Algoritmos
Preparamos funciones 'herramienta' para cada algoritmo de forma a poder analizarlos.

"""  #

#%% md

## K-Means

#%% md

"""  #    
Métrica R cadrado. No usamos directamente la de sklean al esta necesitar la clases reales.
Nos permite valorar el ratio de distancia intraclúster con respecto a la distancia interclúster.

"""  #

#%% md

"""  #    
Función para generar gráficamente la evolución de las métricas R² y Silueta según el número de cluters, de forma a escoger el número de clusters óptimo, usando la técnica del codo.

"""  #


#%%

def kmeans_plot_clusters_selection(dataset: pd.DataFrame, max_clusters: int = 10):
    dataset = np.array(dataset)
    silhouette_values = []
    r2_values = []
    min_clusters = 2

    for k in np.arange(min_clusters, max_clusters):
        model = KMeans(n_clusters=k).fit(dataset)
        prediction = model.predict(dataset)
        centroids = model.cluster_centers_

        silhouette_values += [metrics.silhouette_score(dataset, prediction)]
        r2_values += [r2_score(dataset, prediction, centroids)]

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(np.arange(min_clusters, max_clusters), silhouette_values, linestyle='-', marker='o')
    ax[0].set_xlabel("Número de clústeres")
    ax[0].set_ylabel("Medida de ancho de silueta")

    ax[1].plot(np.arange(min_clusters, max_clusters), r2_values, linestyle='-', marker='o')
    ax[1].set_xlabel("Número de clústeres")
    ax[1].set_ylabel("Medida de R cuadrado")


#%% md

# Análisis dataset extrínseca
## Algoritmos

#%% md

"""  #    
### Algoritmo k-means
#### Selección del número de clusters

A fin de implementar el modelo de K-Medios, comencemos por determinar la cantidad óptima de centroides a utilizar a partir del Método del Codo.

"""  #

#%%

kmeans_plot_clusters_selection(extrinsic_dataset)

#%% md

"""  #    
Vemos un "codo" pronunciando con 4 clusters, pero el ancho de silueta es "mínimo" a partir de 6 clusters.     
Escogemos 5 clusters como punto intermedio.

"""  #

#%% md

#### Ejecución del algoritmo

#%%

model = KMeans(n_clusters=5).fit(extrinsic_dataset)
prediction = model.predict(extrinsic_dataset)
extrinsic_metrics['k-means'] = calculate_extrinsic_metrics(extrinsic_classes, prediction)

plot_dataset(extrinsic_dataset, prediction)

#%% md

"""  #     
blablabla

"""  #

#%% md

### Algoritmo Jerárquico Aglomerativo
#### Ejecución del algoritmo

#%%

model = linkage(extrinsic_dataset, 'average')
prediction = cut_tree(model, n_clusters=5).flatten()
extrinsic_metrics['Jerárquico'] = calculate_extrinsic_metrics(extrinsic_classes, prediction)

plot_dataset(extrinsic_dataset, prediction)

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

# Análisis dataset intrínseca
## Algoritmos

#%% md

"""  #    
### Algoritmo k-means
#### Selección del número de clusters

A fin de implementar el modelo de K-Medios, comencemos por determinar la cantidad óptima de centroides a utilizar a partir del Método del Codo.

"""  #

#%%

kmeans_plot_clusters_selection(intrinsic_dataset)

#%% md

"""  #    
Según el procedimiento del codo, escogeríamos entre 5 y 7 clusters

"""  #

#%% md

"""  #
#### Ejecución del algoritmo
Ejecutamos la predicción de k-means con 5 clusters y visualizamos la agrupación generada.


"""  #

#%%

model = KMeans(n_clusters=5).fit(intrinsic_dataset)
fitting = {
    'prediction': model.predict(intrinsic_dataset),
    'centers': model.cluster_centers_
}
intrinsic_metrics['k-means'] = calculate_intrinsic_metrics(intrinsic_dataset, fitting, distancia_euclidiana)

plot_dataset(intrinsic_dataset, fitting['prediction'])

#%% md

"""  #    
Vemos que mientras se han logrado aislar algunos grupos, otros claramente se han quedado a medias.

"""  #

#%% md

### Algoritmo Jerárquico Aglomerativo

#%%

model = linkage(intrinsic_dataset, 'average')

fitting = {
    'prediction': cut_tree(model, n_clusters=5).flatten(),
    # TODO: de momento pongo esto, porqué el _centers_ parece que es solo para kmeans
    # y entonces no se que usar para las métricas que necesitan centros.
    'centers': centroid(cut_tree(model, n_clusters=5))

}

# intrinsic_metrics['Jerárquico'] = calculate_intrinsic_metrics(intrinsic_dataset, fitting, distancia_euclidiana)


# plot_dataset(intrinsic_dataset, fitting['prediction'])

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
### Métricas dataset extínseco

#%%

display(pd.DataFrame(extrinsic_metrics))

#%% md

### Métricas dataset intínseco

#%%

display(pd.DataFrame(intrinsic_metrics))

#%% md

# Conclusión
