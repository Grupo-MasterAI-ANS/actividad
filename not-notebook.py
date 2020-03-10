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

#%% md

### Función de descarga
#%_ blablabla

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

### Función de visualización
#%_ blablabla
#%_ la idea es que esta función sea parametrizable y que pueda colorear los clusters

#%%

def plot_dataset(atributos: pd.DataFrame, clase: pd.DataFrame=None) -> None:
    if clase is not None:
        dataset = pd.concat([clase,atributos], axis=1)
    else:
        dataset = atributos
    sns.pairplot(dataset, hue=1)


#%% md

## Selección

#%% md

### Dataset extrínseca
#%_ lo cilindros están en posición 1 (partiendo de 0)
#%_ no los he eliminado para que veamos el hue

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
print(extrinsic_dataset.shape)
#% - 398 instancias
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
print(extrinsic_dataset.isnull().any())
#%  Sin embargo, en la potencia hay un valor anómalo, un "?" usado donde se desconocía el dato, por lo que se
#%  ha incorporado a la función de carga de datos un filtro para eliminarlo, ajustable por parámetro (limpiarNA)
#%
#%  Vamos a observar la distribución de nuestra clase:
sns.distplot(extrinsic_classes)


#%% md

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

#%%



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
