#%% md

#%_ Bloque de introducción

#%% md

# Datasets
## Preparación
### Librerías

#%%

import pandas as pd
import seaborn as sns


#%% md

### Función de descarga
#%_ blablabla

#%%

def load_dataset(dataset_url: str, separator: str = '\s+', class_position: int = None, remove: tuple = None):
    """Load a dataset from a specified url into a pandas DataFrame.

    :param str dataset_url: an url from archive.ics.uci.edu
    :param int class_position: column index where classes are defined (starts by 0)
       if left empty (None), no prediction class will be used (intrinsic case).
    """
    # Load dataset as a pandas DataFrame from a specified url.
    dataset = pd.read_csv(dataset_url, sep=separator, header=None)

    # Remove attributes.
    if remove is not None:
        dataset.drop(remove, axis=1, inplace=True)

    # Extrinsic case, dataset comes with its classes.
    if class_position is not None:
        # Extract classes.
        classes = dataset.iloc[:, class_position]
        # TODO: aún no sé si dejar o eliminar las clases del dataset,
        # mas que nada para el hue de seaborn a la hora de presentar los datos.
        # Remove classes from the dataset.
        # dataset = dataset.drop([class_position,], axis=1)

    # Intrinsic case, dataset has no classes.
    else:
        classes = None

    return classes, dataset


#%% md

### Función de visualización
#%_ blablabla
#%_ la idea es que esta función sea parametrizable y que pueda colorear los clusters

#%%

def plot_dataset(dataset: pd.DataFrame, class_position: int = None) -> None:
    # TODO: remove or use the class_position argument.
    sns.pairplot(dataset, hue=class_position)
    # sns.pairplot(dataset)


#%% md

## Selección

#%% md

### Dataset extrínseca
#%_ lo cilindros están en posición 1 (partiendo de 0)
#%_ no los he eliminado para que veamos el hue

#%%

dataset_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
extrinsic_classes, extrinsic_dataset = load_dataset(dataset_url, class_position=1, remove=[6, 7])

#%%

plot_dataset(extrinsic_dataset, class_position=1)

#%% md

#%_ blablabla

#%% md

### Dataset intrínseca
#%_ blablabla

#%%

# load_dataset()

#%%

# plot_dataset()

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
