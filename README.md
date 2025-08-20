README: Modelos Supervisados y No Supervisados en Python
En el mundo del aprendizaje automático, los algoritmos se dividen en dos categorías principales: modelos supervisados y modelos no supervisados. Cada uno tiene un propósito diferente y se utiliza según el tipo de datos y la tarea a realizar. Este documento explica las diferencias, sus aplicaciones comunes y cómo se implementan en Python.

1. Modelos de Aprendizaje Supervisado
En el aprendizaje supervisado, el algoritmo aprende a partir de un conjunto de datos que ya está etiquetado. Esto significa que cada punto de datos de entrenamiento tiene una etiqueta de resultado o variable objetivo conocida. El objetivo del modelo es aprender la relación entre las características de entrada y las etiquetas de salida, para luego poder predecir la etiqueta de nuevos datos no vistos.

Casos de Uso Comunes
Clasificación: Se utiliza para predecir una etiqueta discreta o categórica.

Ejemplo: Clasificar un correo electrónico como "spam" o "no spam". 📧

Regresión: Se utiliza para predecir una variable continua.

Ejemplo: Predecir el precio de una casa basándose en sus características. 🏡

Librerías de Python para Modelos Supervisados
La librería más popular para este tipo de modelos en Python es Scikit-learn. Algunos de los algoritmos más utilizados incluyen:

Clasificación: sklearn.linear_model.LogisticRegression, sklearn.svm.SVC, sklearn.ensemble.RandomForestClassifier

Regresión: sklearn.linear_model.LinearRegression, sklearn.ensemble.RandomForestRegressor

2. Modelos de Aprendizaje No Supervisado
A diferencia del aprendizaje supervisado, los modelos no supervisados trabajan con datos no etiquetados. El algoritmo no tiene una variable objetivo y su misión es encontrar patrones, estructuras o relaciones ocultas dentro de los datos por sí mismo.

Casos de Uso Comunes
Clustering (Agrupamiento): Agrupar puntos de datos similares en clústeres o grupos.

Ejemplo: Segmentar a los clientes de una empresa según sus hábitos de compra. 🛍️

Asociación: Encontrar relaciones o reglas de asociación entre variables en grandes conjuntos de datos.

Ejemplo: Recomendar productos a un cliente basándose en su historial de compras (ej. "los que compraron X también compraron Y"). 🛒

Librerías de Python para Modelos No Supervisados
Scikit-learn también ofrece una amplia gama de algoritmos de aprendizaje no supervisado. Algunos de ellos son:

Clustering: sklearn.cluster.KMeans, sklearn.cluster.DBSCAN

Reducción de dimensionalidad: sklearn.decomposition.PCA (Análisis de Componentes Principales)
