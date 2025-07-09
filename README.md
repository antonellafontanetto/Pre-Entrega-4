# 🛢️ Proyecto 4x4 YPF - Análisis de Vaca Muerta
# Pre-Entrega 4 - Data Science
# Grupo 3 - Fundación YPF / Programa Ingenias+

# Introducción

El proyecto 4x4 de YPF tiene como objetivo principal cuadruplicar la producción de la compañía en los próximos 4 años, enfocándose en la eficiencia operativa, la generación de valor y la concentración de esfuerzos en los activos más rentables a corto plazo, como Vaca Muerta, la principal cuenca petrolera del país.

Este trabajo busca analizar el crecimiento de la producción de petróleo y gas en la cuenca neuquina desde el año 2021 en adelante, considerando exclusivamente los datos correspondientes a YPF S.A.. El análisis se basa en indicadores clave como la producción no convencional, las principales 5 áreas de concesiones y el subtipo de recurso producido, ya sea shale o tight en los últimos años.

En el análisis, se busca aportar evidencia empírica sobre la evolución y proyección del potencial de expansión de YPF en Vaca Muerta, en línea con los objetivos planteados en su plan estratégico 4x4.

# Objetivo

El objetivo del proyecto es analizar y predecir el crecimiento de la producción de petróleo y gas por parte de YPF S.A. en la cuenca neuquina, evaluando el impacto de diferentes variables relacionadas con la actividad extractiva y operativa.

# Integrantes - Grupo 3

Este proyecto fue desarrollado por el Grupo 3 del curso de Data Science de Fundación YPF en el marco del programa Ingenias+, que busca incorporar más mujeres al mundo de la programación.

* Cyntia Nasabun – Diplomada en Análisis de Datos e IA, IFES, Neuquén. 

* Antonella Fontanetto – Licenciada en Economía, Universidad de Buenos Aires

# Datasets utilizados

Los datos provienen de fuentes públicas oficiales del Gobierno Nacional y de la Provincia del Neuquén:

* Producción de Pozos de Petróleo y Gas http://datos.energia.gob.ar/dataset/produccion-de-petroleo-y-gas-por-pozo

Mapas y documentos complementarios:

* Mapa Energía Río Negro https://mapa.rionegro.com.ar/energia

* Distribución de Fluidos – Energía Neuquén https://www.energianeuquen.gob.ar/distribucion-de-fluidos/

* Línea estratégica YPF 4x4 https://www.rystadenergy.com/news/vaca-muerta-smashes-crude-output-record-in-3q

* Presentación IR Day 2025 – YPF https://inversores.ypf.com/documents/presentaciones/YPF-IR-DAY-2025-presentation.pdf

* YPF – Mayo 2024 – IAméricas https://iamericas.org/2024/05/YPF_Horacio-Marin_Mesa-Redonda_Mayo-8-presentation.pdf

* Análisis potencial exportador de Vaca Muerta – La Nación  https://www.lanacion.com.ar/el-potencial-exportador-vaca-muerta-el-planparaalcanzarlos30milmillonesdedolares

# Estructura del repositorio

📁

├── data/           # Datasets procesados

├── notebooks/      # Notebooks (diferentes versiones estudiadas)

├── README.md       # Este archivo

# Metodología

La metodología implementada en el proyecto está relacionada con la búsqueda minuciosa de información relevante para el desarrollo del mismo, en función de las respuestas a nuestro modelo predictivo. Teniendo como eje principal la proyección de la producción de petróleo y gas en Vaca Muerta por parte de YPF S.A. para los próximos años.

En el análisis se implementaron varios modelos de aprendizaje no supervisado, es decir, modelos que aprenden de datos no etiquetados del pasado para luego hacer predicciones, de esta manera estos modelos nos permiten dar soporte a decisiones estratégicas para la compañía.

Los modelos utilizados fueron:

* KMeans: Algoritmo de clustering que agrupa los datos en k grupos, minimizando la distancia entre los puntos y el centroide del cluster. Requiere definir el número de clusters previamente.

* MeanShift: Algoritmo de clustering que no necesita definir k previamente. Agrupa puntos buscando áreas de mayor densidad, ajustando dinámicamente el número de clusters.

* TimeSeries KMeans: Versión de KMeans diseñada para datos temporales. Agrupa series de tiempo según su forma o patrón de comportamiento a lo largo del tiempo, usando métricas como DTW o Euclídea.

* DBSCAN: Algoritmo basado en densidad. Agrupa puntos cercanos y densamente conectados, y clasifica como ruido a los puntos que no pertenecen a ningún grupo. No requiere especificar el número de clusters.

* PCA (Análisis de Componentes Principales): Técnica de reducción de dimensionalidad. Transforma los datos originales en nuevas variables (componentes) que conservan la mayor parte de la varianza con menos dimensiones, útil para visualizar o preprocesar datos.

* Kohonen SOM: Es un modelo de red neuronal no supervisada diseñado para reducir la dimensionalidad de datos complejos, descubrir patrones y visualizarlos en 2D, conservando la estructura topológica (es decir, manteniendo la cercanía entre puntos similares del espacio original).

* Mezcla Gaussiana: No necesitas datos con forma circular para que funcione bien.El modelo de mezcla Gaussiana utiliza múltiples distribuciones Gaussianas para ajustar datos que tienen formas arbitrarias.

Autoencoder para reducción a 2D (Redes Neuronales): Un autoencoder para reducción a 2D es un tipo de red neuronal que aprende a representar tus datos originales —que pueden tener muchas variables— en solo 2 dimensiones, de manera que conserve la mayor cantidad de información posible sobre la estructura y relaciones entre los datos.

Además por cada modelo se implementó la utilización y cálculo de métricas para corroborar que tan bien el modelo podía predecir nuestro objetivo. Las métricas utilizadas fueron:

* Silhouette Score: Mide qué tan bien están definidos los clusters: combina la cohesión interna (qué tan cerca están los puntos de su propio grupo) y la separación externa (qué tan lejos están de otros clusters).
Rango: de -1 a 1 (valores cercanos a 1 son mejores).

* Davies-Bouldin Index: Evalúa la similitud entre clusters, comparando la distancia entre ellos y su dispersión interna.
Rango: positivo, y valores más bajos indican mejor separación entre clusters.

* Calinski-Harabasz Index (o Variance Ratio Criterion): Mide la proporción entre la dispersión entre clusters y dentro de los clusters. Valores más altos indican una mejor definición de los grupos.

* Inercia (Within-cluster sum of squares): Utilizada en KMeans, mide la suma de las distancias cuadradas entre los puntos y el centroide del cluster. A menor inercia, mejor ajuste (aunque puede sobreajustar si hay demasiados clusters).

* Error Cuadrático Medio (Quantization Error): Mide la distancia promedio entre cada dato y su neurona ganadora (la más parecida).

* Topographic Error: Evalúa cuántos datos fueron mal asignados en términos de topología (es decir, si cayeron lejos de neuronas similares).

# Breve análisis de cada versión estudiada

* Versión con datos agrupados diferentes modelos: Se aplicó el algoritmo K-Means para agrupar series temporales según su comportamiento de producción. Este modelo no supervisado particiona los datos en clústeres minimizando la variación interna dentro de cada grupo. El resultado fue altamente satisfactorio, con un Silhouette Score de 0.900, lo que indica una excelente cohesión dentro de los clústeres y clara separación entre ellos. Esto valida que las series fueron correctamente segmentadas y que existen patrones bien diferenciados en los datos, lo que puede ser útil para identificar estrategias de operación, eficiencia o cambios en el rendimiento a lo largo del tiempo. Además, se utilizó el modelo Time Series K-Means con la métrica DTW (Dynamic Time Warping) para agrupar series temporales de producción. Este enfoque permite comparar series con diferentes ritmos o desplazamientos temporales, capturando mejor las similitudes en su forma. El modelo alcanzó un Silhouette Score de 0.900, lo que indica una excelente calidad de agrupamiento, con alta cohesión interna y buena separación entre clústeres. Esto valida la efectividad de usar DTW para identificar patrones temporales comunes en la producción, y ofrece una base sólida para análisis operativos y decisiones estratégicas. Por otro lado, el modelo DBSCAN (Density-Based Spatial Clustering of Applications with Noise) agrupa datos en función de su densidad, sin necesidad de definir la cantidad de clústeres previamente. Es especialmente útil para detectar formas arbitrarias y outliers. En este caso, identificó 17 clústeres con un Silhouette Score de 0.831, superior al obtenido con K-Means. Y Finalmente, el modelo MeanShift es un algoritmo de clustering basado en estimación de densidad, que no requiere definir previamente la cantidad de clústeres. Agrupa los datos desplazándose hacia las regiones de mayor densidad. En este caso, con un bandwidth de 0.562, se obtuvieron 12 clústeres y un Silhouette Score de 0.866, lo que indica una muy buena calidad de agrupamiento, con alta cohesión interna y buena separación entre grupos. El resultado muestra que MeanShift logró identificar patrones naturales en los datos, capturando estructuras complejas sin necesidad de parametrización estricta.

* Versión con datos agrupados modelo Kohonen SOM: El modelo Kohonen SOM (Self-Organizing Map) permite reducir la dimensionalidad de los datos y visualizar relaciones complejas de forma intuitiva mediante un mapa bidimensional. En este caso, representa cómo se agrupan las operaciones de producción de petróleo y gas en función de su similitud, revelando patrones ocultos, anomalías y fronteras entre distintos comportamientos operativos. Es una herramienta poderosa para segmentar, analizar y optimizar estrategias en contextos industriales complejos como el sector energético. Además, facilita la toma de decisiones basada en datos, al traducir información multivariable en una representación visual clara. Se identificaron zonas de color compacto que representan grupos estables y homogéneos de datos, como el Cluster 1 (alta producción de petróleo) y el Cluster 2 (alta producción de gas). Las áreas pequeñas o aisladas, como los bloques turquesa o azul oscuro, indican comportamientos atípicos o segmentos especiales. Los bordes irregulares entre colores reflejan transiciones bruscas en los patrones de producción, útiles para análisis estratégicos. El Cluster 0 agrupa casos marginales o de baja producción. 

# Herramientas utilizadas 

Lenguaje: Python 3.10+

Librerías:

Cuarta entrega

Librerías: NumPy, Pandas, Matplotlib, Scikit-learn, SciPy, PyTorch, TensorFlow y Keras

Modelado: PCA, Autoencoder, GMM, KMeans, DBSCAN

Visualización: matplotlib, seaborn, plotly, contextily, scipy

Otros: jupyter, google colab, GitHub
