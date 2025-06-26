🏡 Predicción del Precio de Viviendas en Lima, Perú usando Redes Neuronales
📌 Descripción
En el presente proyecto se realiza la recolección, modelamiento y visualización interactiva de un sistema para predecir precios de viviendas en Lima Metropolitana. Este trabajo fue desarrollado como parte del curso Ciencia de Datos 1, y combina técnicas de scraping, preprocesamiento de datos y redes neuronales artificiales.

🔍 Objetivo
Construir modelos de redes neuronales que permitan predecir el precio de venta de inmuebles en Lima, utilizando como entrada características estructurales y geográficas del inmueble. Además, se compara el desempeño entre dos enfoques distintos:

Scikit-learn (MLPRegressor)

PyTorch (modelo personalizado)

🌐 Fuentes de los Datos
Se realizó web scraping desde las siguientes plataformas inmobiliarias:

Infocasas – Casas en venta en Lima

Infocasas – Departamentos en venta en Lima

Se obtuvo un dataset consolidado de 434 registros, con variables como:

Dormitorios, Baños, Garajes

M² construidos

Tipo de Propiedad

Estado del Inmueble

Zona Macro

Precio de Venta

🧪 Técnicas Aplicadas
Limpieza de datos: winsorización de outliers, agrupación de categorías raras

Transformación del target: raíz cuadrada del precio + escalado robusto

Modelos:

Scikit-learn: Pipeline con MLPRegressor + tuning con RandomizedSearchCV

PyTorch: red neuronal multicapa definida manualmente con tuning interno

Evaluación: métricas R², MAE, RMSE

Despliegue: aplicación interactiva con Streamlit que permite seleccionar el modelo, ingresar datos y obtener la predicción del precio
