# Machine-Learning Proyect

## Objetivo 

Estamos ante un proyecto que consiste en la creación de un modelo predictivo para tratar de predecir precios de ordenadores a partir de una base de datos de una competición de [Kaggle]([https://www.kaggle.com/competitions/predict-the-price-for-laptops/overview])

**Spoiler**: Quedé en cuarta posición de 22 aplicantes. 

Se presentan los siguientes archivos: 

 - **train.csv**: Aquí podemos encontrar una muestra de datos con alrededor de 1000 modelos de ordenadores con su precio y respectivas caracteristicas como almacenamiento, RAM, CPU, CGU... Este archivo nos servirá para entrenar los distintos modelos. 
 - **test.csv**: Muestra de datos parecida a la de train, sin embargo esta no trae la variable dependiente, es decir los precios a predecir.
 - **muestra.csv**: eschema para subir predicciones a la competición de Kaggle. 

## Analisis inicial

Aquí buscamos familiarizarnos con los datos. Esto es un paso imprescindible para ML, ya que va a ser necesario optimizar los datos para el modelo.


Algunas de las incognitas que buscamos despejar a través de este analisis son:


 - Datos nulos y duplicados
    - Habrá que ver si eliminamos las filas con muchos datos nulos o los rellenamos con media, moda o mediana
 - Tipo de dato por columna 
    - Cuales son numericos o no
    - Cuales son categoricos. Estos los tendremos que transformar a formato numerico. 
      - Para los categoricos tambien va a haber que evaluar cuantos valores unicos hay por columna. 
      - En el caso de que sean muchos habrá que evaluar como agruparlos. De lo contario podria confundir al modelo. 
 - Correlacion/colinealidad entre columnas
 - Distribución de los datos: outliers y sesgo(skew)


## Limpieza y adecuación

Tras realizar este primero analisis exploratorio de los datos, pasamos a la limpieza.

<details>
<summary>Algunas de las adecuaciones realizadas incluyen::</summary>
<br> 

- Quitar valores no numericos de las columnas **'RAM'**, **'Weight'** y **'Screen Size'** como 'GB', 'kg' y '"' (pulgadas). Posteriormente pasar a Dtype numerico.
- Borrar columna **'Model Name'** por contener 2/3 de valores unicos. No nos aporta información relevante. 
- Borrar columna **'Operating System Version'** contiene muchos nulos. Además de que las updates de sistemas operativos no tienen costo para el usuario. No deberian para el proveedor tampoco. 
- Columna **'Screen'**: 
    - Aplicar un bucle condicional para leer strings dentro de la columna, alimentando una nueva columna **'Touchscreen'** con 0 y 1 en función de si tiene pantalla tactil.
    - Usar Regex para quitar valores no numericos.
    - Aplicar bucle condicional para tomar la resolución y categorizar (HD, QHD, UHD).
    - Pasar categorias a valores numericos (0 a 2)
- Columna **'CPU'**: 
    - Bucle para extraer valores dentro de la columna para alimentar columna **'Processor brand'**
    - Sobre la columna usamos bucle para dejar el modelo de CPU.  
- Columna **Storage**:
    - Usamos una funcion regex para extraer unicamente los valores numericos que pasamos a una nueva columna **StorageGB** En los casos donde haya almacenamiento hibrido hacemos la sumatoria. Con el objetivo de obtener columna de gigas de almacenamiento.  
    - Creamos nueva columna **StorageType** que saque la información de tipo de almacenamiento (SDD, Hibrido, HDD). Posteriormente numerizamos en función de la escala de precio (2,1 y 0).
    - Descratamos la columna **Storagre**
- Columna **GPU**:
    - Separamos por marca. 

Realizadas estas adecuaciones, pasamos todas las columnas a tipo float/int para valores numericos y para datos categoricos en string aplicamos un get dummies. 

## Entrenamiento de modelos

Ya con los datos listos, utilizamos lazy regressor y H2o para que cada uno haga test de entrenamiento sobre distintos modelos y determine cuales son los mas eficaces. 

Tomamos los dos mejores de Lazy y el mejor de H2o que entrenareamos nuevamente y utilizaremos para hacer nuestras predicciones. 
