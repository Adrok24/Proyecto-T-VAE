# Autoencoder Variacional Transformer-Based para Generación de frases

## Autores:
#### Adrián Di Paolo
#### Patricio Guinle

## Motivación 


<p align="center">
  <img src="https://github.com/Adrok24/Proyecto-T-VAE/blob/branch_3/images/transformer.jpg?raw=true" alt="grafico_1" width="400" height="200"/>
</p>

Nuestra motivación es crear un modelo de transferencia de estilo para texto, utilizando herramientas de NLP (transformers) y aprovechando la capacidad los Autoencoders Variacionales de generar un espacio contínuo. Para ello reprodujimos la arquitectura del paper https://ieeexplore.ieee.org/document/8852155 [1] como punto de partida.


## Dataset

* Seleccionamos 120 libros de diferentes generos: técnicos, narrativos, y poéticos

* Los separamos en líneas de entre 35 y 160 caracteres

* Las filtramos y pre procesamos, definiendo algunas reglas con REGEX

* Enmascaramos Nombres propios y números (@name @number)

* Aplicamos un análisis de sentimiento Positivo, Neutral y Negativo a cada linea

* Tokenizamos cada línea por sub words

* Finalmente aplicamos padding hasta alcanzar nuestra longitud máxima (32 tokens x línea)

* En ésta oportunidad la longitud la estamos contando en nº de Tokens y no en caracteres

Sampleo del Dataset resultante:


<p align="center">
  <img src="https://github.com/Adrok24/Proyecto-T-VAE/blob/branch_3/images/dataset.png?raw=true" alt="grafico_2" width="800" height="200"/>
</p>

## Modelo utilizado

<p align="center">
  <img src="https://github.com/Adrok24/Proyecto-T-VAE/blob/branch_3/images/model.png?raw=true" alt="grafico_3" width="500" height="500"/>
</p>

Un análisis estadístico del dataset puede ser consultado en la siguiente notebook: [Notebook](https://github.com/Adrok24/classification-of-plant-diseases/blob/first_version/Estadistica.ipynb)

## Modelos entrenados para la clasificación del dataset

 [presentación](https://github.com/Adrok24/classification-of-plant-diseases/blob/first_version/presentacion/Presentacion.pptx).


### Modelos simple output
* CNN Custom [Notebook]()



* Otras pruebas pueden encontrarse en la [carpeta]() 



## Visualizaciones




<a id="1">[1]</a> A Transformer-Based Variational Autoencoder for Sentence Generation1 st Danyang Liu, 2 nd Gongshen Liu (2019)




