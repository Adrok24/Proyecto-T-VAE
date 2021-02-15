# Autoencoder Variacional Transformer-Based para Generación de frases

<p align="center">
  <img src="https://github.com/Adrok24/Proyecto-T-VAE/blob/branch_3/images/transformer.jpg?raw=true" alt="grafico_1" width="400" height="200"/>
</p>

## Autores:
##### Adrián Di Paolo
##### Patricio Guinle

## Motivación 


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

El tokenizador se puede encontrar en el siguiente archivo pkl: [tokenizer.pkl](https://github.com/Adrok24/Proyecto-T-VAE/blob/first_version/tokenizer.pkl).

## Modelo utilizado

Una breve explicación del proceso de armado del modelo puede ser consultado en la siguiente [presentacion](https://github.com/Adrok24/Proyecto-T-VAE/blob/first_version/Presentacion.pptx). El siguiente es un equema del modelo final construído [1].

<p align="center">
  <img src="https://github.com/Adrok24/Proyecto-T-VAE/blob/branch_3/images/model.png?raw=true" alt="grafico_3" width="500" height="400"/>
</p>


## Funciones de Distribución

<p align="center">
  <img src="https://github.com/Adrok24/Proyecto-T-VAE/blob/branch_3/images/normal_distribution.jpg?raw=true" alt="grafico_4" width="30%"/>
  <img src="https://github.com/Adrok24/Proyecto-T-VAE/blob/branch_3/images/von_mises_fisher.png?raw=true" alt="grafico_5" width="31%"/>
</p>


## Visualizaciones

<p align="center">
  <img src="https://github.com/Adrok24/Proyecto-T-VAE/blob/branch_3/images/latent_space_long.png?raw=true" alt="grafico_6" width="700" height="500"/>
</p>

<p align="center">
  <img src="https://github.com/Adrok24/Proyecto-T-VAE/blob/branch_3/images/latent_space_poetry.png?raw=true" alt="grafico_7" width="700" height="500"/>
</p>



 [presentación](https://github.com/Adrok24/Proyecto-T-VAE/blob/first_version/Presentacion.pptx).


<a id="1">[1]</a> A Transformer-Based Variational Autoencoder for Sentence Generation1 st Danyang Liu, 2 nd Gongshen Liu (2019)




