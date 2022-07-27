

# Name Entity Recognition (NER) para el quechua

Este repositorio proporciona el modelo de reconocimiento de entidades nombradas para la lengua quechua sureño. El modelo es una mejora del modelo NER presentado por [Llamacha](https://aclanthology.org/2022.deeplo-1.1/). 

# Instalación


Se debe crear un entorno virtual e instalar las librerías de requerimiento
```
python -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

# Forma de usar

```
usage: NER quechua sureño

required arguments:
  -t root_train, --train root_train
                        Dataset de entrenamiento en formato csv.
  -e root_eval, --evaluation root_eval
                        Dataset de evaluación en formato csv.
  -o root_output, --output root_output
                        Archivo csv procesado.

optional arguments:
  -h, --help            show help message and exit

  -u root_text, --usage root_text
                        Archivo quechua a procesar.
  -i, --interactive     Interactuar con el modelo NER desde consola
```

## Modo interactivo

En este modo se puede interactuar mediante linea de comandos para reconocer las entidades nombradas del texto.

```
python src/main.py -i
```

## Procesar un archivo csv


```
python ner_quechua/src/main.py -u ner_quechua/resources/data.csv -o ner_quechua/results/
```

## Entrenar nuevo modelo NER

```
python src/main.py -t ner_quechua/resources/train_qu_ner.csv -e ner_quechua/resources/test_que_ner.csv
```
