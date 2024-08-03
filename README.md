# multilayer-perceptron
Ce projet est une introduction aux réseaux neuronaux artificiels, avec l'implémentation d'un perceptron multi-couches.

## instalation

```bash
poetry install
```

## separation des data

```bash
poetry run python multilayer_perceptron/data_split.py
```

## entrainement du model

```bash
poetry run python multilayer_perceptron/train.py
```

## prediction du model

```bash
poetry run python multilayer_perceptron/test.py
```
ou 
```bash
poetry run python multilayer_perceptron/test.py nom/model nom/des/data
```