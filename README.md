# Brief-PySpark

## Les commandes utiles du projet

### Init

```
uv init --python 3.12
uv add pandas
uv add jupyter
uv add pyspark
uv add --dev pytest pandas pyspark ruff black isort pre-commit
```

### Lancement des scripts

Attention à l'emplacement du fichier lors de la commande dans le terminal.

```
uv run python pipeline_pyspark.py
```

### Pre-Commit

Pour activer et lancer manuellement:
```
uv run pre-commit install
uv run pre-commit run -a
```
