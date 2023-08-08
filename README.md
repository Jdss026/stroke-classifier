# stroke-classifier

- Análise exploratória, Limpeza da base, Modelagem e API de consumo de uma base de dados desbalanceado de caso de AVC 

## Definição da Base

- Base de dados sobre detalhes medicos de pacientes em que alguns sofreram de AVC
- Base de dados desbalanceada
- Fonte:  [Data for: A hybrid machine learning approach to cerebral stroke prediction based on imbalanced medical-datasets](https://data.mendeley.com/datasets/x8ygrw87jw/1)

## Análise exploratória da base de dados
- Visualização da base, levantamento de perguntas, tratamento da base, tratamento de outliers
- Limpeaza da base de dados

## ML Notebook
- Preprocessamento da base
- Tratamento do desbalanceamento da base: RandomUndersampling
- Modelagem via KNeighborsClassifier

## APP 
- API em Flask para consumo do modelo

# Estrutura de Pastas

- **data/**: todos os dados usados no projeto
  - **data_clean.csv**: dados limpos (pós EDA)
  - **dataset.csv**: dados brutos
- **env/**: environment python
- **models/**: modelos pickle salvos
- **notebooks/**: Notebooks jupyter contendo EDA e ML
- **templates/**: template para página HTML da API
  - **index.html**
- **app.py**: código da API
- **README.md**: este arquivo
- **requirements.txt**: lista de dependencias para este projeto

# Uso

1. Instale as dependencias e ative o venv

```bash
source env/bin/activate
pip install -r requirements.txt
```
2. Execute os notebooks EDA e ML para gerar os dados a serem utilizados pela API

3. Execute a API

```bash
python app.py
```
