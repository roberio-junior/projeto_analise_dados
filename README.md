# Projeto de Análise de Dados

Projeto acadêmico desenvolvido na disciplina **Técnicas de Análise de Dados**, do IFRN – Campus Currais Novos.

**Equipe:** Robério Batista de Azevêdo Júnior e Daniel Goes.

## Base de dados

[Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance), disponível no Kaggle. O arquivo insurance.csv contém 1.338 registros e sete colunas. A análise considera características como idade, IMC, tabagismo e custos médicos (charges). A documentação da base não informa claramente o período correspondente aos custos.

## Organização

    projeto_analise_dados/
    ├── codigo/
    │   ├── atividade_02_preprocessamento.py
    │   └── atividade_03_agrupamento.py
    ├── dados/
    │   └── insurance.csv
    ├── documentos/
    │   ├── 01-proposta.pdf
    │   ├── 02-preprocessamento.pdf
    │   └── 03-agrupamento.pdf
    ├── resultados/       # gerada automaticamente; ignorada pelo Git
    ├── .gitignore
    ├── requirements.txt
    ├── README.md
    └── LICENSE

Os PDFs das entregas permanecem na pasta documentos/. Os códigos Python estão em codigo/ e reutilizam o mesmo insurance.csv. O pré-processamento da Atividade 2 é executado pela Atividade 3, evitando duplicar sua implementação.

## Como executar (Debian/Ubuntu, sem ambiente virtual)

Instale as dependências caso ainda não estejam disponíveis:

    sudo apt update
    sudo apt install python3-pandas python3-sklearn

Na raiz do projeto, execute:

    python3 codigo/atividade_02_preprocessamento.py
    python3 codigo/atividade_03_agrupamento.py

A Atividade 2 gera resultados/insurance_limpo.csv. A Atividade 3 gera resultados/atividade3_metricas.csv, resultados/atividade3_matriz_kmeans.csv e resultados/atividade3_matriz_hierarquico.csv.

## Atividades

- **Atividade 1 – Proposta e EDA:** descrição da equipe, problema investigado e caracterização da base.
- **Atividade 2 – Pré-processamento:** inspeção de valores ausentes e duplicatas, limpeza, codificação de categorias e padronização.
- **Atividade 3 – Agrupamento:** aplicação de K-Means e agrupamento hierárquico aglomerativo; avaliação com ARI, NMI e Silhouette.

Na Atividade 3, as faixas **Baixo**, **Médio** e **Alto** são criadas a partir de charges usando quantis. Elas **não são rótulos originalmente existentes na base**; são usadas somente para avaliar os grupos. A coluna charges não participa da formação dos clusters. As métricas divulgadas nos PDFs precisam ser conferidas executando o código com o arquivo original.

**Nota de organização:** os caminhos de leitura/gravação nos scripts deste repositório foram ajustados à estrutura de pastas; os procedimentos de análise das entregas foram preservados.
