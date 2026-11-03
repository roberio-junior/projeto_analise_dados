"""Atividade 2 - Pre-processamento do conjunto Medical Cost Personal Datasets.
Dupla: Robério Batista de Azevêdo Júnior e Daniel Goes.
Arquivo necessário: insurance.csv, disponível no Kaggle.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Carregar o conjunto de dados utilizado na primeira entrega.
df = pd.read_csv("insurance.csv")
print("Dimensões originais:", df.shape)
print("Tipos de dados:\n", df.dtypes)

# 2. Conferir os problemas identificados na análise exploratória (EDA).
print("Valores ausentes por coluna:\n", df.isna().sum())
print("Registros repetidos:", df.duplicated().sum())
print("Distribuição do tabagismo:\n", df["smoker"].value_counts())
print("Resumo dos custos:\n", df["charges"].describe())

# 3. Remover somente linhas inteiramente repetidas.
# A EDA encontrou 1 linha repetida e nenhum valor ausente.
# Sem identificador de pessoa, a igualdade não comprova identidade;
# a remoção é uma opção documentada para tratar a repetição exata.
df = df.drop_duplicates().copy()

# Não aplicar dropna() nem fillna(): a base original não possui nulos.
# Não excluir custos altos: podem ser casos reais e são o alvo da análise.
# Preservar as categorias originais e o valor de charges em dólares.
df.to_csv("insurance_limpo.csv", index=False)
print("Dimensões após a limpeza:", df.shape)

# 4. Identificar entradas e alvo para uma possível regressão futura.
variaveis_numericas = ["age", "bmi", "children"]
variaveis_categoricas = ["sex", "smoker", "region"]
X = df[variaveis_numericas + variaveis_categoricas]
y = df["charges"]

# 5. Separar treino e teste ANTES de calcular parâmetros de escala.
# Estratificar por smoker preserva a proporção de fumantes em ambos.
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=X["smoker"]
)

# 6. Padronizar apenas os preditores numéricos; codificar as
# categorias sem impor uma ordem artificial (one-hot encoding).
# Estas transformações são voltadas ao uso futuro em modelos,
# especialmente os sensíveis a distâncias e escalas.
preprocessador = ColumnTransformer(
    transformers=[
        ("numericas", StandardScaler(), variaveis_numericas),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"),
         variaveis_categoricas),
    ]
)

# Ajustar apenas com os dados de treino para evitar vazamento de dados.
X_treino_preparado = preprocessador.fit_transform(X_treino)
X_teste_preparado = preprocessador.transform(X_teste)

print("Treino - dados preparados:", X_treino_preparado.shape)
print("Teste - dados preparados:", X_teste_preparado.shape)
print("Colunas preparadas:\n", preprocessador.get_feature_names_out())
print("Alvo mantido em dólares; treino:", y_treino.shape)
print("Alvo mantido em dólares; teste:", y_teste.shape)
