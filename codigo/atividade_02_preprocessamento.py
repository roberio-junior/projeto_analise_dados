"""Atividade 2 - Pré-processamento do Medical Cost Personal Datasets.
Dupla: Robério Batista de Azevêdo Júnior e Daniel Goes.
Código da disciplina ajustado para a organização do repositório.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
ARQUIVO_DADOS = RAIZ / "dados" / "insurance.csv"
PASTA_RESULTADOS = RAIZ / "resultados"


def preparar_dados():
    """Realiza a Atividade 2 e devolve dados preparados à Atividade 3."""
    df = pd.read_csv(ARQUIVO_DADOS)
    print("Dimensões originais:", df.shape)
    print("Tipos de dados:\n", df.dtypes)

    # Conferências que orientaram o pré-processamento na EDA.
    print("Valores ausentes por coluna:\n", df.isna().sum())
    print("Registros repetidos:", df.duplicated().sum())
    print("Distribuição do tabagismo:\n", df["smoker"].value_counts())
    print("Resumo dos custos:\n", df["charges"].describe())

    # A EDA identificou 1 duplicata e nenhum valor ausente.
    # Não aplicar dropna() ou fillna(), nem remover custos elevados.
    df = df.drop_duplicates().copy()
    if df.isna().any().any():
        raise ValueError("Existem valores ausentes não previstos na EDA.")
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    df.to_csv(PASTA_RESULTADOS / "insurance_limpo.csv", index=False)
    print("Dimensões após a limpeza:", df.shape)

    # O alvo charges fica separado das variáveis explicativas.
    variaveis_numericas = ["age", "bmi", "children"]
    variaveis_categoricas = ["sex", "smoker", "region"]
    X = df[variaveis_numericas + variaveis_categoricas]
    y = df["charges"]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=X["smoker"]
    )

    preprocessador = ColumnTransformer(transformers=[
        ("numericas", StandardScaler(), variaveis_numericas),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"),
         variaveis_categoricas),
    ])
    X_treino_preparado = preprocessador.fit_transform(X_treino)
    X_teste_preparado = preprocessador.transform(X_teste)
    print("Treino - dados preparados:", X_treino_preparado.shape)
    print("Teste - dados preparados:", X_teste_preparado.shape)
    print("Colunas preparadas:\n", preprocessador.get_feature_names_out())
    print("Alvo mantido em dólares; treino:", y_treino.shape)
    print("Alvo mantido em dólares; teste:", y_teste.shape)

    return X_treino_preparado, X_teste_preparado, y_treino, y_teste


if __name__ == "__main__":
    preparar_dados()
