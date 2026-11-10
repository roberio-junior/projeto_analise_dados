"""Atividade 4 - Classificacao com Naive Bayes e Arvore de Decisao.

IFRN - Tecnicas de Analise de Dados
Dupla: Roberio Batista de Azevedo Junior e Daniel Goes
Dados: Medical Cost Personal Datasets (dados/insurance.csv)

A coluna charges e continua. Para manter o foco das atividades anteriores,
criamos as classes Baixo, Medio e Alto usando os tercis de charges no TREINO.
Essas classes nao existem originalmente no arquivo insurance.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RAIZ = Path(__file__).resolve().parents[1]
ARQUIVO = RAIZ / "dados" / "insurance.csv"
RESULTADOS = RAIZ / "resultados"
CLASSES = ["Baixo", "Medio", "Alto"]


def carregar_e_separar():
    """Limpeza guiada pela EDA e classes definidas pelo treino."""
    df = pd.read_csv(ARQUIVO)
    print("Dimensoes originais:", df.shape)
    print("Valores ausentes:", int(df.isna().sum().sum()))
    print("Duplicatas:", int(df.duplicated().sum()))
    df = df.drop_duplicates().copy()
    if df.isna().any().any():
        raise ValueError("A base possui valores ausentes nao tratados.")

    # A coluna-alvo continua nao pode aparecer entre os preditores.
    X = df.drop(columns=["charges"])
    custos = df["charges"]
    X_treino, X_teste, custos_treino, custos_teste = train_test_split(
        X, custos, test_size=0.20, random_state=42, stratify=X["smoker"]
    )

    # Mesma proposta da Atividade 3: tres faixas de custo por quantis.
    # Os limites sao calculados SOMENTE no conjunto de treinamento.
    y_treino, limites = pd.qcut(
        custos_treino, q=3, labels=CLASSES, retbins=True
    )
    limites[0], limites[-1] = -np.inf, np.inf
    y_teste = pd.cut(
        custos_teste, bins=limites, labels=CLASSES, include_lowest=True
    )
    print("Registros apos limpeza:", len(df))
    print("Treino e teste:", len(X_treino), len(X_teste))
    print("Distribuicao dos rotulos de treino:\n", y_treino.value_counts())
    print("Limites internos (US$):", [round(x, 2) for x in limites[1:-1]])
    return X_treino, X_teste, y_treino, y_teste


def criar_pipeline(classificador):
    """Prepara as variaveis DENTRO de cada dobra da validacao cruzada."""
    pre = ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler(), ["age", "bmi", "children"]),
            ("categoricas", OneHotEncoder(handle_unknown="ignore"),
             ["sex", "smoker", "region"]),
        ],
        sparse_threshold=0.0,  # GaussianNB exige matriz densa.
    )
    return Pipeline([("preprocessamento", pre), ("classificador", classificador)])


def main():
    X_treino, X_teste, y_treino, y_teste = carregar_e_separar()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Naive Bayes Gaussiano + validacao cruzada OBRIGATORIA.
    naive_bayes = criar_pipeline(GaussianNB())
    scores_nb = cross_val_score(
        naive_bayes, X_treino, y_treino, cv=cv, scoring="accuracy"
    )
    naive_bayes.fit(X_treino, y_treino)

    # 2. Arvore de Decisao + GridSearchCV OBRIGATORIO.
    arvore = criar_pipeline(DecisionTreeClassifier(random_state=42))
    parametros = {
        "classificador__criterion": ["gini", "entropy"],
        "classificador__max_depth": [3, 5, 7, None],
        "classificador__min_samples_leaf": [1, 5],
    }
    busca = GridSearchCV(
        arvore, param_grid=parametros, cv=cv,
        scoring="f1_macro", n_jobs=-1, refit=True
    )

    # 3. cross_val_score sobre a PROPRIA busca: validacao cruzada aninhada.
    # A escolha de hiperparametros acontece apenas nas dobras internas.
    scores_arvore = cross_val_score(
        busca, X_treino, y_treino, cv=cv, scoring="accuracy"
    )
    busca.fit(X_treino, y_treino)
    melhor_arvore = busca.best_estimator_

    # 4. Comparar desempenho da validacao cruzada e do teste reservado.
    resultados = []
    for nome, modelo, scores in [
        ("Naive Bayes", naive_bayes, scores_nb),
        ("Arvore de Decisao", melhor_arvore, scores_arvore),
    ]:
        previsoes = modelo.predict(X_teste)
        acuracia = accuracy_score(y_teste, previsoes)
        f1 = f1_score(y_teste, previsoes, average="macro")
        print("\n===", nome, "===")
        print("Acuracias nas 5 dobras:", np.round(scores, 4))
        print("CV media +/- desvio:", round(scores.mean(), 4),
              "+/-", round(scores.std(), 4))
        print("Acuracia no teste:", round(acuracia, 4))
        print("F1-macro no teste:", round(f1, 4))
        print("Relatorio:\n", classification_report(
            y_teste, previsoes, labels=CLASSES, zero_division=0
        ))
        matriz = confusion_matrix(y_teste, previsoes, labels=CLASSES)
        print("Matriz de confusao (Baixo, Medio, Alto):\n", matriz)
        resultados.append({
            "algoritmo": nome, "cv_media": scores.mean(),
            "cv_desvio": scores.std(), "acuracia_teste": acuracia,
            "f1_macro_teste": f1,
        })
    print("\nMelhores hiperparametros da Arvore:", busca.best_params_)

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(resultados).round(4).to_csv(
        RESULTADOS / "atividade4_comparativo.csv", index=False
    )
    print("\nResumo salvo em resultados/atividade4_comparativo.csv")


if __name__ == "__main__":
    main()
