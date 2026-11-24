"""Atividade 5 - Classificacao com k-Vizinhos Mais Proximos (k-NN).

IFRN - Tecnicas de Analise de Dados
Dupla: Roberio Batista de Azevedo Junior e Daniel Goes
Dados: Medical Cost Personal Datasets (dados/insurance.csv)

Mantemos a classificacao em Baixo, Medio e Alto adotada na Atividade 4.
Essas faixas sao criadas a partir de charges, nao existem no CSV original.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAIZ = Path(__file__).resolve().parents[1]
ARQUIVO = RAIZ / "dados" / "insurance.csv"
RESULTADOS = RAIZ / "resultados"
CLASSES = ["Baixo", "Medio", "Alto"]


def carregar_e_separar():
    """Usa a mesma divisao e as mesmas faixas da Atividade 4."""
    df = pd.read_csv(ARQUIVO)
    print("Dimensoes originais:", df.shape)
    print("Valores ausentes:", int(df.isna().sum().sum()))
    print("Duplicatas:", int(df.duplicated().sum()))
    df = df.drop_duplicates().copy()
    if df.isna().any().any():
        raise ValueError("Ha valores ausentes nao tratados no conjunto de dados.")

    X = df.drop(columns=["charges"])
    custos = df["charges"]
    X_treino, X_teste, custos_treino, custos_teste = train_test_split(
        X, custos, test_size=0.20, random_state=42, stratify=X["smoker"]
    )

    # Limites aprendidos somente com o treino; nao utilizar o alvo
    # charges como entrada nem usar custos_teste para definir faixas.
    y_treino, limites = pd.qcut(
        custos_treino, q=3, labels=CLASSES, retbins=True
    )
    limites[0], limites[-1] = -np.inf, np.inf
    y_teste = pd.cut(
        custos_teste, bins=limites, labels=CLASSES, include_lowest=True
    )
    print("Registros apos limpeza:", len(df))
    print("Treino e teste:", len(X_treino), len(X_teste))
    print("Classes no treino:\n", y_treino.value_counts())
    print("Limites internos (US$):", np.round(limites[1:-1], 2))
    return X_treino, X_teste, y_treino, y_teste


def criar_pipeline():
    """Padronizacao e one-hot DENTRO de cada dobra de validacao."""
    pre = ColumnTransformer(transformers=[
        ("numericas", StandardScaler(), ["age", "bmi", "children"]),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"),
         ["sex", "smoker", "region"]),
    ])
    return Pipeline([
        ("preprocessamento", pre),
        ("knn", KNeighborsClassifier(weights="uniform", metric="minkowski", p=2)),
    ])


def main():
    X_treino, X_teste, y_treino, y_teste = carregar_e_separar()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Obrigatorio: selecionar o melhor k com GridSearchCV.
    # Valores impares (1 a 31), como no exemplo dos slides.
    valores_k = list(range(1, 32, 2))
    busca = GridSearchCV(
        estimator=criar_pipeline(),
        param_grid={"knn__n_neighbors": valores_k},
        scoring="f1_macro", cv=cv, n_jobs=-1, refit=True,
    )

    # 2. Obrigatorio: cross_val_score avalia todo o processo, incluindo
    # a busca de k em cada rodada (validacao cruzada aninhada).
    acuracias_cv = cross_val_score(
        busca, X_treino, y_treino, cv=cv, scoring="accuracy"
    )
    print("\nAcuracias nas 5 dobras:", np.round(acuracias_cv, 4))
    print("Media +/- desvio (CV):", round(acuracias_cv.mean(), 4),
          "+/-", round(acuracias_cv.std(), 4))

    # 3. Ajustar a busca no treino completo e avaliar no teste reservado.
    busca.fit(X_treino, y_treino)
    melhor_k = busca.best_params_["knn__n_neighbors"]
    previsoes = busca.predict(X_teste)
    acuracia = accuracy_score(y_teste, previsoes)
    f1 = f1_score(y_teste, previsoes, average="macro")
    matriz = confusion_matrix(y_teste, previsoes, labels=CLASSES)
    print("\nMelhor k (GridSearchCV):", melhor_k)
    print("F1-macro medio na busca interna:", round(busca.best_score_, 4))
    print("Acuracia no teste:", round(acuracia, 4))
    print("F1-macro no teste:", round(f1, 4))
    print("Relatorio:\n", classification_report(
        y_teste, previsoes, labels=CLASSES, zero_division=0
    ))
    print("Matriz de confusao (Baixo, Medio, Alto):\n", matriz)

    # 4. Guardar resultados e grafico do impacto de k.
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "algoritmo": "k-NN", "melhor_k": melhor_k,
        "cv_media": acuracias_cv.mean(), "cv_desvio": acuracias_cv.std(),
        "acuracia_teste": acuracia, "f1_macro_teste": f1,
    }]).round(4).to_csv(RESULTADOS / "atividade5_resumo.csv", index=False)

    tabela_k = pd.DataFrame({
        "k": valores_k,
        "f1_macro_cv": busca.cv_results_["mean_test_score"],
    })
    tabela_k.to_csv(RESULTADOS / "atividade5_valores_k.csv", index=False)
    pd.DataFrame(matriz, index=CLASSES, columns=CLASSES).to_csv(
        RESULTADOS / "atividade5_matriz_confusao.csv"
    )

    plt.figure(figsize=(8, 4))
    plt.plot(tabela_k["k"], tabela_k["f1_macro_cv"], marker="o")
    plt.axvline(melhor_k, color="gray", linestyle="--", label=f"Melhor k = {melhor_k}")
    plt.xlabel("Numero de vizinhos (k)")
    plt.ylabel("F1-macro medio (CV interna)")
    plt.title("GridSearchCV: avaliacao de diferentes valores de k")
    plt.xticks(valores_k)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTADOS / "atividade5_curva_k.png", dpi=160)
    plt.close()
    print("\nResultados salvos na pasta resultados/.")


if __name__ == "__main__":
    main()
