"""Atividade 6 - Classificacao utilizando RNA (MLP) e SVM.

IFRN - Tecnicas de Analise de Dados
Dupla: Roberio Batista de Azevedo Junior e Daniel Goes
Base: Medical Cost Personal Datasets (dados/insurance.csv)

As classes Baixo, Medio e Alto sao criadas a partir de charges,
assim como nas Atividades 4 e 5. Nao existem originalmente no CSV.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


RAIZ = Path(__file__).resolve().parents[1]
ARQUIVO = RAIZ / "dados" / "insurance.csv"
RESULTADOS = RAIZ / "resultados"
CLASSES = ["Baixo", "Medio", "Alto"]


def carregar_dados():
    """Aplica a mesma limpeza, divisao e discretizacao das atividades anteriores."""
    df = pd.read_csv(ARQUIVO)
    print("Dimensoes originais:", df.shape)
    print("Valores ausentes:", int(df.isna().sum().sum()))
    print("Duplicatas:", int(df.duplicated().sum()))
    df = df.drop_duplicates().copy()
    if df.isna().any().any():
        raise ValueError("Existem valores ausentes nao tratados na base.")

    X = df.drop(columns=["charges"])
    custos = df["charges"]
    X_treino, X_teste, custos_treino, custos_teste = train_test_split(
        X, custos, test_size=0.20, random_state=42, stratify=X["smoker"]
    )

    # Os tercis sao definidos somente pelos custos do treinamento.
    faixas_treino, limites = pd.qcut(
        custos_treino, q=3, labels=CLASSES, retbins=True
    )
    limites[0], limites[-1] = -np.inf, np.inf
    faixas_teste = pd.cut(
        custos_teste, bins=limites, labels=CLASSES, include_lowest=True
    )
    # Codigos 0, 1, 2 evitam problemas de comparacao de strings no MLP.
    y_treino = pd.Series(faixas_treino.cat.codes, index=X_treino.index)
    y_teste = pd.Series(faixas_teste.cat.codes, index=X_teste.index)
    if (y_treino < 0).any() or (y_teste < 0).any():
        raise ValueError("Foram encontrados custos fora das faixas definidas.")
    print("Registros apos limpeza:", len(df))
    print("Treino e teste:", len(X_treino), len(X_teste))
    print("Classes no treino:\n", y_treino.map(dict(enumerate(CLASSES))).value_counts())
    print("Limites internos (US$):", np.round(limites[1:-1], 2))
    return X_treino, X_teste, y_treino, y_teste


def criar_pipeline(classificador):
    """Evita vazamento ao ajustar o pre-processamento dentro de cada fold."""
    pre = ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler(), ["age", "bmi", "children"]),
            ("categoricas", OneHotEncoder(handle_unknown="ignore"),
             ["sex", "smoker", "region"]),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline([
        ("preprocessamento", pre), ("classificador", classificador)
    ])


def avaliar(nome, busca, X_treino, X_teste, y_treino, y_teste, cv_externo):
    """Avalia a busca inteira com cross_val_score e testa o melhor modelo."""
    print("\n===", nome, "===")

    # Validação cruzada ANINHADA: GridSearchCV e executado dentro
    # de cada uma das 5 divisoes externas do cross_val_score.
    acuracias = cross_val_score(
        busca, X_treino, y_treino, cv=cv_externo, scoring="accuracy"
    )
    print("Acuracias nas 5 dobras:", np.round(acuracias, 4))
    print("Media +/- desvio (CV):", round(acuracias.mean(), 4),
          "+/-", round(acuracias.std(), 4))

    # Reajusta a busca no treino completo e avalia o teste reservado.
    busca.fit(X_treino, y_treino)
    previsoes = busca.predict(X_teste)
    acuracia = accuracy_score(y_teste, previsoes)
    f1 = f1_score(y_teste, previsoes, average="macro")
    matriz = confusion_matrix(y_teste, previsoes, labels=[0, 1, 2])

    print("Melhores hiperparametros (GridSearchCV):", busca.best_params_)
    print("F1-macro medio na busca interna:", round(busca.best_score_, 4))
    print("Acuracia no teste:", round(acuracia, 4))
    print("F1-macro no teste:", round(f1, 4))
    print("Relatorio:\n", classification_report(
        y_teste, previsoes, labels=[0, 1, 2],
        target_names=CLASSES, zero_division=0
    ))
    print("Matriz de confusao (Baixo, Medio, Alto):\n", matriz)

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matriz, index=CLASSES, columns=CLASSES).to_csv(
        RESULTADOS / ("atividade6_matriz_" + nome.lower() + ".csv")
    )
    return {
        "modelo": nome, "cv_media": acuracias.mean(),
        "cv_desvio": acuracias.std(), "f1_cv_interna": busca.best_score_,
        "acuracia_teste": acuracia, "f1_macro_teste": f1,
        "melhores_parametros": str(busca.best_params_),
    }


def main():
    X_treino, X_teste, y_treino, y_teste = carregar_dados()
    cv_externo = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_interno = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # 1. Rede neural MLP, inspirada no exemplo dos slides do professor.
    mlp = criar_pipeline(MLPClassifier(
        activation="relu", solver="adam", max_iter=400,
        early_stopping=True, validation_fraction=0.20,
        n_iter_no_change=20, random_state=42,
    ))
    parametros_mlp = {
        "classificador__hidden_layer_sizes": [(16, 8), (32, 16)],
        "classificador__alpha": [0.0001, 0.01],
        "classificador__learning_rate_init": [0.001, 0.01],
    }
    busca_mlp = GridSearchCV(
        mlp, param_grid=parametros_mlp, cv=cv_interno,
        scoring="f1_macro", n_jobs=-1, refit=True,
    )

    # 2. SVM: testar os kernels linear e RBF, C e gamma (apenas RBF).
    svm = criar_pipeline(SVC())
    parametros_svm = [
        {"classificador__kernel": ["linear"],
         "classificador__C": [0.1, 1, 10]},
        {"classificador__kernel": ["rbf"],
         "classificador__C": [0.1, 1, 10],
         "classificador__gamma": ["scale", 0.1]},
    ]
    busca_svm = GridSearchCV(
        svm, param_grid=parametros_svm, cv=cv_interno,
        scoring="f1_macro", n_jobs=-1, refit=True,
    )

    resultados = []
    for nome, busca in [("MLP", busca_mlp), ("SVM", busca_svm)]:
        resultados.append(avaliar(
            nome, busca, X_treino, X_teste, y_treino, y_teste, cv_externo
        ))

    pd.DataFrame(resultados).round(4).to_csv(
        RESULTADOS / "atividade6_comparativo.csv", index=False
    )
    print("\nResultados salvos na pasta resultados/.")


if __name__ == "__main__":
    main()
