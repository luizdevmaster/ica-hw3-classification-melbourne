# Treina modelos de classificação para HighEnergy (alto consumo diário)
from mlxtend.evaluate import mcnemar_table, mcnemar
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns  # Adicionado para um visual melhor

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score

# --- CONFIGURAÇÃO DE CAMINHOS ---
BASE_DIR = Path(r"C:\Users\augus\Documents\ICA - HOMEWORK 1\Homework 3")
# Cria uma pasta 'figures' dentro do diretório base para o gráfico
FIGURES_DIR = BASE_DIR / "figures"
OUT_DIR = BASE_DIR / "outputs_hw3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)  # Garante que a pasta de figuras exista


def load_data():
    """Carrega os dados de treino e teste."""
    train_df = pd.read_csv(OUT_DIR / "train_classification.csv")
    test_df = pd.read_csv(OUT_DIR / "test_classification.csv")

    feature_cols = [c for c in train_df.columns if c != "HighEnergy"]

    X_train = train_df[feature_cols].values
    y_train = train_df["HighEnergy"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["HighEnergy"].values

    return X_train, X_test, y_train, y_test, feature_cols


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """Treina, avalia e retorna métricas, matriz de confusão e predições."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    print(f"\n=== {name} ===")
    print("Acurácia teste:", acc)
    print("Matriz de confusão (linhas = verdadeiro, colunas = predito):")
    print(cm)

    results = {
        "model": name,
        "accuracy_test": acc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    return results, y_pred


# --- NOVA FUNÇÃO PARA PLOTAGEM ---
def plot_model_comparison(results_df, output_path):
    """Gera e salva um gráfico de barras comparando a acurácia dos modelos."""

    # Ordena os modelos pela acurácia para um gráfico mais claro
    df_sorted = results_df.sort_values("accuracy_test", ascending=False)

    plt.style.use('seaborn-v0_8-whitegrid')  # Estilo visual
    fig, ax = plt.subplots(figsize=(10, 6))

    # Cria as barras
    bars = sns.barplot(
        x='model',
        y='accuracy_test',
        data=df_sorted,
        ax=ax,
        palette='viridis'  # Paleta de cores
    )

    # Adiciona os valores de acurácia em cima de cada barra
    for bar in bars.patches:
        ax.annotate(
            f'{bar.get_height():.1%}',  # Formata como porcentagem
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha='center',
            va='center',
            size=11,
            xytext=(0, 8),
            textcoords='offset points'
        )

    # Configurações do gráfico
    ax.set_title('Comparação de Acurácia dos Modelos no Conjunto de Teste', fontsize=16)
    ax.set_xlabel('Modelo', fontsize=12)
    ax.set_ylabel('Acurácia', fontsize=12)
    ax.set_ylim(0, 1.0)  # Eixo Y de 0 a 100%
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotaciona os nomes dos modelos

    plt.tight_layout()  # Ajusta o layout para não cortar os rótulos

    # Salva a figura
    plt.savefig(output_path)
    print(f"\nGráfico de comparação salvo em: {output_path}")


def main():
    X_train, X_test, y_train, y_test, feature_cols = load_data()

    all_results = []
    predictions = {}

    # --- Treinamento e Avaliação dos Modelos ---
    # 1) Logistic Regression
    log_reg = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=1000, random_state=42)
    res, y_pred_log = evaluate_model("LogisticRegression", log_reg, X_train, y_train, X_test, y_test)
    all_results.append(res)
    predictions["LogisticRegression"] = y_pred_log

    # 2) LDA
    lda = LinearDiscriminantAnalysis()
    res, y_pred_lda = evaluate_model("LDA", lda, X_train, y_train, X_test, y_test)
    all_results.append(res)
    predictions["LDA"] = y_pred_lda

    # 3) k-NN
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn = KNeighborsClassifier()
    param_grid_knn = {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]}
    grid_knn = GridSearchCV(knn, param_grid_knn, cv=cv, scoring="accuracy", n_jobs=-1)
    grid_knn.fit(X_train, y_train)
    best_knn = grid_knn.best_estimator_
    print("\nk-NN melhor params:", grid_knn.best_params_)
    res, y_pred_knn = evaluate_model("kNN", best_knn, X_train, y_train, X_test, y_test)
    all_results.append(res)
    predictions["kNN"] = y_pred_knn

    # 4) SVM
    svm = SVC(kernel="rbf", probability=False, random_state=42)
    param_grid_svm = {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1]}
    grid_svm = GridSearchCV(svm, param_grid_svm, cv=cv, scoring="accuracy", n_jobs=-1)
    grid_svm.fit(X_train, y_train)
    best_svm = grid_svm.best_estimator_
    print("\nSVM melhor params:", grid_svm.best_params_)
    res, y_pred_svm = evaluate_model("SVM_RBF", best_svm, X_train, y_train, X_test, y_test)
    all_results.append(res)
    predictions["SVM_RBF"] = y_pred_svm

    # --- Fim do Treinamento ---

    # Salva o resumo em CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "hw3_classification_summary.csv", index=False)
    print("\nResumo salvo em:", OUT_DIR / "hw3_classification_summary.csv")

    # --- GERAÇÃO DO GRÁFICO ---
    plot_model_comparison(results_df, FIGURES_DIR / "model_comparison_bar.png")

    # --- TESTE DE MCNEMAR ---
    print("\n" + "=" * 50)
    print("Análise Estatística Comparativa (Teste de McNemar)")
    print("=" * 50)
    print("\nComparando k-NN vs. LDA:")
    tb = mcnemar_table(y_target=y_test, y_model1=predictions["kNN"], y_model2=predictions["LDA"])
    print("Tabela de Contingência (k-NN vs. LDA):")
    print("[[Ambos acertaram, k-NN errou/LDA acertou], \n [k-NN acertou/LDA errou, Ambos erraram]]")
    print(tb)
    chi2, p_value = mcnemar(ary=tb, exact=True)
    print(f"\nP-valor: {p_value:.4f}")
    alpha = 0.05
    if p_value < alpha:
        print(
            f"Conclusão: Como p-valor ({p_value:.4f}) < {alpha}, a diferença de desempenho é ESTATISTICAMENTE SIGNIFICATIVA.")
    else:
        print(
            f"Conclusão: Como p-valor ({p_value:.4f}) >= {alpha}, a diferença de desempenho NÃO é estatisticamente significativa.")


if __name__ == "__main__":
    main()
