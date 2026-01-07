import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# caminhos
BASE_DIR = Path(r"/")
DATA_CLEAN = BASE_DIR / "data" / "Data-Melbourne_F_clean.csv"
OUTPUT_DIR = BASE_DIR / "outputs_hw3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA_CLEAN)

    # define corte (mediana) para alto consumo
    cutoff = df["total_grid"].median()
    df["HighEnergy"] = (df["total_grid"] > cutoff).astype(int)

    feature_cols = [
        "avg_outflow", "avg_inflow", "Am", "BOD", "COD", "TN",
        "T", "TM", "Tm", "SLP", "H", "PP", "VV", "V", "VM", "VG",
        "year", "month", "day"
    ]

    X = df[feature_cols].values
    y = df["HighEnergy"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # log em PP
    def log_transform(X, cols, names):
        Xc = X.copy()
        for col in cols:
            idx = names.index(col)
            Xc[:, idx] = np.log1p(np.maximum(Xc[:, idx], 0.0))
        return Xc

    cols_to_log = ["PP"]
    X_train_log = log_transform(X_train, cols_to_log, feature_cols)
    X_test_log  = log_transform(X_test,  cols_to_log, feature_cols)

    # padronização
    scaler = StandardScaler()
    X_train_prep = scaler.fit_transform(X_train_log)
    X_test_prep  = scaler.transform(X_test_log)

    # salvar arquivos para HW3
    train_df = pd.DataFrame(X_train_prep, columns=feature_cols)
    train_df["HighEnergy"] = y_train
    train_df.to_csv(OUTPUT_DIR / "train_classification.csv", index=False)

    test_df = pd.DataFrame(X_test_prep, columns=feature_cols)
    test_df["HighEnergy"] = y_test
    test_df.to_csv(OUTPUT_DIR / "test_classification.csv", index=False)

    print("Cutoff mediano total_grid:", cutoff)
    print("Treino:", train_df.shape, "Teste:", test_df.shape)
    print("Proporção classe 1 (treino):", train_df["HighEnergy"].mean())

if __name__ == "__main__":
    main()
