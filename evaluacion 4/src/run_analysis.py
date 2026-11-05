"""Base reproducible para evaluación 4: EDA, clasificación y análisis no supervisado."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "results"
SEED = 42


def load_uci_csv(path):
    """Los CSV oficiales contienen metadatos antes de la fila 'class,...'."""
    with path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if line.strip().strip('"').lower().startswith("class,"):
                break
        else:
            raise ValueError(f"No se encontro cabecera class en {path}")
    df = pd.read_csv(path, skiprows=index, na_values=["na", "NA", ""])
    if "class" not in df.columns:
        raise ValueError("No existe columna class")
    df["class"] = df["class"].astype(str).str.strip().str.strip('"')
    if not set(df["class"]).issubset({"pos", "neg"}):
        raise ValueError("Etiquetas distintas a pos/neg")
    x = df.drop(columns="class").apply(pd.to_numeric, errors="coerce")
    y = df["class"].map({"neg": 0, "pos": 1}).astype(int)
    return x, y


def report_metrics(name, model, x, y):
    prediction = model.predict(x)
    score = model.predict_proba(x)[:, 1]
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "model": name, "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "pr_auc": average_precision_score(y, score),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "cost_10fp_500fn": 10 * fp + 500 * fn,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_file, test_file = (
        DATA / "aps_failure_training_set.csv", DATA / "aps_failure_test_set.csv"
    )
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError("Ejecuta primero: python src/download_data.py")
    x, y = load_uci_csv(train_file)
    x_test, y_test = load_uci_csv(test_file)
    x_test = x_test.reindex(columns=x.columns)
    missing = x.isna().mean().sort_values(ascending=False)
    missing.rename("missing_fraction").to_csv(OUT / "missing_values.csv")
    pd.DataFrame({"clase": ["neg", "pos"], "conteo": [(y == 0).sum(), (y == 1).sum()]}).to_csv(
        OUT / "class_balance.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    missing.head(20).sort_values().plot.barh(ax=ax)
    ax.set(title="20 variables con mas faltantes", xlabel="Proporcion faltante")
    fig.tight_layout()
    fig.savefig(OUT / "missing_values.png", dpi=160)
    plt.close(fig)

    # Entrenamiento original -> validacion estratificada; test oficial reservado.
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, stratify=y, test_size=0.2, random_state=SEED
    )
    models = {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=SEED)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                             class_weight="balanced_subsample",
                                             random_state=SEED, n_jobs=-1)),
        ]),
    }
    validation = []
    for name, model in models.items():
        print("Entrenando:", name)
        model.fit(x_train, y_train)
        validation.append(report_metrics(name, model, x_val, y_val))
    pd.DataFrame(validation).to_csv(OUT / "supervised_validation.csv", index=False)

    # Elegimos segun PR-AUC en validacion; la prueba no interviene en la eleccion.
    best_name = max(validation, key=lambda item: item["pr_auc"])["model"]
    best = models[best_name]
    # Evaluacion final sin reentrenar: el test oficial permanece intacto.
    pd.DataFrame([report_metrics(best_name, best, x_test, y_test)]).to_csv(
        OUT / "supervised_test.csv", index=False
    )
    print("Mejor PR-AUC en validacion:", best_name)

    # Para una demostracion computacionalmente manejable, muestra sin etiquetas como entrada.
    x_unsup = x_train.sample(n=min(10000, len(x_train)), random_state=SEED)
    labels_for_interpretation = y_train.loc[x_unsup.index]  # solo para comparar DESPUES
    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    z = prep.fit_transform(x_unsup)  # ajusta solo subset de entrenamiento
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=2, random_state=SEED)
    z2 = pca.fit_transform(z)

    kmeans = MiniBatchKMeans(n_clusters=2, n_init=10, random_state=SEED, batch_size=1024)
    clusters = kmeans.fit_predict(z)
    anomalies = IsolationForest(n_estimators=150, contamination=0.02,
                                random_state=SEED, n_jobs=-1).fit_predict(z)
    anomalies = (anomalies == -1).astype(int)
    sample_n = min(2000, len(z))
    sample_idx = np.random.default_rng(SEED).choice(len(z), size=sample_n, replace=False)
    silhouette = silhouette_score(z[sample_idx], clusters[sample_idx]) if len(set(clusters[sample_idx])) > 1 else np.nan
    pd.DataFrame([{"algorithm": "MiniBatchKMeans", "silhouette_sample": silhouette},
                  {"algorithm": "IsolationForest", "silhouette_sample": np.nan}]).to_csv(
        OUT / "unsupervised_metrics.csv", index=False
    )
    profiles = pd.DataFrame({
        "cluster": clusters, "anomaly": anomalies,
        "aps_positive_for_interpretation_only": labels_for_interpretation.to_numpy(),
    })
    profiles.groupby("cluster").agg(count=("anomaly", "size"),
                                     anomaly_rate=("anomaly", "mean"),
                                     aps_positive_rate=("aps_positive_for_interpretation_only", "mean")).to_csv(
        OUT / "cluster_profiles.csv"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(z2[:, 0], z2[:, 1], c=clusters, s=4, alpha=0.5)
    ax.set(title="PCA: clusters sin utilizar etiquetas", xlabel="Componente 1", ylabel="Componente 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(OUT / "pca_clusters.png", dpi=160)
    plt.close(fig)
    print("Resultados exportados a", OUT)


if __name__ == "__main__":
    main()
