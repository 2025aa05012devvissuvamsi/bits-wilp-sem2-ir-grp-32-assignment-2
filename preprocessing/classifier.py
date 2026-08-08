"""
Document classification (assignment requirement C).

Labels come from `seed_source` in the crawl metadata (the 7 seed topics
the crawler started from) rather than raw Wikipedia categories, since
categories are noisy free text (e.g. "CS1 German-language sources") while
seed_source gives a clean topical label already present in the pipeline.

The corpus is small (49 docs / 7 classes), so headline performance is
reported via stratified k-fold cross-validation rather than a single
train/test split, which would leave only a handful of test samples.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from scipy import sparse

PROCESSED_DIR = "data/processed"
FIGURES_DIR = "data/processed/figures"

MODELS = {
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
}


def load_features_and_labels(processed_dir: str = PROCESSED_DIR):
    matrix = sparse.load_npz(os.path.join(processed_dir, "tfidf_matrix.npz"))
    doc_ids = pd.read_csv(os.path.join(processed_dir, "tfidf_doc_ids.csv"))["doc_id"]
    corpus_index = pd.read_csv(os.path.join(processed_dir, "corpus_index.csv")).set_index("doc_id")
    labels = corpus_index.loc[doc_ids, "seed_source"].reset_index(drop=True)
    return matrix, labels


def cross_validated_report(matrix, labels, n_splits: int = 4) -> pd.DataFrame:
    """Mean accuracy / macro-F1 per model under stratified k-fold CV."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []
    for name, model in MODELS.items():
        acc = cross_val_score(model, matrix, labels, cv=skf, scoring="accuracy")
        f1 = cross_val_score(model, matrix, labels, cv=skf, scoring="f1_macro")
        rows.append({
            "model": name,
            "cv_folds": n_splits,
            "mean_accuracy": float(acc.mean()),
            "std_accuracy": float(acc.std()),
            "mean_macro_f1": float(f1.mean()),
            "std_macro_f1": float(f1.std()),
        })
    return pd.DataFrame(rows)


def confusion_matrix_plot(matrix, labels, model, out_path: str, n_splits: int = 4):
    """Confusion matrix built from cross-validated (held-out) predictions."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds = cross_val_predict(model, matrix, labels, cv=skf)

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(labels, preds, ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix (cross-validated predictions)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return preds


def holdout_classification_report(matrix, labels, model) -> str:
    """A single stratified train/test split, for a human-readable precision/recall table."""
    X_train, X_test, y_train, y_test = train_test_split(
        matrix, labels, test_size=0.25, stratify=labels, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, zero_division=0)


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)

    matrix, labels = load_features_and_labels()
    print("Class distribution:")
    print(labels.value_counts())

    n_splits = min(4, labels.value_counts().min())
    cv_results = cross_validated_report(matrix, labels, n_splits=n_splits)
    cv_results.to_csv(os.path.join(PROCESSED_DIR, "classification_cv_results.csv"), index=False)
    print(cv_results)

    best_model_name = cv_results.sort_values("mean_macro_f1", ascending=False).iloc[0]["model"]
    best_model = MODELS[best_model_name]
    print(f"\nBest model by macro-F1: {best_model_name}")

    confusion_matrix_plot(
        matrix, labels, best_model,
        os.path.join(FIGURES_DIR, "confusion_matrix.png"),
        n_splits=n_splits,
    )

    report = holdout_classification_report(matrix, labels, best_model)
    print("\nHold-out classification report:\n", report)
    with open(os.path.join(PROCESSED_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best model: {best_model_name}\n\n{report}")

    print(f"\nSaved classification results to {PROCESSED_DIR}")
