# src/logistic_regression_model.py
# Train Logistic Regression: classification_report + Confusion Matrix + Nhận xét Recall

import os
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../outputs"))
MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../models"))


def _print_recall_comment(report_dict, class_names):
    """Nhận xét Recall cho từng loại tấn công."""
    print("\n  --- Nhan xet Recall theo tung loai tan cong (Logistic Regression) ---")
    for cls in class_names:
        if cls not in report_dict:
            continue
        recall = report_dict[cls]["recall"]
        support = report_dict[cls]["support"]
        if recall >= 0.90:
            level = "Rat tot"
        elif recall >= 0.75:
            level = "Kha tot"
        elif recall >= 0.50:
            level = "Trung binh"
        else:
            level = "Kem - can cai thien"
        print(f"  [{cls}]  Recall = {recall:.4f}  ({level})  | Support = {support}")


def train_logistic_regression(X_train, y_train, X_test, y_test, le):
    print("=" * 60)
    print("Training Logistic Regression...")
    t0 = time.time()

    lr_model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="auto",
        random_state=42,
        n_jobs=-1,
    )
    lr_model.fit(X_train, y_train)
    print(f"  Done in {time.time() - t0:.1f}s")

    y_pred = lr_model.predict(X_test)

    # Classification Report
    print("\n  Classification Report - Logistic Regression:")
    report_str = classification_report(y_test, y_pred, target_names=le.classes_)
    report_dict = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True
    )
    print(report_str)

    # Nhận xét Recall
    _print_recall_comment(report_dict, le.classes_)

    # Confusion Matrix
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True, fmt="d", cmap="Blues",
        xticklabels=le.classes_, yticklabels=le.classes_,
    )
    plt.title("Confusion Matrix - Logistic Regression", fontsize=14, fontweight="bold")
    plt.ylabel("Thuc te", fontsize=12)
    plt.xlabel("Du doan", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix_lr.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [OK] Da luu: {save_path}")

    # Lưu model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    print("  [OK] Da luu: models/logistic_regression.pkl")

    return lr_model, y_pred
