# src/random_forest_model_train.py
# Phương — Cell 15+16: Train Random Forest + lưu .pkl + bảng so sánh 5 model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, \
                            precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import time
import os

def train_random_forest(X_train_scaled, y_train_bal, X_test_scaled, y_test, le):
    print("=" * 50)
    print("Training Random Forest (100 cây)...")
    t0 = time.time()
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train_bal)
    print(f"Done in {time.time()-t0:.1f}s")
    
    y_pred_rf = rf_model.predict(X_test_scaled)
    print("\nClassification Report - Random Forest:")
    print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
    
    # Confusion Matrix
    os.makedirs('outputs', exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred_rf),
                annot=True, fmt='d', cmap='Oranges',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Random Forest - Best Model')
    plt.ylabel('Thuc te'); plt.xlabel('Du doan')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix_rf.png', dpi=150)
    plt.show()
    print("Da luu: outputs/confusion_matrix_rf.png")
    
    # Luu model ra .pkl
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/random_forest.pkl')
    print("Da luu: models/random_forest.pkl")
    
    return rf_model, y_pred_rf


def build_comparison_table(y_test, y_pred_lr, y_pred_svm, y_pred_nb, y_pred_knn, y_pred_rf):
    """Cell 18 — Tổng hợp bảng so sánh 5 model (copy kết quả từ Phát/Người 3)"""
    models = ['Logistic Regression', 'SVM', 'Naive Bayes', 'KNN', 'Random Forest']
    preds  = [y_pred_lr, y_pred_svm, y_pred_nb, y_pred_knn, y_pred_rf]
    
    rows = []
    for name, pred in zip(models, preds):
        rows.append({
            'Model'    : name,
            'Accuracy' : round(accuracy_score(y_test, pred), 4),
            'Precision': round(precision_score(y_test, pred, average='weighted', zero_division=0), 4),
            'Recall'   : round(recall_score(y_test, pred, average='weighted', zero_division=0), 4),
            'F1-Score' : round(f1_score(y_test, pred, average='weighted', zero_division=0), 4),
        })
    
    df_cmp = pd.DataFrame(rows).sort_values('F1-Score', ascending=False)
    print("\nComparison Table - 5 Models:")
    print(df_cmp.to_string(index=False))
    
    os.makedirs('outputs', exist_ok=True)
    df_cmp.to_csv('outputs/model_comparison.csv', index=False)
    print("Da luu: outputs/model_comparison.csv")
    return df_cmp