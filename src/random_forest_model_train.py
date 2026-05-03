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
    print("\n" + "*" * 30)
    print("FINAL MODEL COMPARISON TABLE")
    print("*" * 30)
    print(df_cmp.to_string(index=False))
    
    os.makedirs('outputs', exist_ok=True)
    df_cmp.to_csv('outputs/model_comparison.csv', index=False)
    print("\n[OK] Da luu: outputs/model_comparison.csv")
    
    # Ve bieu do so sanh
    plot_comparison_chart(df_cmp)
    
    return df_cmp

def plot_comparison_chart(df_cmp):
    """Vẽ biểu đồ cột so sánh các chỉ số giữa các model"""
    plt.figure(figsize=(12, 7))
    
    # Chuyển dataframe sang dạng long-form để vẽ seaborn dễ hơn
    df_plot = df_cmp.melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    ax = sns.barplot(data=df_plot, x='Model', y='Score', hue='Metric', palette='viridis')
    
    # Thêm số liệu cụ thể lên đầu mỗi cột
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
    
    plt.title('Performance Comparison of 5 Models', fontsize=15, fontweight='bold')
    plt.ylim(0, 1.2)
    plt.ylabel('Score (0.0 - 1.0)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = 'outputs/model_comparison_chart.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Da luu bieu do so sanh: {save_path}")