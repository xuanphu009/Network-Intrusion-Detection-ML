# src/knn_model.py
# Phương — Cell 14: Train KNN + Confusion Matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

def train_knn(X_train_scaled, y_train_bal, X_test_scaled, y_test, le):
    print("=" * 50)
    print("Training KNN (k=5)...")
    
    knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn_model.fit(X_train_scaled, y_train_bal)
    
    y_pred_knn = knn_model.predict(X_test_scaled)
    print("\nClassification Report - KNN:")
    print(classification_report(y_test, y_pred_knn, target_names=le.classes_))
    
    # Ve Confusion Matrix
    os.makedirs('outputs', exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred_knn),
                annot=True, fmt='d', cmap='Greens',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - KNN')
    plt.ylabel('Thuc te'); plt.xlabel('Du doan')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix_knn.png', dpi=150)
    plt.close() # Close to avoid display issues
    print("Da luu: outputs/confusion_matrix_knn.png")

    # Luu model
    os.makedirs('models', exist_ok=True)
    joblib.dump(knn_model, 'models/knn.pkl')
    print("Da luu: models/knn.pkl")
    
    return knn_model, y_pred_knn