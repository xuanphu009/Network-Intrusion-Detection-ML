
import os
import sys
import pandas as pd
import json
import joblib
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler

import numpy as np

# Import các module từ src
from preprocess import DATA_PATH, OUTPUT_DIR
import knn_model
import random_forest_model_train
import logistic_regression_model
import svm_model
import naive_bayes_model
import deploy_realtime

# Cấu hình
FAST_MODE = False  # Chuyển thành False nếu muốn chạy trên toàn bộ dữ liệu test (sẽ chậm)

def print_header(text):
    print("\n" + "=" * 60)
    print(f"--- {text.upper()} ---")
    print("=" * 60)

def main():
    print_header("Network Intrusion Detection System (NIDS) - Pipeline")
    
    # 1. Kiểm tra môi trường & Dữ liệu
    base_dir = os.path.dirname(__file__)
    outputs_dir = os.path.normpath(os.path.join(base_dir, "../outputs"))
    models_dir = os.path.normpath(os.path.join(base_dir, "../models"))
    balanced_data_path = os.path.join(outputs_dir, "cicids2017_balanced.csv")
    
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 2. Bước Tiền xử lý & Cân bằng dữ liệu
    cleaned_data_path = os.path.join(outputs_dir, "cicids2017_cleaned.csv")
    if not os.path.exists(balanced_data_path):
        print("[WARN] Khong tim thay du lieu da can bang.")
        
        # Chạy preprocess.py nếu chưa có cleaned
        if not os.path.exists(cleaned_data_path):
            print("\n[STEP 1] Chay Preprocessing...")
            os.system(f"\"{sys.executable}\" \"{os.path.join(base_dir, 'preprocess.py')}\"")
        else:
            print("[INFO] Da co cicids2017_cleaned.csv, bo qua Step 1.")
        
        # Chạy balance_and_select.py
        print("\n[STEP 2] Chay Balancing & Feature Selection...")
        os.system(f"\"{sys.executable}\" \"{os.path.join(base_dir, 'balance_and_select.py')}\"")
    else:
        print("[INFO] Da tim thay du lieu da xu ly tai outputs/cicids2017_balanced.csv")

    # 3. Load Dữ liệu
    print_header("Loading Data")
    df = pd.read_csv(balanced_data_path)
    
    # Tách Train/Test dựa trên cột 'split' đã lưu trong balance_and_select.py
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    X_train = train_df.drop(['Label', 'split'], axis=1)
    y_train = train_df['Label']
    X_test = test_df.drop(['Label', 'split'], axis=1)
    y_test = test_df['Label']
    
    # Load LabelEncoder và Scaler (nếu đã có)
    le_path = os.path.join(models_dir, "label_encoder.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    
    if os.path.exists(le_path) and os.path.exists(scaler_path):
        le = joblib.load(le_path)
        scaler = joblib.load(scaler_path)
        print("[INFO] Da load LabelEncoder va Scaler tu models/")
    else:
        print("[WARN] Thieu Scaler hoac LabelEncoder. Dang tao moi...")
        le = LabelEncoder()
        with open(os.path.join(outputs_dir, "label_classes.json"), "r") as f:
            le.classes_ = np.array(json.load(f))
            
        scaler = StandardScaler()
        scaler.fit(X_train) # Fit lại để có object scaler

    # 4. Subsampling data if FAST_MODE (to speed up Training AND Evaluation)
    if FAST_MODE:
        from sklearn.model_selection import train_test_split
        print(f"[INFO] FAST_MODE active: Subsampling BOTH Train and Test sets to 10000 samples.")
        
        # Subsample Train
        _, X_train, _, y_train = train_test_split(
            X_train, y_train, test_size=min(10000, len(X_train)), stratify=y_train, random_state=42
        )
        
        # Subsample Test
        _, X_test_eval, _, y_test_eval = train_test_split(
            X_test, y_test, test_size=min(10000, len(X_test)), stratify=y_test, random_state=42
        )
    else:
        X_test_eval, y_test_eval = X_test, y_test
        print("[INFO] FAST_MODE disabled: Using FULL datasets for Training and Evaluation.")

    # 5. Huấn luyện/Load KNN
    knn_path = os.path.join(models_dir, "knn.pkl")
    if not os.path.exists(knn_path):
        print_header("Training KNN Model")
        knn_obj, y_pred_knn = knn_model.train_knn(X_train, y_train, X_test_eval, y_test_eval, le)
    else:
        print("[INFO] Da co models/knn.pkl, dang load va predict...")
        knn_obj = joblib.load(knn_path)
        y_pred_knn = knn_obj.predict(X_test_eval)

    # 6. Huấn luyện/Load Random Forest
    rf_model_path = os.path.join(models_dir, "random_forest.pkl")
    if not os.path.exists(rf_model_path):
        print_header("Training Random Forest Model")
        rf_obj, y_pred_rf = random_forest_model_train.train_random_forest(X_train, y_train, X_test_eval, y_test_eval, le)
    else:
        print("[INFO] Da co models/random_forest.pkl, dang load va predict...")
        rf_obj = joblib.load(rf_model_path)
        y_pred_rf = rf_obj.predict(X_test_eval)

    # 7. Huấn luyện/Load Logistic Regression
    print_header("Training Logistic Regression")
    lr_path = os.path.join(models_dir, "logistic_regression.pkl")
    if not os.path.exists(lr_path):
        lr_obj, y_pred_lr = logistic_regression_model.train_logistic_regression(X_train, y_train, X_test_eval, y_test_eval, le)
    else:
        print("[INFO] Da co models/logistic_regression.pkl, dang load va predict...")
        lr_obj = joblib.load(lr_path)
        y_pred_lr = lr_obj.predict(X_test_eval)

    # 8. Huấn luyện/Load SVM
    print_header("Training SVM (LinearSVC)")
    svm_path = os.path.join(models_dir, "svm_linearsvc.pkl")
    if not os.path.exists(svm_path):
        svm_obj, y_pred_svm = svm_model.train_svm(X_train, y_train, X_test_eval, y_test_eval, le)
    else:
        print("[INFO] Da co models/svm_linearsvc.pkl, dang load va predict...")
        svm_obj = joblib.load(svm_path)
        y_pred_svm = svm_obj.predict(X_test_eval)

    # 9. Huấn luyện/Load Naive Bayes
    print_header("Training Naive Bayes")
    nb_path = os.path.join(models_dir, "naive_bayes.pkl")
    if not os.path.exists(nb_path):
        nb_obj, y_pred_nb = naive_bayes_model.train_naive_bayes(X_train, y_train, X_test_eval, y_test_eval, le)
    else:
        print("[INFO] Da co models/naive_bayes.pkl, dang load va predict...")
        nb_obj = joblib.load(nb_path)
        y_pred_nb = nb_obj.predict(X_test_eval)

    # 10. Tổng hợp bảng so sánh 5 model
    print_header("Model Comparison Table")
    random_forest_model_train.build_comparison_table(
        y_test_eval, y_pred_lr, y_pred_svm, y_pred_nb, y_pred_knn, y_pred_rf
    )

    # 11. Real-time Deployment Simulation
    print_header("Real-time Deployment Simulation")
    selected_features = X_train.columns.tolist()
    # Chuyển X_test_eval sang numpy để simulation dễ dùng
    X_test_numpy = X_test_eval.values
    deploy_realtime.run_simulation(X_test_numpy, selected_features, rf_obj, scaler, le, n_samples=15)

    print_header("Pipeline Completed Successfully")
    print("Tat ca mo hinh da duoc huan luyen va luu.")
    print("Kiem tra 'outputs/' de xem bieu do va log.")
    print("Kiem tra 'models/' de xem cac file .pkl.")

if __name__ == "__main__":
    main()
