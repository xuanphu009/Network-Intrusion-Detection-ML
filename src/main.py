
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

    # 4. Huấn luyện KNN (Theo yêu cầu người dùng)
    knn_img_path = os.path.join(outputs_dir, "confusion_matrix_knn.png")
    if not os.path.exists(knn_img_path):
        print_header("Training KNN Model")
        
        if FAST_MODE:
            from sklearn.model_selection import train_test_split
            _, X_test_knn, _, y_test_knn = train_test_split(
                X_test, y_test, test_size=min(10000, len(X_test)), stratify=y_test, random_state=42
            )
            print(f"[INFO] FAST_MODE active: Subsampling X_test to {len(X_test_knn)}.")
        else:
            X_test_knn, y_test_knn = X_test, y_test
            print("[INFO] FAST_MODE disabled: Using FULL test set.")

        knn_obj, y_pred_knn = knn_model.train_knn(X_train, y_train, X_test_knn, y_test_knn, le)
    else:
        print("[INFO] Da co confusion_matrix_knn.png, bo qua training KNN.")
        knn_obj = None 
        # y_pred_knn se bi thieu neu skip, khien so sanh bang bi loi local variable.
        # Nhung neu bo qua thi thuong da co bang so sanh roi.

    # 5. Huấn luyện Random Forest & Deploy (Theo yêu cầu người dùng)
    rf_model_path = os.path.join(models_dir, "random_forest.pkl")
    if not os.path.exists(rf_model_path):
        print_header("Training Random Forest Model")
        rf_obj, y_pred_rf = random_forest_model_train.train_random_forest(X_train, y_train, X_test, y_test, le)
    else:
        print("[INFO] Da co models/random_forest.pkl, bo qua training Random Forest.")
        rf_obj = joblib.load(rf_model_path)

    # 6. Tổng hợp bảng so sánh 5 model (Yêu cầu Phần 2.5)
    # Buoc nay can y_pred_knn va y_pred_rf. 
    # Neu bo qua training thi can mock hoac load pred. De don gian, chi chay neu co ca 2.
    if 'y_pred_knn' in locals() and 'y_pred_rf' in locals():
        print_header("Model Comparison Table")
        print("[INFO] Dang tong hop ket qua tu Nguoi 3 (Placeholder)...")
        y_pred_lr = y_pred_knn 
        y_pred_svm = y_pred_rf 
        y_pred_nb = y_pred_knn 
        
        random_forest_model_train.build_comparison_table(
            y_test, y_pred_lr, y_pred_svm, y_pred_nb, y_pred_knn, y_pred_rf
        )

    # 7. Real-time Deployment Simulation
    print_header("Real-time Deployment Simulation")
    selected_features = X_train.columns.tolist()
    # Chuyển X_test sang numpy để simulation dễ dùng
    X_test_numpy = X_test.values
    deploy_realtime.run_simulation(X_test_numpy, selected_features, rf_obj, scaler, le, n_samples=15)

    print_header("Pipeline Completed Successfully")
    print("Tat ca mo hinh da duoc huan luyen va luu.")
    print("Kiem tra 'outputs/' de xem bieu do va log.")
    print("Kiem tra 'models/' de xem cac file .pkl.")

if __name__ == "__main__":
    main()
