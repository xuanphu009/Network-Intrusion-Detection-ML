# ==========================================================
# CIC-IDS2017 - Class Imbalance Handling & Feature Selection
# Người thực hiện: Phú (Người 2)
#
# Input : outputs/cicids2017_cleaned.csv   (từ người 1)
# Output: outputs/cicids2017_balanced.csv  (cho người 3)
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib

# ==========================================================
# 1. ĐƯỜNG DẪN
# ==========================================================

BASE_DIR   = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "../outputs")
INPUT_FILE = os.path.join(OUTPUT_DIR, "cicids2017_cleaned.csv")

def run_balancing():
    # ==========================================================
    # 2. LOAD DATA TỪ NGƯỜI 1
    # ==========================================================

    print("Loading cleaned data...")
    df = pd.read_csv(INPUT_FILE)
    print("Shape:", df.shape)
    print("\nLabel distribution (before):")
    # print(df["Label"].value_counts()) # Bo qua print truc tiep de tranh Unicode error

    # Clear non-ascii labels (Web Attack)
    df["Label"] = df["Label"].str.replace("\ufffd", "-", regex=False)
    
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Label"])

    print("\nClasses sau khi encode:")
    for i, cls in enumerate(le.classes_):
        print(f"  {i} -> {cls}")

    # Lưu le ra models/
    os.makedirs(os.path.join(BASE_DIR, "../models"), exist_ok=True)
    joblib.dump(le, os.path.join(BASE_DIR, "../models/label_encoder.pkl"))
    print("Saved: models/label_encoder.pkl")

    # ==========================================================
    # 4. TÁCH X, y
    # ==========================================================

    X = df.drop("Label", axis=1)
    y = df["Label"]

    # ==========================================================
    # 5. FEATURE SELECTION — 18 FEATURES (yêu cầu 2.4)
    # Làm trước Scale để nhẹ hơn
    # ==========================================================

    selected_features = [
        'Flow Duration',
        'Total Fwd Packets',
        'Total Backward Packets',
        'Total Length of Fwd Packets',
        'Total Length of Bwd Packets',
        'Fwd Packet Length Mean',
        'Bwd Packet Length Mean',
        'Flow Bytes/s',
        'Flow Packets/s',
        'Packet Length Mean',
        'Packet Length Std',
        'SYN Flag Count',
        'ACK Flag Count',
        'FIN Flag Count',
        'RST Flag Count',
        'PSH Flag Count',
        'URG Flag Count'
    ]

    # Chỉ lấy feature có trong dataset
    available_features = [f for f in selected_features if f in X.columns]
    missing_features   = [f for f in selected_features if f not in X.columns]

    print(f"\nAvailable features : {len(available_features)}")
    print(f"Missing features   : {missing_features}")

    X = X[available_features]

    # ==========================================================
    # 6. SCALE FEATURES  (yêu cầu 2.3)
    # ==========================================================

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=available_features)

    print("\nScaling done.")

    # Lưu scaler ra models/
    joblib.dump(scaler, os.path.join(BASE_DIR, "../models/scaler.pkl"))
    print("Saved: models/scaler.pkl")

    # ==========================================================
    # 7. TRAIN / TEST SPLIT (80/20)
    # ==========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTrain size: {X_train.shape[0]}")
    print(f"Test  size: {X_test.shape[0]}")

    # ==========================================================
    # 8. SMOTE — OVERSAMPLE MINORITY  (yêu cầu 2.3)
    # Đưa minority classes lên 10% của majority
    # ==========================================================

    majority_count = y_train.value_counts().max()
    threshold      = int(majority_count * 0.10)

    # Chỉ oversample class nào nhỏ hơn threshold
    sampling_up = {
        cls: max(count, threshold)
        for cls, count in y_train.value_counts().items()
        if count < majority_count
    }

    print(f"\nMajority class size : {majority_count}")
    print(f"SMOTE threshold     : {threshold}")

    smote = SMOTE(sampling_strategy=sampling_up, random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("After SMOTE:", pd.Series(y_train_res).value_counts().to_dict())

    # ==========================================================
    # 9. RANDOM UNDER SAMPLER — GIẢM BENIGN  (yêu cầu 2.3)
    # ==========================================================

    benign_label = list(le.classes_).index("BENIGN")
    benign_count_after_smote = pd.Series(y_train_res).value_counts()[benign_label]

    # Giảm BENIGN xuống còn tối đa 5x threshold
    target_benign  = min(benign_count_after_smote, threshold * 5)
    sampling_down  = {benign_label: target_benign}

    rus = RandomUnderSampler(sampling_strategy=sampling_down, random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train_res, y_train_res)

    print("After UnderSample:", pd.Series(y_train_res).value_counts().to_dict())

    # ==========================================================
    # 10. BIỂU ĐỒ SO SÁNH TRƯỚC / SAU
    # ==========================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Trước
    y_original_labels = le.inverse_transform(y_train)
    pd.Series(y_original_labels).value_counts().plot(
        kind="bar", ax=axes[0], color="steelblue"
    )
    axes[0].set_title("Before Balancing (Train)")
    axes[0].set_xlabel("Label")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    # Sau
    y_res_labels = le.inverse_transform(y_train_res)
    pd.Series(y_res_labels).value_counts().plot(
        kind="bar", ax=axes[1], color="darkorange"
    )
    axes[1].set_title("After SMOTE + UnderSampling (Train)")
    axes[1].set_xlabel("Label")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_balance_comparison.png"))
    plt.show()
    print("Saved: class_balance_comparison.png")

    # ==========================================================
    # 11. XUẤT FILE CHO NGƯỜI 3
    # ==========================================================

    # Ghép train đã balance
    train_out = X_train_res.copy()
    train_out["Label"] = y_train_res.values
    train_out["split"] = "train"

    # Test giữ nguyên (không balance test set)
    test_out = X_test.copy()
    test_out["Label"] = y_test.values
    test_out["split"] = "test"

    final_df = pd.concat([train_out, test_out], ignore_index=True)
    final_df.to_csv(os.path.join(OUTPUT_DIR, "cicids2017_balanced.csv"), index=False)

    # Lưu thêm danh sách tên class để người 3 dùng
    import json
    with open(os.path.join(OUTPUT_DIR, "label_classes.json"), "w") as f:
        json.dump(list(le.classes_), f)

    # ==========================================================
    # 12. SUMMARY
    # ==========================================================

    print("\n" + "=" * 50)
    print("SUMMARY - Nguoi 2 (Phu)")
    print("=" * 50)
    print(f"Features da chon  : {len(available_features)}")
    print(f"Train (balanced)  : {X_train_res.shape[0]} samples")
    print(f"Test  (original)  : {X_test.shape[0]} samples")
    print("\nFiles xuat ra outputs/:")
    print("  cicids2017_balanced.csv")
    print("  class_balance_comparison.png")
    print("  label_classes.json")
    print("=" * 50)

if __name__ == "__main__":
    run_balancing()