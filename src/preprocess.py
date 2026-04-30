# ==========================================================
# CIC-IDS2017 - EDA & Data Preprocessing
# Theo cây thư mục của bạn:
#
# LAB6/
# ├── data/
# ├── outputs/
# └── src/
#     └── preprocess.py
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# ==========================================================
# 1. PATH THEO THƯ MỤC CỦA BẠN
# preprocess.py nằm trong src/
# nên phải ../data và ../outputs
# ==========================================================

BASE_DIR = os.path.dirname(__file__)              # src/
DATA_PATH = os.path.join(BASE_DIR, "../data/*.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "../outputs")

# tạo outputs nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_preprocessing():
    # ==========================================================
    # 2. LOAD + MERGE CSV
    # ==========================================================

    print("Loading CSV files...")

    files = glob.glob(DATA_PATH)

    if len(files) == 0:
        print("No CSV files found in data/ directory")
        exit()

    df_list = []

    for file in files:
        print("Reading:", os.path.basename(file))
        temp = pd.read_csv(file, low_memory=False)
        df_list.append(temp)

    df = pd.concat(df_list, ignore_index=True)

    print("Merge completed")
    print("Shape:", df.shape)

    # ==========================================================
    # 3. XÓA KHOẢNG TRẮNG TÊN CỘT
    # ==========================================================

    df.columns = df.columns.str.strip()

    # ==========================================================
    # 4. REPLACE INF -> NaN
    # ==========================================================

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ==========================================================
    # 5. FILL NaN = MEDIAN
    # ==========================================================

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    print("Missing values handled")

    # ==========================================================
    # 6. XÓA CỘT ZERO VARIANCE
    # ==========================================================

    zero_var_cols = df.columns[df.nunique() <= 1]

    print("Zero variance columns:", len(zero_var_cols))

    df.drop(columns=zero_var_cols, inplace=True)

    # ==========================================================
    # 7. XÓA DÒNG TRÙNG
    # ==========================================================

    before = df.shape[0]

    df.drop_duplicates(inplace=True)

    after = df.shape[0]

    print("Duplicate rows removed:", before - after)

    # ==========================================================
    # 8. TỐI ƯU BỘ NHỚ
    # ==========================================================

    before_mem = df.memory_usage().sum() / 1024**2

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    after_mem = df.memory_usage().sum() / 1024**2

    print(f"Memory before: {before_mem:.2f} MB")
    print(f"Memory after : {after_mem:.2f} MB")

    # ==========================================================
    # 9. BIỂU ĐỒ ATTACK DISTRIBUTION
    # ==========================================================

    if "Label" in df.columns:

        plt.figure(figsize=(14,6))
        df["Label"].value_counts().plot(kind="bar")
        plt.title("Attack Type Distribution")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(os.path.join(OUTPUT_DIR, "attack_distribution.png"))
        plt.show()

    else:
        print("⚠ Không có cột Label")

    # ==========================================================
    # 10. HEATMAP
    # ==========================================================

    sample_df = df.select_dtypes(include=np.number)
    sample_df = sample_df.iloc[:, :20]

    plt.figure(figsize=(14,10))
    sns.heatmap(sample_df.corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    plt.show()

    # ==========================================================
    # 11. SAVE CLEAN DATA
    # ==========================================================

    df.to_csv(
        os.path.join(OUTPUT_DIR, "cicids2017_cleaned.csv"),
        index=False
    )

    # ==========================================================
    # 12. SUMMARY
    # ==========================================================

    print("=" * 50)
    print("FINAL SHAPE:", df.shape)
    print("Saved files in outputs/")
    print("attack_distribution.png")
    print("correlation_heatmap.png")
    print("cicids2017_cleaned.csv")
    print("=" * 50)

if __name__ == "__main__":
    run_preprocessing()