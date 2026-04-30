# src/deploy_realtime.py
# Phương — Cell 17: Real-time alert (Phần 2.6)

import pandas as pd
import numpy as np
from datetime import datetime
import os

def detect_realtime(flow_data: dict, rf_model, scaler, le, dst_port: int = 80):
    """
    Nhận 1 luồng mạng (dict gồm 18 features), phân loại & sinh alert.
    """
    input_df     = pd.DataFrame([flow_data])
    input_scaled = scaler.transform(input_df)
    pred_encoded = rf_model.predict(input_scaled)[0]
    pred_label   = le.inverse_transform([pred_encoded])[0]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if pred_label != 'BENIGN':
        msg = (f"[{ts}] [ALERT] Suspicious traffic detected: "
               f"{pred_label}. Destination Port: {dst_port}.")
        print(f"\033[91m{msg}\033[0m")   # In màu đỏ
        with open('outputs/alerts.log', 'a') as f:
            f.write(msg + '\n')
    else:
        print(f"[{ts}] [INFO] Traffic BENIGN — OK")
    
    return pred_label


def run_simulation(X_test, selected_features, rf_model, scaler, le, n_samples=10):
    """Mô phỏng real-time với n mẫu từ test set"""
    os.makedirs('outputs', exist_ok=True)
    print("\n🚀 Mô phỏng real-time detection:\n")
    for i in range(n_samples):
        sample   = dict(zip(selected_features, X_test[i]))
        port     = np.random.choice([22, 80, 443, 8080])
        detect_realtime(sample, rf_model, scaler, le, port)