🛡️ Network Intrusion Detection System (IDS) using Machine Learning

An toàn & Bảo mật HTTT — Lab 6 | Nhóm 5 người | CIC-IDS2017 Dataset

📖 Project Introduction
Dự án xây dựng một Hệ thống Phát hiện Xâm nhập Mạng (NIDS) thời gian thực sử dụng Machine Learning. Hệ thống được huấn luyện trên bộ dữ liệu CIC-IDS2017 (Canadian Institute for Cybersecurity) — bao gồm lưu lượng mạng bình thường (Benign) và nhiều loại tấn công (DoS, DDoS, PortScan, Web Attack, Infiltration...).
Pipeline tổng quan:
[Người 1] EDA & Preprocessing  →  [Người 2] Balancing + Feature Selection
→  [Người 3] LR + SVM  ─┐
                          ├→  [So sánh 5 model]  →  [Người 4] RF Deploy → alerts.log
→  [Người 4] NB+KNN+RF ─┘
Người 5 thực hiện bài tập Standard ACL (Cisco Packet Tracer) độc lập — xem thư mục acl/.

📁 Project Structure
Network-Intrusion-Detection-ML/
│
├── data/                        # Thư mục chứa dataset (KHÔNG commit lên GitHub)
│   ├── Monday-WorkingHours.pcap_ISCX.csv
│   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   ├── Wednesday-WorkingHours.pcap_ISCX.csv
│   └── ... (8 CSV files ~1GB total)
│
├── notebooks/
│   └── main_notebook.ipynb      # Notebook chính của cả nhóm (chạy từ trên xuống)
│
├── src/                         # (Tuỳ chọn) tách code ra .py
│   ├── preprocessing.py         # Người 1 + 2
│   ├── models.py                # Người 3 + 4
│   └── realtime_alert.py        # Người 4
│
├── models/
│   └── rf_model.pkl             # Trained Random Forest model (dùng joblib)
│                                # Nếu > 100MB → cung cấp Google Drive link bên dưới
│
├── outputs/
│   ├── alerts.log               # Log cảnh báo thời gian thực (optional)
│   ├── confusion_matrix_lr.png
│   ├── confusion_matrix_svm.png
│   ├── confusion_matrix_nb.png
│   ├── confusion_matrix_knn.png
│   └── confusion_matrix_rf.png
│
├── acl/                         # Người 5 — Bài tập ACL riêng biệt
│   ├── Configuring_Standard_ACLs.pka   # File Packet Tracer
│   └── acl_report.md            # Báo cáo cấu hình ACL
│
├── requirements.txt             # Danh sách thư viện
├── .gitignore                   # Bỏ qua data/ và file lớn
└── README.md                    # File này

⚙️ Installation Guide
1. Clone repository
bashgit clone https://github.com/<your-username>/Network-Intrusion-Detection-ML.git
cd Network-Intrusion-Detection-ML
2. Cài đặt thư viện
bashpip install -r requirements.txt
3. Tải dataset
Tải 8 file CSV từ Kaggle và đặt vào thư mục data/:
🔗 https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset
bashmkdir data
# Sau khi tải về, giải nén và copy 8 file .csv vào thư mục data/
4. Chạy notebook
bashjupyter notebook notebooks/main_notebook.ipynb

Lưu ý: Chạy từ Cell 1 đến cuối, theo thứ tự. Đảm bảo file CSV đã được đặt đúng vào data/.


📊 Summary of Results
Model Comparison Table
ModelAccuracyPrecisionRecallF1-ScoreTraining TimeLogistic Regression-----Support Vector Machine (SVM)-----Naive Bayes-----K-Nearest Neighbors (KNN)-----Random Forest ✅-----

⚠️ Lưu ý quan trọng: Trong bài toán cybersecurity, Recall (tỷ lệ phát hiện tấn công thực) quan trọng hơn Precision — bỏ sót tấn công thật nguy hiểm hơn báo động giả.

Best Model: Random Forest — được chọn để deploy vì đạt Recall và F1-score cao nhất trên tập test.
Saved Model
Model được lưu tại models/rf_model.pkl bằng joblib.
<!-- Nếu file > 100MB, uncomment dòng dưới và thêm link Google Drive: -->
<!-- 📦 **Download model:** [Google Drive Link](...) -->

🚨 Real-time Alert System
Khi một network flow được phân loại là không phải BENIGN, hệ thống tạo cảnh báo dạng Suricata:
[ALERT] 2025-01-01 10:00:00 | Suspicious traffic detected: DDoS | Destination Port: 80 | Severity: HIGH
[ALERT] 2025-01-01 10:00:01 | Suspicious traffic detected: PortScan | Destination Port: 22 | Severity: MEDIUM
Log được ghi vào file outputs/alerts.log.

👥 Team Contributions
Thành viênPhụ tráchSectionNgười 1EDA & Data Preprocessing2.1, 2.2Người 2Class Balancing & Feature Selection2.3, 2.4Người 3Logistic Regression + SVM2.5 (một phần)Người 4Naive Bayes + KNN + Random Forest + Deploy2.5, 2.6Người 5Standard ACL — Cisco Packet Traceracl/

📚 References

CIC-IDS2017 Dataset on Kaggle
Reference Implementation
Intrusion Detection System Notebook
imbalanced-learn Documentation
Scikit-learn Documentation