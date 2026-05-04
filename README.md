# Network Intrusion Detection System (NIDS) using Machine Learning

Dự án này triển khai một hệ thống phát hiện xâm nhập mạng (NIDS) sử dụng các thuật toán Học máy (Machine Learning) trên bộ dữ liệu **CICIDS2017**. Hệ thống bao gồm toàn bộ quy trình từ tiền xử lý dữ liệu, cân bằng dữ liệu, huấn luyện nhiều mô hình khác nhau, đánh giá hiệu năng và mô phỏng triển khai thực tế.

## 🚀 Tính năng chính

- **Tiền xử lý dữ liệu**: Làm sạch dữ liệu, xử lý giá trị thiếu (NaN, Infinity) và chuẩn hóa dữ liệu.
- **Cân bằng dữ liệu & Lựa chọn đặc trưng**: Sử dụng kỹ thuật Undersampling để giải quyết vấn đề mất cân bằng lớp và lựa chọn các đặc trưng (features) quan trọng nhất.
- **Huấn luyện đa mô hình**: Triển khai và so sánh hiệu năng của 5 thuật toán phổ biến:
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Logistic Regression
  - Support Vector Machine (LinearSVC)
  - Gaussian Naive Bayes
- **Đánh giá chi tiết**: So sánh các mô hình dựa trên các chỉ số Accuracy, Precision, Recall, F1-Score và vẽ biểu đồ so sánh.
- **Mô phỏng Real-time**: Giả lập việc nhận luồng dữ liệu mạng và đưa ra dự đoán xâm nhập theo thời gian thực.

## 📂 Cấu trúc thư mục

```text
Network-Intrusion-Detection-ML/
├── data/               # Thư mục chứa dataset gốc (CICIDS2017)
├── models/             # Lưu trữ các mô hình đã huấn luyện (.pkl)
├── notebooks/          # Jupyter Notebooks cho mục đích thử nghiệm
├── outputs/            # Kết quả đầu ra (biểu đồ so sánh, file csv đã xử lý, logs)
├── src/                # Mã nguồn Python
│   ├── preprocess.py           # Tiền xử lý dữ liệu thô
│   ├── balance_and_select.py   # Cân bằng dữ liệu & Chọn feature
│   ├── knn_model.py            # Huấn luyện KNN
│   ├── logistic_regression_model.py
│   ├── naive_bayes_model.py
│   ├── random_forest_model_train.py
│   ├── svm_model.py
│   ├── deploy_realtime.py      # Mô phỏng triển khai
│   └── main.py                 # Script chính chạy toàn bộ pipeline
├── requirements.txt    # Danh sách các thư viện cần thiết
└── README.md
```

## 🛠️ Cài đặt và Sử dụng

### 1. Cài đặt môi trường
Yêu cầu Python 3.8+. Nên sử dụng môi trường ảo (virtualenv).

```bash
# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu
Do bộ dữ liệu **CICIDS2017** có kích thước rất lớn, các file `.csv` gốc không được đẩy lên GitHub. 
- Tải bộ dữ liệu CICIDS2017.
- Giải nén và đặt các file CSV vào thư mục `data/`.

### 3. Chạy hệ thống
Bạn có thể chạy toàn bộ quy trình (từ tiền xử lý đến huấn luyện và mô phỏng) chỉ với một lệnh duy nhất:

```bash
python src/main.py
```

*Lưu ý: Trong `src/main.py`, bạn có thể chỉnh `FAST_MODE = True` để chạy thử nghiệm nhanh với một phần dữ liệu nhỏ.*

## 📊 Kết quả
Sau khi chạy, hệ thống sẽ:
1. Lưu các mô hình tốt nhất vào thư mục `models/`.
2. Xuất các biểu đồ so sánh độ chính xác và Confusion Matrix vào thư mục `outputs/`.
3. Hiển thị bảng so sánh hiệu năng giữa các thuật toán trên console.

## 👥 Thành viên thực hiện
- Dự án được thực hiện cho môn học **An toàn Bảo mật Mạng**.
- Nhóm phát triển: [Tên thành viên/Nhóm của bạn]

---
*Dự án này được phát triển với mục đích học tập và nghiên cứu về ứng dụng AI trong an ninh mạng.*
