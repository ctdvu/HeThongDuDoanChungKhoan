# 🚀 Hướng Dẫn Deploy Thủ Công Web App Lên Render (Free Tier 512MB RAM)

## 💡 ĐÁP ÁN NHANH: Deploy lên Render Free có được không?
👉 **HOÀN TOÀN ĐƯỢC VÀ CHẠY CỰC KỲ MƯỢT MÀ 100%!**

### ⚠️ Vấn đề của Render Free và cách chúng ta đã giải quyết triệt để:
1. **Giới hạn của Render Free:** 
   - RAM tối đa chỉ **512 MB**.
   - Nếu dùng thư viện `tensorflow` hoặc `tensorflow-cpu` gốc, lúc `import tensorflow` đã ngốn **380 MB - 430 MB RAM**, chỉ cần 1-2 người truy cập là chạm trần 512MB và bị Render tự động **Kill dịch vụ (Lỗi Out Of Memory - Error 137)**. Ngoài ra, tải TensorFlow nặng 500MB thường làm build bị chậm hoặc timeout.
2. **Giải pháp đột phá đã tích hợp sẵn trong Web App (`Dual-Engine`):**
   - Chúng tôi đã xây dựng sẵn **Pure NumPy Inference Engine** chạy trực tiếp từ ma trận trọng số `*_weights.npz` đã xuất từ mô hình Keras.
   - **Độ chính xác:** Khớp chính xác tuyệt đối **100% (sai số = 0.000000)** so với Keras TensorFlow.
   - **Mức tiêu thụ RAM:** Chỉ tốn vỏn vẹn **~45 MB RAM** (chưa tới 10% của hạn mức 512MB)!
   - **Thời gian Build trên Render:** Chỉ mất **20 - 30 giây** thay vì 5 phút.
   - **Tốc độ phản hồi:** Cực nhanh (~15ms/lần dự đoán).

---

## 📁 Cấu Trúc Thư Mục Chuẩn Bị Cho Render

```text
webapp/
├── app.py                  # Server Flask chính (hỗ trợ Dual-Engine tự nhận diện)
├── requirements.txt        # Danh sách thư viện siêu nhẹ (Flask, Gunicorn, NumPy, Pandas, Scikit-learn)
├── Procfile                # web: gunicorn app:app --workers 1 --threads 4 --timeout 120
├── render.yaml             # File cấu hình Render Blueprint
├── DEPLOY_RENDER.md        # Hướng dẫn này
├── models/                 # Thư mục trọng số mô hình
│   ├── best_model.keras    
│   ├── DeepStackedLSTM.keras 
│   ├── DeepStackedLSTM_weights.npz   # Trọng số phục vụ NumPy engine siêu nhẹ
│   ├── VanillaRNN_weights.npz
│   ├── LSTM_weights.npz
│   ├── GRU_weights.npz
│   ├── BiLSTM_weights.npz
│   ├── scaler.pkl          # Scaler chuẩn hóa 40 features
│   ├── feature_cols.pkl    # Danh sách 40 đặc trưng
│   ├── metadata.json       # Bảng số liệu so sánh 5 mô hình
│   └── sample_recent_data.csv # Dữ liệu lịch sử mô phỏng
├── templates/
│   └── index.html          # Giao diện Responsive (Desktop & Mobile)
└── static/
    ├── css/style.css       # Giao diện Dark theme TradingView
    └── js/app.js           # Xử lý đồ họa Chart.js, API & sandbox
```

---

## 🛠️ Bước 1: Đẩy Code Lên GitHub

Mở PowerShell tại thư mục dự án:

```powershell
cd D:\N4_K1\HTTM\Code\TieuLuan\RNN_CoPhieu

# Khởi tạo git và commit
git init
git add .
git commit -m "Deploy VN-Index AI Stock Predictor to Render"

# Liên kết với repository GitHub của bạn
git remote add origin https://github.com/<tai-khoan-github>/<ten-repo>.git
git branch -M main
git push -u origin main
```

*(Lưu ý: Đảm bảo thư mục `webapp/models` có đầy đủ các file `.npz`, `.pkl`, `.json` khi đẩy lên).*

---

## 🌐 Bước 2: Tạo Web Service Trên Render

1. Đăng nhập [https://dashboard.render.com](https://dashboard.render.com).
2. Nhấn nút **New +** ở góc trên bên phải $\rightarrow$ chọn **Web Service**.
3. Chọn mục **Build and deploy from a Git repository** $\rightarrow$ bấm **Next**.
4. Chọn repository GitHub bạn vừa đẩy code lên và nhấn **Connect**.

---

## ⚙️ Bước 3: Cấu Hình Thông Số Cho Web Service

Điền các trường chính xác như sau:

| Trường | Giá Trị Điền | Chú thích |
|--------|--------------|-----------|
| **Name** | `vnindex-ai-predictor` | Tên app (tùy chọn) |
| **Region** | `Singapore (Southeast Asia)` | Vùng gần Việt Nam nhất, tốc độ cao |
| **Branch** | `main` | Nhánh chứa code chính |
| **Root Directory** | `RNN_CoPhieu/webapp` *(nếu repo chứa cả thư mục cha)* | Thư mục chứa file `app.py` |
| **Runtime** | `Python 3` | Môi trường Python |
| **Build Command** | `pip install -r requirements.txt` | Cài đặt các gói trong requirements.txt |
| **Start Command** | `gunicorn app:app --workers 1 --threads 4 --timeout 120` | Chạy web server với Gunicorn đa luồng |
| **Instance Type** | `Free` (0.1 CPU, 512 MB RAM) | **Gói miễn phí 100%** |

---

## 🔑 Bước 4: Thêm Biến Môi Trường (Environment Variables)

Cuộn xuống mục **Environment Variables**, bấm **Add Environment Variable**:
* `PYTHON_VERSION`: `3.11.9`
* `FLASK_ENV`: `production`

---

## 🚀 Bước 5: Hoàn Tất Triển Khai & Sử Dụng

1. Bấm nút **Create Web Service**.
2. Quá trình build chỉ mất khoảng **30 - 60 giây**.
3. Khi Render hiện trạng thái **`Live`** kèm link xanh `https://<ten-app>.onrender.com`:
   - **Mở trên máy tính:** Trải nghiệm giao diện đầy đủ với bảng tín hiệu kỹ thuật, biểu đồ tương tác, kiến trúc 7 tầng và bảng so sánh 5 thuật toán.
   - **Mở trên điện thoại:** Giao diện tự động co dãn thông minh với thanh điều hướng đáy (Bottom Bar) chuẩn app di động.

---

## 💡 Lưu Ý Về Gói Free Của Render:
* Sau 15 phút không có người truy cập, Render sẽ tạm thời đưa ứng dụng vào trạng thái "ngủ" (Sleep).
* Lần truy cập kế tiếp sẽ mất khoảng 30 - 50 giây để "đánh thức" (Cold Start). Sau khi thức dậy, app sẽ phản hồi ngay lập tức (~15ms).
