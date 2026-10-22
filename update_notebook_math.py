# -*- coding: utf-8 -*-
import nbformat

nb_path = r"D:\N4_K1\HTTM\Code\TieuLuan\RNN_CoPhieu\RNN_StockPrediction.ipynb"
nb = nbformat.read(nb_path, as_version=4)

math_md = r"""### 📐 GIẢI THÍCH BẰNG CÁCH THỂ HIỆN HỢP CÁC HÀM (FUNCTION COMPOSITION)

Mô hình Deep Stacked RNN (>5 layers) về bản chất toán học là một **hợp các ánh xạ hàm liên tiếp (Composite Function)**:

$$\hat{y} = (f_7 \circ f_6 \circ f_5 \circ f_4 \circ f_3 \circ f_2 \circ f_1)(X)$$

Trong đó $X \in \mathbb{R}^{B \times 30 \times 40}$ là tensor dữ liệu 30 phiên gần nhất với 40 đặc trưng. Dữ liệu được truyền tuần tự qua từng hàm biến đổi:

$$X \xrightarrow{f_1} H^{(1)} \xrightarrow{f_2} H^{(2)} \xrightarrow{f_3} H^{(3)} \xrightarrow{f_4} h^{(4)} \xrightarrow{f_5} z^{(5)} \xrightarrow{f_6} z^{(6)} \xrightarrow{f_7} \hat{y}$$

---

### 🔍 GIẢI THÍCH CHI TIẾT TỪNG HÀM THÀNH PHẦN:

#### 1. Hàm $f_1$: Tầng LSTM Thứ Nhất (256 units, return_sequences=True)
- **Công thức:** $H^{(1)}_t, C^{(1)}_t = \text{LSTM}_1(X_t, H^{(1)}_{t-1}, C^{(1)}_{t-1})$
- **Nhiệm vụ:** Tiếp nhận tensor đầu vào $(30, 40)$, trích xuất các phụ thuộc chuỗi thời gian ngắn hạn và chuyển tiếp toàn bộ chuỗi trạng thái ẩn $H^{(1)} \in \mathbb{R}^{30 \times 256}$ lên tầng tiếp theo. Tích hợp chuẩn hóa $L_2$ ($10^{-4}$) để chống bùng nổ trọng số.

#### 2. Hàm $f_2$: Tầng LSTM Thứ Hai (128 units, return_sequences=True)
- **Công thức:** $H^{(2)}_t, C^{(2)}_t = \text{LSTM}_2(H^{(1)}_t, H^{(2)}_{t-1}, C^{(2)}_{t-1})$
- **Nhiệm vụ:** Nhận chuỗi biểu diễn cấp 1 từ $f_1$, áp dụng Batch Normalization để ổn định phân phối gradient, và Dropout ($0.30$) để ngắt ngẫu nhiên các liên kết thừa. Output giữ nguyên chiều thời gian: $H^{(2)} \in \mathbb{R}^{30 \times 128}$.

#### 3. Hàm $f_3$: Tầng LSTM Thứ Ba (64 units, return_sequences=True)
- **Công thức:** $H^{(3)}_t, C^{(3)}_t = \text{LSTM}_3(H^{(2)}_t, H^{(3)}_{t-1}, C^{(3)}_{t-1})$
- **Nhiệm vụ:** Học các mô thức biến động trung hạn và phân kỳ chỉ báo. Không gian đặc trưng tiếp tục được cô đọng: $H^{(3)} \in \mathbb{R}^{30 \times 64}$.

#### 4. Hàm $f_4$: Tầng LSTM Thứ Tư (32 units, return_sequences=False)
- **Công thức:** $h^{(4)} = \text{LSTM}_4(H^{(3)}_T, h^{(4)}_{T-1}, C^{(4)}_{T-1})$
- **Nhiệm vụ:** Bước chuyển tiếp then chốt từ miền chuỗi thời gian sang miền vector đặc trưng tĩnh. Chỉ trả về hidden vector tại bước thời gian cuối cùng $T=30$, tổng hợp toàn bộ ngữ cảnh quá khứ thành $h^{(4)} \in \mathbb{R}^{32}$.

#### 5. Hàm $f_5$: Tầng Kết Nối Dày Đặc Dense 1 (64 units + ReLU)
- **Công thức:** $z^{(5)} = \text{ReLU}(W_5 \cdot h^{(4)} + b_5) = \max(0, W_5 h^{(4)} + b_5)$
- **Nhiệm vụ:** Chiếu vector ẩn vào không gian phi tuyến bậc cao hơn để giải mã các tương quan phức tạp giữa các đặc trưng xu hướng và khối lượng. Đi kèm Batch Normalization và Dropout ($0.20$).

#### 6. Hàm $f_6$: Tầng Kết Nối Dày Đặc Dense 2 (32 units + ReLU)
- **Công thức:** $z^{(6)} = \text{ReLU}(W_6 \cdot z^{(5)} + b_6)$
- **Nhiệm vụ:** Nén không gian quyết định từ 64 chiều về 32 chiều, tập trung vào ranh giới phân tách giữa 2 lớp TĂNG và GIẢM.

#### 7. Hàm $f_7$: Tầng Phân Lớp Đầu Ra (1 unit + Sigmoid)
- **Công thức:** $\hat{y} = \sigma(W_7 \cdot z^{(6)} + b_7) = \frac{1}{1 + e^{-(W_7 z^{(6)} + b_7)}}$
- **Nhiệm vụ:** Ánh xạ tổng tuyến tính cuối cùng về khoảng xác suất $\hat{y} \in (0, 1)$. Nếu $\hat{y} \ge 0.5 \implies$ Dự báo giá ngày mai **TĂNG**; ngược lại là **GIẢM**.

---

### ⚙️ CHI TIẾT CÁC HÀM CỔNG NỘI TẠI TRONG TỪNG LSTM CELL:
Mỗi hàm $f_1, f_2, f_3, f_4$ ở trên đều chứa 6 phép biến đổi bên trong cell tại mỗi thời điểm $t$:
1. **Forget Gate:** $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ — Quyết định phần trăm thông tin cũ bị lãng quên.
2. **Input Gate:** $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ — Xác định mức độ cập nhật thông tin mới.
3. **Candidate Memory:** $\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$ — Tạo trạng thái ứng viên tiềm năng.
4. **Cell State Update:** $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ — Cập nhật dòng chảy bộ nhớ dài hạn.
5. **Output Gate:** $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ — Lọc thông tin xuất ra ngoài.
6. **Hidden State:** $h_t = o_t \odot \tanh(C_t)$ — Trạng thái ẩn ngắn hạn truyền tiếp cho bước sau.

---

### 📊 HỢP HÀM TIỀN XỬ LÝ ĐẶC TRƯNG (FEATURE PIPELINE):
Tương tự kiến trúc mạng, toàn bộ chuỗi tiền xử lý dữ liệu được biểu diễn toán học dưới dạng hợp các hàm biến đổi dữ liệu:

$$\mathcal{D}_{\text{final}} = (g_{\text{target}} \circ g_{\text{macd}} \circ g_{\text{vol}} \circ g_{\text{vola}} \circ g_{\text{mom}} \circ g_{\text{ma}} \circ g_{\text{ret}})(\mathcal{D}_{\text{raw}})$$

- $g_{\text{ret}}$: Tính tỉ suất sinh lời ngày (Daily Return) và Log Return.
- $g_{\text{ma}}$: Tính các đường trung bình động ngắn và dài hạn (SMA, EMA 5, 10, 20, 50).
- $g_{\text{mom}}$: Tính các chỉ báo xung lượng (RSI 14, Stochastic %K/%D, Williams %R, CCI, ROC).
- $g_{\text{vola}}$: Đo lường biên độ dao động (Bollinger Bands, ATR 14, Historical Volatility 20).
- $g_{\text{vol}}$: Chuẩn hóa khối lượng (Volume log, Volume ratio, OBV, MFI 14).
- $g_{\text{macd}}$: Tính toán MACD, MACD Signal và Histogram phân kỳ.
- $g_{\text{target}}$: Gán nhãn phân loại nhị phân ($y_t = 1$ nếu $P_{t+1} > P_t$, ngược lại $0$).
"""

# Insert right after the section introducing the models
inserted = False
for i, cell in enumerate(nb.cells):
    if 'Kiến Trúc Mô Hình Chính' in cell.source or 'Deep Stacked LSTM (5 layer)' in cell.source:
        nb.cells.insert(i + 1, nbformat.v4.new_markdown_cell(math_md))
        inserted = True
        break

if not inserted:
    nb.cells.insert(12, nbformat.v4.new_markdown_cell(math_md))

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Successfully injected mathematical function composition into notebook!")
