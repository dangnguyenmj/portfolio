# Dự Án Dự Đoán Giá Cổ Phiếu

## Tổng Quan
Dự án này tập trung vào việc dự đoán giá cổ phiếu bằng cách sử dụng các mô hình học máy. Ứng dụng cho phép người dùng phân tích dữ liệu cổ phiếu, trực quan hóa xu hướng và đưa ra dự đoán bằng các mô hình khác nhau.

## Tính Năng
1. **Lựa Chọn Mô Hình**: Chọn từ nhiều mô hình học máy như Logistic Regression, RNN, SARIMAX.
2. **Phân Tích Dữ Liệu Cổ Phiếu**: Hiển thị dữ liệu cổ phiếu lịch sử, bao gồm giá Mở (Open), Đóng (Close), Cao (High), và Thấp (Low).
3. **Trực Quan Hóa Dữ Liệu**:
   - Xu hướng giá cổ phiếu (Giá đóng theo thời gian).
   - Các chỉ báo kỹ thuật: MACD (Moving Average Convergence Divergence), RSI, EMA, SMA: 20 & 50.
4. **Trực Quan Hóa Dự Đoán**: Xem biểu đồ dự đoán giá cổ phiếu cùng với giá thực tế. Các chỉ số kỹ thuật: R2, MAPE, RMSE.

## Cài Đặt
1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
2. Run All các file .ipynb trong  notebooks 

3. Khởi chạy ứng dụng Flask:
   ```bash
   python app.run
   ```

## Sử Dụng
1. Mở ứng dụng trên trình duyệt (mặc định: [http://127.0.0.1:5000](http://127.0.0.1:5000)).
2. Nhập mã cổ phiếu, ngày bắt đầu và kết thúc, số ngày dữ liệu cho dự đoán.
4. Phân tích dữ liệu cổ phiếu lịch sử và biểu đồ trực quan.
5. Xem biểu đồ dự đoán và so sánh giá dự đoán với giá thực tế.  