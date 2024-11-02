# Lê Hoàng Tâm 2274802010780
# Ứng Dụng Phân Loại Văn Bản Bằng KNN với Cosine Similarity

Ứng dụng này thực hiện phân loại văn bản dựa trên thuật toán K-Nearest Neighbors (KNN) kết hợp với phép đo khoảng cách Cosine. Người dùng có thể tải tệp CSV chứa văn bản, xử lý dữ liệu, chia tập train/test và kiểm tra kết quả phân loại.

## Tính Năng

- **Đọc Dữ Liệu CSV**: Hàm `loadCsv()` cho phép đọc dữ liệu từ tệp CSV.
- **Chia Tập Train/Test**: Hàm `splitTrainTest()` giúp chia dữ liệu thành tập huấn luyện và kiểm tra theo tỷ lệ chỉ định.
- **Lấy Tần Số Từ**: Hàm `get_words_frequency()` tạo ma trận tần số từ dựa trên túi từ.
- **Chuyển Đổi Dữ Liệu Test**: Hàm `transform()` chuyển đổi dữ liệu test sang dạng ma trận tần số từ để tính khoảng cách.
- **Khoảng Cách Cosine**: Hàm `cosine_distance()` tính khoảng cách Cosine giữa các vector văn bản trong tập huấn luyện và kiểm tra.
- **Thuật Toán KNN**: Lớp `KNNText` triển khai thuật toán KNN để phân loại các văn bản dựa trên khoảng cách Cosine.
- **Đánh Giá Độ Chính Xác**: Hàm `score()` tính toán độ chính xác của mô hình trên tập kiểm tra.


