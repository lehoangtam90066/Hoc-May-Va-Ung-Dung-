# Dự Án Tính Toán Hàm Mất Mát và Hàm Kích Hoạt trong Machine Learning

Dự án này cung cấp giao diện người dùng để tính toán các hàm mất mát (loss functions) và hàm kích hoạt (activation functions) trong machine learning sử dụng thư viện PyTorch và Streamlit.

## Giới Thiệu

Trong machine learning, các hàm mất mát được sử dụng để đo lường độ chính xác của mô hình, trong khi các hàm kích hoạt giúp mô hình học các đặc điểm phức tạp. Dự án này cho phép người dùng nhập các giá trị đầu vào và mục tiêu, sau đó tính toán và hiển thị kết quả cho nhiều loại hàm mất mát và hàm kích hoạt.

## Thư Viện Cần Thiết

- `torch`
- `streamlit`

## Cài Đặt

Trước tiên, bạn cần cài đặt các thư viện cần thiết. Bạn có thể sử dụng pip để cài đặt:

```bash
pip install torch streamlit
Nhập Giá Trị:

Nhập giá trị đầu vào (inputs) và giá trị mục tiêu (targets) dưới dạng các số thực, cách nhau bằng dấu phẩy.
Nhấn nút "Tính Loss Functions" để tính toán các hàm mất mát:
Mean Square Error (MSE)
Binary Cross-Entropy Loss
Cross Entropy Loss
Tính Toán Các Hàm Kích Hoạt:

Nhấn nút "Tính Activation Functions" để tính toán các hàm kích hoạt:
Sigmoid
ReLU
Softmax
Tanh
Các Hàm Mất Mát
Mean Square Error (MSE): Đo lường độ chính xác của các dự đoán bằng cách tính trung bình bình phương của sự khác biệt giữa giá trị thực và giá trị dự đoán.

Binary Cross-Entropy Loss: Được sử dụng cho các bài toán phân loại nhị phân, đo lường độ chính xác giữa giá trị mục tiêu và giá trị dự đoán.

Cross Entropy Loss: Được sử dụng cho các bài toán phân loại đa lớp, đo lường sự khác biệt giữa phân phối xác suất dự đoán và phân phối xác suất thực tế.

Các Hàm Kích Hoạt
Sigmoid: Hàm kích hoạt phổ biến trong mạng nơ-ron, chuyển đổi đầu vào thành một giá trị giữa 0 và 1.

ReLU (Rectified Linear Unit): Hàm kích hoạt thường được sử dụng trong các mạng nơ-ron sâu, giữ nguyên các giá trị dương và đặt giá trị âm về 0.

Softmax: Chuyển đổi một vector thành một phân phối xác suất, thường được sử dụng trong các mô hình phân loại đa lớp.

Tanh: Hàm kích hoạt cho phép đầu ra nằm giữa -1 và 1, giúp cải thiện độ hội tụ của mạng nơ-ron.

Ví Dụ Kết Quả
Sau khi nhập dữ liệu, bạn sẽ nhận được kết quả cho các hàm mất mát và hàm kích hoạt, ví dụ:

Mean Square Error: 0.123

Binary Entropy Loss: 0.456

Cross Entropy Loss: 0.789

Sigmoid: tensor([0.7311, 0.9933, 0.0180, 0.9526, 0.1192])

ReLU: tensor([1., 5., 0., 3., 0.])

Softmax: tensor([0.0175, 0.7311, 0.0001, 0.2650, 0.0001])

Tanh: tensor([0.7616, 0.9999, -0.9993, 0.9951, -0.9640])




### Lưu ý
- Cập nhật thông tin liên hệ trong phần "Liên Hệ" cho đúng với thông tin của bạn.
- Điều chỉnh phần "Ví Dụ Kết Quả" cho phù hợp với đầu ra thực tế từ ứng dụng của bạn.
