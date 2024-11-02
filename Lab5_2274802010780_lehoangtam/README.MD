LÊ HOÀNG TÂM 2274802010780
# Dự Án Support Vector Machine (SVM)

Dự án này sử dụng thuật toán Support Vector Machine (SVM) để phân loại dữ liệu bằng cách tối ưu hóa hàm mục tiêu thông qua lập trình bậc hai (quadratic programming). Thư viện CVXOPT được sử dụng để giải quyết bài toán tối ưu này.

## Giới Thiệu

Dự án gồm hai phần:
1. **Hard Margin SVM**: Phân loại dữ liệu mà không có sự cho phép cho các điểm dữ liệu nằm trong ranh giới.
2. **Soft Margin SVM**: Phân loại dữ liệu cho phép một số điểm dữ liệu nằm trong ranh giới với chi phí cho sự vi phạm này.

## Thư Viện Cần Thiết

- `numpy`
- `cvxopt`
- `matplotlib`

## Cài Đặt

Trước tiên, bạn cần cài đặt các thư viện cần thiết. Bạn có thể sử dụng pip để cài đặt:

```bash
pip install numpy cvxopt matplotlib


Sử Dụng
Chạy Code: Bạn có thể chạy đoạn mã Python dưới đây để thấy quá trình thực hiện SVM cho cả hard margin và soft margin.

Kết Quả: Mã sẽ xuất ra các thông số sau:

λ (Lagrange multipliers)
w (vector trọng số)
b (bias term)
Bên cạnh đó, bạn sẽ thấy đồ thị hiển thị các điểm dữ liệu, đường phân cách (decision boundary), cũng như các ranh giới cho cả hard margin và soft margin.

Kết Quả
Hard Margin SVM: Đường phân cách được xây dựng dựa trên các điểm dữ liệu với không có vi phạm cho các lớp.
Soft Margin SVM: Cho phép một số điểm dữ liệu nằm trong ranh giới với chi phí được tính toán từ các biến slack (ξ).
Đồ Thị
Đoạn mã sẽ tạo ra đồ thị thể hiện:

Các điểm dữ liệu với màu sắc khác nhau cho từng lớp.
Đường phân cách (decision boundary) giữa các lớp.
Ranh giới dương và âm (positive và negative boundary).
Giá trị của các biến slack cho từng điểm dữ liệu trong phần soft margin.

lambda = [0.5, 0.5, 0.0]
w = [1.0, 2.0]
b = -1.0


lambda = [0.0, 0.8, 0.5, ...]
w = [0.7, 1.5]
b = -0.3



### Lưu ý
- Thay `# Code SVM của bạn ở đây` bằng mã code thực tế của bạn.
- Điều chỉnh phần "Ví Dụ Kết Quả" để phù hợp với kết quả thực tế mà mã của bạn tạo ra.
- Cập nhật thông tin liên hệ trong phần "Liên Hệ" cho đúng với thông tin của bạn.
