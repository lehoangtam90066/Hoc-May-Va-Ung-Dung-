LÊ HOÀNG TÂM 2274802010780
# Ứng Dụng Dự Đoán Loại Thuốc Bằng Học Máy

Ứng dụng này là một dự án sử dụng các mô hình học máy để dự đoán loại thuốc phù hợp cho bệnh nhân dựa trên các đặc điểm như giới tính, tuổi, huyết áp, và mức cholesterol. Các mô hình được triển khai bao gồm Decision Tree và Random Forest, được tích hợp trên giao diện Streamlit để dễ dàng tương tác.

## Giới Thiệu

Dự án này sử dụng dữ liệu về bệnh nhân để huấn luyện các mô hình Decision Tree và Random Forest, cho phép dự đoán loại thuốc phù hợp cho bệnh nhân dựa trên các yếu tố đầu vào. Ứng dụng cho phép người dùng nhập thông tin trực tiếp và đưa ra dự đoán với từng mô hình.

## Các Mô Hình Học Máy

- **Decision Tree**: Mô hình cây quyết định giúp phân loại các loại thuốc dựa trên các điều kiện cụ thể từ thông tin bệnh nhân.
- **Random Forest**: Là một tập hợp của nhiều cây quyết định, giúp tăng độ chính xác và độ ổn định của mô hình dự đoán.

## Dữ Liệu

Dữ liệu được sử dụng là tập `drug200.csv`, bao gồm các thông tin về:
- **Giới tính (Sex)**: Nam hoặc Nữ
- **Tuổi (Age)**: Tuổi của bệnh nhân
- **Huyết áp (BP)**: Mức huyết áp (Cao, Bình thường, Thấp)
- **Mức Cholesterol (Cholesterol)**: Cao hoặc Bình thường
- **Loại Thuốc (Drug)**: Loại thuốc được chỉ định (drugA, drugB, drugC, drugX, DrugY)

Các cột định tính đã được mã hóa thành các giá trị số để phù hợp với các mô hình học máy.

## Cài Đặt

1. **Clone dự án**:
   ```bash
   git clone <URL của repository>
   cd <tên thư mục dự án>

Tiền xử lý dữ liệu: Mã hóa các biến định tính thành số.
Tạo tập huấn luyện và kiểm tra: Sử dụng train_test_split để chia dữ liệu thành 80% huấn luyện và 20% kiểm tra.
Huấn luyện mô hình: Sử dụng DecisionTreeClassifier và RandomForestClassifier để huấn luyện mô hình trên dữ liệu huấn luyện.
Dự đoán và tính toán độ chính xác: Tính toán độ chính xác của mô hình trên tập kiểm tra và đưa ra dự đoán từ các đầu vào người dùng.

