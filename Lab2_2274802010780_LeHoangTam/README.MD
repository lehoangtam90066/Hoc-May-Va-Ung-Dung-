LÊ HOÀNG TÂM 2274802010780
# Ứng dụng Phân Loại Văn Bản với Naive Bayes

Ứng dụng này sử dụng Streamlit để hiển thị và phân loại dữ liệu văn bản trong lĩnh vực giáo dục. Bài toán sử dụng mô hình Naive Bayes để dự đoán nhãn phân loại cho các văn bản.

## Yêu cầu

- Python 3.7 trở lên
- Các thư viện: `streamlit`, `pandas`, `scikit-learn`

## Cài đặt

1. Tải hoặc clone dự án về máy:
   ```bash
   git clone <URL repository>
   cd <tên thư mục dự án>


Mô tả Dữ liệu
Ứng dụng sử dụng dữ liệu từ file Education.csv, bao gồm các cột:

Text: Nội dung văn bản
Label: Nhãn phân loại của văn bản
Ứng dụng sẽ thực hiện mã hóa nhãn và biến văn bản thành các đặc trưng bằng cách sử dụng LabelEncoder và CountVectorizer của thư viện scikit-learn.

Mô tả Mô hình
Có hai mô hình Naive Bayes được áp dụng để phân loại văn bản:

Bernoulli Naive Bayes: Mô hình Naive Bayes với đặc trưng nhị phân.
Multinomial Naive Bayes: Mô hình Naive Bayes với đặc trưng đếm số lần xuất hiện.
Kết quả
Sau khi chạy ứng dụng, Streamlit sẽ hiển thị các kết quả sau:

Dữ liệu ban đầu và dữ liệu sau khi mã hóa nhãn
Độ chính xác của hai mô hình Naive Bayes trên tập kiểm tra

