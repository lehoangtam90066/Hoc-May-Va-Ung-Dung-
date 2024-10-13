import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Hàm để huấn luyện mô hình
def train_models(data):
    # Biến đổi dữ liệu định tính sang định lượng
    data['Sex'] = data['Sex'].replace({'M': 0, 'F': 1})
    data['BP'] = data['BP'].replace({'HIGH': 2, 'NORMAL': 1, 'LOW': 0})
    data['Cholesterol'] = data['Cholesterol'].replace({'HIGH': 1, 'NORMAL': 0})
    data['Drug'] = data['Drug'].replace({'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'DrugY': 4})

    # Chuyển đổi sang kiểu int
    data['Sex'] = data['Sex'].astype(int)
    data['BP'] = data['BP'].astype(int)
    data['Cholesterol'] = data['Cholesterol'].astype(int)
    data['Drug'] = data['Drug'].astype(int)

    # Tạo tập X và Y
    X = data.drop('Drug', axis=1)  # Bỏ cột Drug để tạo tập X
    Y = data['Drug']  # Cột Drug làm nhãn

    # Tạo dữ liệu train test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)

    random_forest = RandomForestClassifier(n_estimators=3)
    random_forest.fit(X_train, y_train)

    return decision_tree, random_forest, X_test, y_test

# Tải dữ liệu
data = pd.read_csv("F:/Học/Học máy và ứng dụng/Machinelearning/LeHoangTam_2274802010780_Buoi4/Lab4/drug200.csv")

# Huấn luyện mô hình
decision_tree, random_forest, X_test, y_test = train_models(data)

# Giao diện Streamlit
st.title("Dự đoán Thuốc")
st.write("Nhập thông tin của bệnh nhân:")

# Nhập thông tin từ người dùng
sex = st.selectbox("Giới tính:", ("Nam", "Nữ"))
bp = st.selectbox("Huyết áp:", ("HIGH", "NORMAL", "LOW"))
cholesterol = st.selectbox("Cholesterol:", ("HIGH", "NORMAL"))
age = st.number_input("Tuổi:", min_value=0, max_value=120, value=30)

# Chuyển đổi đầu vào thành định dạng mà mô hình yêu cầu
input_data = {
    'Age': age,
    'Sex': 1 if sex == 'Nữ' else 0,
    'BP': 2 if bp == 'HIGH' else (1 if bp == 'NORMAL' else 0),
    'Cholesterol': 1 if cholesterol == 'HIGH' else 0
}

# Tạo DataFrame từ input_data
input_df = pd.DataFrame([input_data])

# Đảm bảo các cột trong input_df khớp với X_train
for column in X_test.columns:
    if column not in input_df.columns:
        input_df[column] = 0  # Thay 0 bằng giá trị mặc định thích hợp

# Đảm bảo thứ tự cột
input_df = input_df[X_test.columns]

# Dự đoán với Decision Tree
if st.button("Dự đoán với Decision Tree"):
    prediction = decision_tree.predict(input_df)
    st.write("Dự đoán Drug: ", prediction[0])

# Dự đoán với Random Forest
if st.button("Dự đoán với Random Forest"):
    prediction = random_forest.predict(input_df)
    st.write("Dự đoán Drug: ", prediction[0])
