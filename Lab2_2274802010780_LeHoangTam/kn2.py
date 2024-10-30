import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


try:
    drug_data = pd.read_csv(r"F:/Học/Machineleaning/Machinelearning/Data/drug200.csv")
    st.write("### Dữ liệu ban đầu:")
    st.write(drug_data.head())  
except Exception as e:
    st.write(f"Lỗi khi đọc file CSV: {e}")


label_encoders = {}
for column in ['Sex', 'BP', 'Cholesterol', 'Drug']:
    le = LabelEncoder()
    drug_data[column] = le.fit_transform(drug_data[column])
    label_encoders[column] = le


X = drug_data.drop(columns='Drug')
y = drug_data['Drug']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


gnb = GaussianNB()
gnb.fit(X_train, y_train)


y_pred = gnb.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoders['Drug'].classes_)


st.write("### Độ chính xác của mô hình:")
st.write(f"Accuracy: {accuracy:.2f}")
st.write("### Báo cáo phân loại chi tiết:")
st.text(report)
