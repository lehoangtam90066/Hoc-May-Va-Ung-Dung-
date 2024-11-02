LÊ HOÀNG TÂM 2274802010780
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
Dữ liệu bao gồm chiều cao (X) và cân nặng (y) của một số sinh viên, được thể hiện dưới dạng

X = np.array([180, 162, 183, 174, 160, 163, 180, 165, 175, 170, 170, 169,
              168, 175, 169, 171, 155, 158, 175, 165]).reshape(-1,1)
y = np.array([86, 55, 86.5, 70, 62, 54, 60, 72, 93, 89, 60, 82, 59, 75,
              56, 89, 45, 60, 60, 72]).reshape((-1,1))
Thêm hằng số (bias) vào X

X = np.insert(X, 0, 1, axis=1)
Khởi tạo hệ số theta cho mô hình hồi quy:

theta = np.array([0, 0.5])
x1 = 150
y1 = theta[0] + theta[1] * x1  
x2 = 190
y2 = theta[0] + theta[1] * x2  
Tạo Đồ Thị
Dữ liệu được hiển thị với đường hồi quy và các điểm dữ liệu sinh viên:
fig, ax = plt.subplots()
ax.plot([x1, x2], [y1, y2], 'r-', label='Đường hồi quy (góc gần 45 độ)')
ax.plot(X[:,1], y[:,0], 'bo', label='Dữ liệu sinh viên')
ax.set_xlabel('Chiều cao (cm)')
ax.set_ylabel('Cân nặng (kg)')
ax.set_title('Chiều cao và cân nặng của sinh viên VLU')
ax.legend()
Hiển Thị Đồ Thị Trên Streamlit


