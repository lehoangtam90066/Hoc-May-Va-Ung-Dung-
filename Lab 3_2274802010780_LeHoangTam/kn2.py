# import numpy as np
# import pandas as pd
# import sys
# sys.stdout.reconfigure(encoding='utf-8')

# # Hàm load dữ liệu từ file CSV
# def loadCsv(filename) -> pd.DataFrame:
#     df = pd.read_csv(filename)  # Đọc file CSV
#     return df

# # Hàm biến đổi cột định tính thành one-hot encoding
# def transform(data, columns_trans):  # columns_trans là danh sách các cột cần biến đổi
#     for i in columns_trans:
#         unique = data[i].unique() + '-' + i  # Tạo unique cho cột
#         matrix_0 = np.zeros((len(data), len(unique)), dtype=int)  # Tạo ma trận 0
#         frame_0 = pd.DataFrame(matrix_0, columns=unique)  # DataFrame cho one-hot
#         for index, value in enumerate(data[i]):
#             frame_0.at[index, value + '-' + i] = 1
#         data[unique] = frame_0  # Gắn vào data gốc
#     return data

# # Hàm scale dữ liệu về [0, 1] (min-max scaler)
# def scale_data(data, columns_scale):  # columns_scale là danh sách các cột cần scale
#     for i in columns_scale:
#         _max = data[i].max()
#         _min = data[i].min()
#         min_max_scaller = lambda x: round((x - _min) / (_max - _min), 3)  # Hàm scale
#         data[i] = data[i].apply(min_max_scaller)
#     return data

# # Hàm tính khoảng cách Cosine
# def cosine_distance(train_X, test_X):
#     dict_distance = dict()
#     for index, value in enumerate(test_X, start=1):
#         for j in train_X:
#             result = np.sqrt(np.sum((j - value)**2))  # Tính khoảng cách Euclidean
#             if index not in dict_distance:
#                 dict_distance[index] = [result]
#             else:
#                 dict_distance[index].append(result)
#     return dict_distance

# # Hàm dự đoán kết quả theo KNN
# def pred_test(k, train_X, test_X, train_y):
#     lst_predict = list()
#     dict_distance = cosine_distance(train_X, test_X)
#     train_y = train_y.to_frame(name='target').reset_index(drop=True)  # Chuyển train_y thành DataFrame
#     frame_concat = pd.concat([pd.DataFrame(dict_distance), train_y], axis=1)
#     for i in range(1, len(dict_distance) + 1):
#         sort_distance = frame_concat[[i, 'target']].sort_values(by=i, ascending=True)[:k]  # Lấy K khoảng cách nhỏ nhất
#         target_predict = sort_distance['target'].value_counts(ascending=False).index[0]
#         lst_predict.append([i, target_predict])
#     return lst_predict

# # Load dữ liệu từ file CSV
# data = loadCsv('F:\Học\Học máy và ứng dụng\Machinelearning\LeHoangTam_2274802010780_Buoi3\Lab 3_2274802010780_LeHoangTam\drug200.csv')

# print(data.head())  # Kiểm tra dữ liệu

# # Biến đổi các cột định tính
# df = transform(data, ['Sex', 'BP', 'Cholesterol']).drop(['Sex', 'BP', 'Cholesterol'], axis=1)
# print(df)

# # Scale các cột số về khoảng [0,1]
# scale_data(df, ['Age', 'Na_to_K'])  # Hàm trả về data đã scale
# print(df)

# # Tạo data_X và data_y
# data_X = df.drop(['Drug'], axis=1).values
# data_y = df['Drug']
# print(data_X)
# print(data_y)

# # Chia dữ liệu thành tập train và test
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=0)

# print(len(X_train), len(X_test), len(y_train), len(y_test))  # Kiểm tra kích thước các tập dữ liệu

# # Dự đoán với KNN
# test_pred = pred_test(6, X_train, X_test, y_train)

# # Chuyển kết quả thành DataFrame để so sánh với giá trị thực tế
# df_test_pred = pd.DataFrame(test_pred).drop([0], axis=1)
# df_test_pred.index = range(1, len(test_pred) + 1)
# df_test_pred.columns = ['Predict']
# df_actual = pd.DataFrame(y_test)
# df_actual.index = range(1, len(y_test) + 1)
# df_actual.columns = ['Actual']

# # Kết hợp kết quả dự đoán và thực tế
# result = pd.concat([df_test_pred, df_actual], axis=1)
# print(result)


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Hàm load data
def loadCsv(filename) -> pd.DataFrame:
    df = pd.read_csv(filename)
    return df

# Hàm biến đổi định tính (One-hot encoding)
def transform(data, columns_trans): 
    for i in columns_trans:
        unique = data[i].unique() + '-' + i
        matrix_0 = np.zeros((len(data), len(unique)), dtype = int)
        frame_0 = pd.DataFrame(matrix_0, columns = unique)
        for index, value in enumerate(data[i]):
            frame_0.at[index, value + '-' + i] = 1
        data[unique] = frame_0
    return data

# Hàm scale dữ liệu (Min-Max Scaling)
def scale_data(data, columns_scale): 
    for i in columns_scale:  
        _max = data[i].max()
        _min = data[i].min()
        min_max_scaler = lambda x: round((x - _min) / (_max - _min), 3) if _max != _min else 0
        data[i] = data[i].apply(min_max_scaler)
    return data

# Hàm tính khoảng cách Cosine
def cosine_distance(train_X, test_X): 
    dict_distance = dict()
    for index, value in enumerate(test_X, start = 1):
        for j in train_X:
            result = np.sqrt(np.sum((j - value)**2))
            if index not in dict_distance:
                dict_distance[index] = [result]
            else:
                dict_distance[index].append(result)
    return dict_distance

# Hàm dự đoán dựa trên k khoảng cách gần nhất
def pred_test(k, train_X, test_X, train_y):
    lst_predict = list()
    dict_distance = cosine_distance(train_X, test_X)
    train_y = train_y.to_frame(name='target').reset_index(drop=True)
    frame_concat = pd.concat([pd.DataFrame(dict_distance), train_y], axis=1)
    for i in range(1, len(dict_distance) + 1):
        sort_distance = frame_concat[[i, 'target']].sort_values(by=i, ascending=True)[:k]
        target_predict = sort_distance['target'].value_counts(ascending=False).index[0]
        lst_predict.append([i, target_predict])
    return lst_predict

# Đọc dữ liệu
st.title('Phân loại thuốc bằng KNN')

uploaded_file = st.file_uploader("Tải lên file CSV", type="csv")
if uploaded_file is not None:
    # Load dữ liệu
    data = loadCsv(uploaded_file)
    
    # Xử lý dữ liệu - loại bỏ dấu câu khỏi cột Text (nếu có)
    if 'Text' in data.columns:
        data['Text'] = data['Text'].apply(lambda x: x.replace(',', '').replace('.', ''))
            
    st.write("Dữ liệu đã tải:")
    st.dataframe(data)

    # Hiển thị dữ liệu ban đầu
    st.write("Mẫu dữ liệu:", data.head())

    # Biến đổi dữ liệu
    df = transform(data, ['Sex', 'BP', 'Cholesterol']).drop(['Sex', 'BP', 'Cholesterol'], axis=1)
    scale_data(df, ['Age', 'Na_to_K'])

    # Hiển thị dữ liệu đã được biến đổi
    st.write("Dữ liệu sau khi biến đổi:", df.head())

    # Tạo data_X và target
    data_X = df.drop(['Drug'], axis=1).values
    data_y = df['Drug']

    # Chia dữ liệu thành train và test
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=0)

    # Số k gần nhất
    k = st.slider('Chọn giá trị K cho KNN', 1, 10, 6)

    # Dự đoán
    test_pred = pred_test(k, X_train, X_test, y_train)
    df_test_pred = pd.DataFrame(test_pred).drop([0], axis=1)
    df_test_pred.index = range(1, len(test_pred) + 1)
    df_test_pred.columns = ['Dự đoán']

    # Thực tế
    df_actual = pd.DataFrame(y_test)
    df_actual.index = range(1, len(y_test) + 1)
    df_actual.columns = ['Thực tế']

    # Kết quả dự đoán so với thực tế
    result_df = pd.concat([df_test_pred, df_actual], axis=1)

    # Hiển thị kết quả
    st.write("Kết quả Dự đoán so với Thực tế:", result_df)
else:
    st.write("Vui lòng tải lên một tệp CSV để tiếp tục.")