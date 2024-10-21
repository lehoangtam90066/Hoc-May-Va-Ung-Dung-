import torch
import torch.nn.functional as F
import streamlit as st

# Công thức tính CrossEntropy Loss
def crossEntropyLoss(output, target):
    return F.cross_entropy(output.unsqueeze(0), target.unsqueeze(0))

# Công thức tính Mean Square Error
def meanSquareError(output, target):
    return torch.mean((output - target) ** 2)

# Công thức tính Binary Cross-Entropy Loss
def binaryEntropyLoss(output, target, n):
    return F.binary_cross_entropy(output, target, reduction='sum') / n

# Công thức hàm sigmoid
def sigmoid(x: torch.tensor):
    return 1 / (1 + torch.exp(-x))

# Công thức hàm relu
def relu(x: torch.tensor):
    return torch.max(torch.tensor(0.0), x)

# Công thức hàm softmax
def softmax(zi: torch.tensor):
    exp_zi = torch.exp(zi)
    return exp_zi / torch.sum(exp_zi)

# Công thức hàm tanh
def tanh(x: torch.tensor):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

# Giao diện với Streamlit
st.title("Machine Learning Loss & Activation Functions")

# Nhập các giá trị inputs và targets
inputs = st.text_input("Nhập giá trị inputs (cách nhau bằng dấu phẩy):", "0.1, 0.3, 0.6, 0.7")
inputs = torch.tensor([float(i) for i in inputs.split(",")])

target = st.text_input("Nhập giá trị targets (cách nhau bằng dấu phẩy):", "0.31, 0.32, 0.8, 0.2")
target = torch.tensor([float(i) for i in target.split(",")])

n = len(inputs)

# Tính toán các hàm loss
if st.button("Tính Loss Functions"):
    mse = meanSquareError(inputs, target)
    binary_loss = binaryEntropyLoss(inputs, target, n)
    cross_loss = crossEntropyLoss(inputs, target)

    st.write(f"Mean Square Error: {mse.item()}")
    st.write(f"Binary Entropy Loss: {binary_loss.item()}")
    st.write(f"Cross Entropy Loss: {cross_loss.item()}")

# Tính toán các hàm activation
x = torch.tensor([1, 5, -4, 3, -2])

if st.button("Tính Activation Functions"):
    f_sigmoid = sigmoid(x)
    f_relu = relu(x)
    f_softmax = softmax(x)
    f_tanh = tanh(x)

    st.write(f"Sigmoid = {f_sigmoid}")
    st.write(f"Relu = {f_relu}")
    st.write(f"Softmax = {f_softmax}")
    st.write(f"Tanh = {f_tanh}")
