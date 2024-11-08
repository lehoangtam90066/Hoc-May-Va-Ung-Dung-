Phân loại ảnh CIFAR-10 với MLP
Dự án này minh họa quy trình cơ bản để huấn luyện một mạng MLP (Multi-Layer Perceptron) nhằm phân loại hình ảnh từ tập dữ liệu CIFAR-10 bằng cách sử dụng PyTorch.

Cấu trúc Dự Án
Chuẩn bị Dữ liệu: Tải và chuẩn hóa tập dữ liệu CIFAR-10.
Kiến trúc Mô hình: Xây dựng mô hình MLP đơn giản.
Huấn luyện & Đánh giá: Huấn luyện mô hình, theo dõi hiệu suất và đánh giá trên tập kiểm tra.
Hàm Chuẩn hóa Tuỳ chỉnh: Tự xây dựng hàm chuẩn hóa zScoreScaling và minMaxScaling.
Lớp Linear Tuỳ chỉnh: Tự xây dựng lớp Linear đơn giản từ đầu.
Các Yêu Cầu
Python 3.x
PyTorch
torchvision
numpy
matplotlib
Cài đặt các thư viện cần thiết bằng:

bash
Sao chép mã
pip install torch torchvision numpy matplotlib
Tổng Quan về Code
Bước 1: Chuẩn bị Dữ liệu
Dữ liệu CIFAR-10 được tải xuống, chuẩn hóa với giá trị trung bình 0.5 và độ lệch chuẩn 0.5, sau đó chia thành tập huấn luyện và tập kiểm tra.

python
Sao chép mã
import torchvision
from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset =  torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
Bước 2: Mô hình MLP
Mô hình MLP bao gồm hai lớp với hàm kích hoạt ReLU.

python
Sao chép mã
import torch.nn as nn

def getModel(n_features):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_features, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return model
Bước 3: Hàm Mất mát và Bộ Tối ưu
Mô hình sử dụng CrossEntropyLoss làm hàm mất mát và SGD làm bộ tối ưu với tốc độ học là 0.01.

python
Sao chép mã
import torch
from torch.optim import SGD

n_features = 3 * 32 * 32
model = getModel(n_features)
loss_fn = nn.CrossEntropyLoss()
optim = SGD(model.parameters(), lr=0.01)
Bước 4: Huấn luyện và Đánh giá
Mỗi epoch bao gồm bước huấn luyện và đánh giá. Độ chính xác của tập huấn luyện và tập kiểm tra được ghi lại và hiển thị qua đồ thị.

python
Sao chép mã
n_epochs = 10
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(n_epochs):
    # Huấn luyện và đánh giá tại đây
Bước 5: Hiển thị Kết quả
Vẽ đồ thị cho tổn thất và độ chính xác của tập kiểm tra qua các epoch.

python
Sao chép mã
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 5))
plt.subplot(121)
plt.title('Loss Epochs')
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.subplot(122)
plt.title('Accuracy Epoch')
plt.plot(test_accuracies, label='Test Accuracy')
plt.legend()
Bước 6: Hàm Chuẩn hóa Tùy chỉnh
Hàm zScoreScaling và minMaxScaling dùng để chuẩn hóa tensor.

python
Sao chép mã
def zScoreScaling(tensor):
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True)
    return (tensor - mean) / std

def minMaxScaling(tensor):
    min_val = tensor.min(dim=0, keepdim=True).values
    max_val = tensor.max(dim=0, keepdim=True).values
    return (tensor - min_val) / (max_val - min_val)
Bước 7: Lớp Linear Tuỳ chỉnh
Xây dựng lớp Linear đơn giản từ đầu để thực hiện các phép tính trong mạng neural cơ bản.

python
Sao chép mã
class Linear:
    def __init__(self, in_features, out_features):
        self.weight = torch.randn(out_features, in_features)
        self.bias = torch.randn(out_features)
Sử dụng
Để huấn luyện và đánh giá mô hình, chỉ cần chạy mã trong môi trường đã cài đặt các thư viện cần thiết.
