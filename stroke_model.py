import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 데이터셋 정의
class StrokeDataset(Dataset):
    def __init__(self, file_path):
        self.data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        # print(self.data)
    
    def __len__(self):
        # print(len(self.data))
        return len(self.data)
    
    def __getitem__(self, idx):
        # print(self.data[idx])
        return self.data[idx]

# 모델 정의
class StrokeModel(nn.Module):
    def __init__(self):
        super(StrokeModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # 입력 차원은 6 (시작점 x, 시작점 y, 끝점 x, 끝점 y, 벡터 x, 벡터 y), 은닉층의 크기는 64
        self.fc2 = nn.Linear(64, 32)  # 은닉층의 크기는 64, 32
        self.fc3 = nn.Linear(32, 3)   # 은닉층의 크기는 32, 출력 차원은 3 (예측된 벡터 값)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 첫 번째 은닉층에 ReLU 활성화 함수 적용
        x = torch.relu(self.fc2(x))  # 두 번째 은닉층에 ReLU 활성화 함수 적용
        x = self.fc3(x)
        return x

# 데이터 로드
dataset = StrokeDataset("strokes_data.txt")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 초기화
model = StrokeModel()

# 손실 함수 및 최적화 알고리즘 설정
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss 사용
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 학습 실행
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for data in dataloader:
        inputs = data[:, 5:7]  # 입력 데이터 (시작점 x, 시작점 y, 끝점 x, 끝점 y, 벡터 x, 벡터 y)
        labels = data[:, 0].long() - 1  # 라벨 데이터 (라벨링 값)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")
# 학습이 완료된 후 모델을 저장합니다.
torch.save(model.state_dict(), "stroke_model.pth")

print("학습 완료")
