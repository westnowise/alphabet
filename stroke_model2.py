from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# 데이터셋 로드
dataset = np.loadtxt("strokes_data.txt", delimiter=',', dtype=np.float32)
X = dataset[:, 5:7]  # 입력 데이터 (시작점 x, 시작점 y, 끝점 x, 끝점 y, 벡터 x, 벡터 y)
y = dataset[:, 0] - 1  # 라벨 데이터 (라벨링 값)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree 모델 초기화
model = DecisionTreeClassifier()

# 모델 학습
model.fit(X_train, y_train)

# 테스트 데이터로 평가
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# 학습이 완료된 후 모델을 저장
import joblib
joblib.dump(model, "decision_tree_model.pkl")

print("학습 완료")
