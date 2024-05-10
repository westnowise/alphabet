import cv2
import numpy as np
import torch
from stroke_model import StrokeModel

# 이미지 전처리 및 획 추출 함수
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 이미지 이진화
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return binary

def extract_strokes(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    strokes = []
    for contour in contours:
        # 각 윤곽선의 근사화
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        for point in approx:
            strokes.append((point[0], point[0], (0, 0)))  # 시작점과 끝점이 같은 선분으로 처리
        
        # 근사화된 윤곽선의 점들을 사용하여 획 추출
        for i in range(len(approx) - 1):
            start_point = approx[i][0]
            end_point = approx[i + 1][0]
            vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])
            strokes.append((start_point, end_point, vector))

    return strokes


# 모델을 사용하여 획 분류하는 함수
def classify_strokes(strokes, model):
    classified_strokes = []
    for stroke in strokes:
        # 획의 벡터값을 모델에 입력으로 제공하여 분류
        classification_result = model.predict(np.array(stroke[2]).reshape(1, -1))
        classified_strokes.append((stroke, classification_result))
    return classified_strokes

# 예시 모델 생성
model = StrokeModel()
model.load_state_dict(torch.load("stroke_model.pth"))
model.eval()

# 이미지 경로
image_path = "../img/A.png"

# 이미지 전처리 및 획 추출
binary_image = preprocess_image(image_path)
strokes = extract_strokes(binary_image)

# (0, 0)인 값은 제외
strokes = [stroke for stroke in strokes if stroke[2] != (0, 0)]

# 획 분류
classified_strokes = []
for stroke in strokes:
    # 획의 벡터값을 모델에 입력으로 제공하여 분류
    stroke_vector = torch.tensor(stroke[2], dtype=torch.float32).unsqueeze(0)
    print(stroke_vector)
    classification_result = model(stroke_vector)
    classified_strokes.append((stroke, classification_result.argmax().item() + 1))

# 결과 출력
for stroke, classification_result in classified_strokes:
    print("Stroke:", stroke)
    print("Classification Result:", classification_result)

# 원본 이미지 불러오기
original_image = cv2.imread(image_path)

# 분류된 획 중에서 가장 값이 좋은 3개의 획을 선택
# top_three_strokes = sorted(classified_strokes, key=lambda x: x[1], reverse=True)[:3]

# 각 그룹에서 가장 높은 값을 가진 획을 선택

# 각 라벨별로 획을 그룹화
stroke_groups = {}
for stroke, label in classified_strokes:
    if label not in stroke_groups:
        stroke_groups[label] = []
    stroke_groups[label].append((stroke, label))

top_three_strokes = []
for label, strokes in stroke_groups.items():
    top_stroke = max(strokes, key=lambda x: x[1])
    top_three_strokes.append(top_stroke)
    
# 선택된 획을 원본 이미지 위에 그리기
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # 초록색, 빨간색, 파란색
for idx, (stroke, label) in enumerate(top_three_strokes):
    start_point = stroke[0]
    end_point = stroke[1]
    color = colors[label - 1]  # 라벨에 해당하는 색 선택
    cv2.line(original_image, start_point, end_point, color, 2)  # 선 그리기

# 시각화된 이미지 보여주기
cv2.imshow("Top 3 Strokes", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 분류된 획을 원본 이미지 위에 그리기
for stroke, classification_result in classified_strokes:
    start_point = stroke[0]
    end_point = stroke[1]
    color = (0, 255, 0)  # 초록색
    if classification_result == 2:
        color = (0, 0, 255)  # 빨간색
    elif classification_result == 3:
        color = (255, 0, 0)  # 파란색
    cv2.line(original_image, start_point, end_point, color, 2)  # 선 그리기

cv2.imshow("Top 3 Strokes", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
