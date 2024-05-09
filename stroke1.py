import cv2
import numpy as np

# 새로운 빈 이미지 생성
canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255

# 이전 마우스 위치 초기화
prev_pt = None

# 획의 시작점과 끝점을 저장할 리스트 초기화
strokes = []

def draw(event, x, y, flags, param):
    global prev_pt, strokes

    # 마우스 왼쪽 버튼이 눌린 경우
    if event == cv2.EVENT_LBUTTONDOWN:
        prev_pt = (x, y)

    # 마우스 이동 중인 경우
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if prev_pt is not None:
            cv2.line(canvas, prev_pt, (x, y), (0, 0, 0), 2)
            prev_pt = (x, y)

    # 마우스 왼쪽 버튼이 놓인 경우
    elif event == cv2.EVENT_LBUTTONUP:
        if prev_pt is not None:
            strokes.append((prev_pt, (x, y)))
            prev_pt = None

# 윈도우 생성 및 이벤트 콜백 함수 설정
cv2.namedWindow('Canvas')
cv2.setMouseCallback('Canvas', draw)

# 그림판 열기
while True:
    cv2.imshow('Canvas', canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 키 누르면 종료
        break

# 각 획의 시작점과 끝점 출력
print("획 순서, 시작점, 끝점")
for i, (start, end) in enumerate(strokes):
    print(f"{i+1}: 시작점={start}, 끝점={end}")

# 그림판 닫기
cv2.destroyAllWindows()
