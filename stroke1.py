import cv2
import numpy as np

# 파일명
file_name = "strokes_data.txt"

# 획의 시작점, 끝점 및 해당 획의 벡터값을 저장할 리스트 초기화
strokes = []

def draw(event, x, y, flags, param):
    global prev_pt, strokes
    # 마우스 왼쪽 버튼이 눌린 경우
    if event == cv2.EVENT_LBUTTONDOWN:
        strokes.append((x, y))
        prev_pt = (x, y)

    # 마우스 이동 중인 경우
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if prev_pt is not None:
            cv2.line(canvas, prev_pt, (x, y), (0, 0, 0), 2)
            prev_pt = (x, y)

    # 마우스 왼쪽 버튼이 놓인 경우
    elif event == cv2.EVENT_LBUTTONUP:
        if prev_pt is not None:
            strokes.append((x, y))
            prev_pt = None

            # 획이 3개 그려졌을 때 파일에 추가
            if len(strokes) == 6:
                # 시작점, 끝점 및 해당 획의 벡터값을 배열에 저장
                stroke_data = []
                for i in range(0, len(strokes), 2):
                    start = strokes[i]
                    end = strokes[i + 1]
                    vector = np.array(end) - np.array(start)
                    stroke_data.append([start[0], start[1], end[0], end[1], vector[0], vector[1]])
                # 배열을 파일에 추가 저장
                with open(file_name, 'a') as f:
                    for data in stroke_data:
                        f.write(','.join(map(str, data)) + '\n')  # 데이터를 새로운 줄에 추가
                print("획 데이터를 파일에 추가했습니다.")
                strokes.clear()  # strokes 비우기

# 새로운 빈 이미지 생성
canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255

# 윈도우 생성 및 이벤트 콜백 함수 설정
cv2.namedWindow('Canvas')
cv2.setMouseCallback('Canvas', draw)

# 그림판 열기
while True:
    cv2.imshow('Canvas', canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 키 누르면 종료
        break

# 그림판 닫기
cv2.destroyAllWindows()
