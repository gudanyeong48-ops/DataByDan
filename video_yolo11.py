from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO("yolo11n.pt")

# 영상 열기
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# 영상 정보
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('result_yolo11.mp4', fourcc, fps, (width, height))

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # 객체 검출
    results = model(frame)

    # 결과 프레임
    annotated_frame = results[0].plot()

    # 저장
    out.write(annotated_frame)

    # 화면 출력
    cv2.imshow("YOLO11 Detection", annotated_frame)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("완료")