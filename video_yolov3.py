import cv2
import numpy as np

# 클래스 이름 불러오기
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# YOLOv3 모델 로드
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# 출력 레이어 설정
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 영상 불러오기
cap = cv2.VideoCapture("test.mp4")

# 영상 저장 설정
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('result_yolov3.mp4', fourcc, fps, (width, height))

while True:

    ret, frame = cap.read()

    if not ret:
        break

    height_frame, width_frame, channels = frame.shape

    # blob 생성
    blob = cv2.dnn.blobFromImage(
        frame,
        1 / 255.0,
        (416, 416),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # 객체 탐지
    for output in outputs:

        for detection in output:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:

                center_x = int(detection[0] * width_frame)
                center_y = int(detection[1] * height_frame)
                w = int(detection[2] * width_frame)
                h = int(detection[3] * height_frame)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 박스 그리기
    for i in range(len(boxes)):

        if i in indexes:

            x, y, w, h = boxes[i]

            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    out.write(frame)

    cv2.imshow("YOLOv3 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()