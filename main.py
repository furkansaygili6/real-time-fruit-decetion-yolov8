import cv2
from ultralytics import YOLO

model = YOLO(r"C:\Users\90530\Downloads\best.pt")  # Eğitilmiş modelin dosya adını kullan

cap = cv2.VideoCapture(0)  # 0, varsayılan kamerayı açar

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
