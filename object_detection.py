from ultralytics import YOLO
import cv2

# Load the YOLOv8 model (You can specify 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x' for different versions)
model = YOLO('yolov8n.pt')  # 'n' for nano (small and fast model)

# Load the video
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("people-detection.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Draw results on the frame
    annotated_frame = results[0].plot()  # Plot the bounding boxes and labels on the frame

    # Display the frame with detections
    cv2.imshow('YOLOv8 Object Detection', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
