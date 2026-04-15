from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

for i in range(0, 50, 5):
    img = cv2.imread(f"frames/frame_{i:04d}.png")
    results = model(img)
    print(i, results[0].boxes)

results = model(img)

for r in results:
    print(r.boxes)