from ultralytics import YOLO
import cv2

model = YOLO("models/best_fp32.onnx")
cap = cv2.VideoCapture("sample_video.mp4")
ok, frame = cap.read()
print("Kích thước frame:", frame.shape)

results = model.track(frame, persist=True, conf=0.1)
print("Số box phát hiện (không crop):", len(results[0].boxes))

import numpy as np

h, w = frame.shape[:2]
left = [(0.33*w, 0.45*h), (0.50*w, 0.45*h), (0.55*w, 0.95*h), (0.05*w, 0.95*h)]
right = [(0.50*w, 0.45*h), (0.67*w, 0.45*h), (0.95*w, 0.95*h), (0.45*w, 0.95*h)]
all_pts = np.array(left + right)
x1, y1 = int(all_pts[:,0].min()), int(all_pts[:,1].min())
x2, y2 = int(all_pts[:,0].max()), int(all_pts[:,1].max())
cv2.imwrite("debug_crop.jpg", frame[y1:y2, x1:x2])
print(f"Vùng crop: ({x1},{y1}) đến ({x2},{y2})")