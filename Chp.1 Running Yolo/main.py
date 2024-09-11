from unittest import result
from ultralytics import YOLO

import cv2
model= YOLO('\YoloWeights\yolov8n.pt')
results=model("Images/bikes.jpeg",show=True)
cv2.waitKey(0)