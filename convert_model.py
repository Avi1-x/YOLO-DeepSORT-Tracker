from ultralytics import YOLO

model = YOLO("./models/yolov8l.pt")
model.export(format="onnx")