from ultralytics import YOLO

# load a pretrained Yolov8n model
# 이 부분은 무조건 절대경로로 설정해야함.
model_path = "C:/Users/Jimin Lee/OneDrive/바탕 화면/AIproject/2023-02-Giri/GiriProject/new_data_yolo.pt"
model = YOLO(model_path)
#model = YOLO('yolov5s.pt')
# run inference on the source
results = model(source=1, show=True, conf=0.2, save=True) ##generator of results objects

# Extract bounding box dimensions
result = next(results)
boxes = result.boxes.xywh.cpu()
for box in boxes:
    x, y, w, h = box
    print("Width of Box: {}, Height of Box: {}".format(w, h))