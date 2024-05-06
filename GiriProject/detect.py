from ultralytics import YOLO

# load a pretrained Yolov8n model
model = YOLO('best.pt')

# run inference on the source
results = model(source=1, show=True, conf=0.4, save=True) #generator of results objects