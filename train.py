import os
from ultralytics import YOLO

if not os.path.exists("runs"):
    model = YOLO("./yolo11n.pt")
    model.train(data="./data.yaml", epochs=20, imgsz=640)
else:
    # look for latest run and then continue training
    path = "./runs/detect/" + [i for i in os.listdir("./runs/detect")][-1] + "/weights/best.pt"
    print("Continuing training from last run", path)
    model = YOLO(path)
    model.train(data="./data.yaml", resume=True)
