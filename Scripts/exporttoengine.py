from ultralytics import YOLO

model=YOLO("Modelos/yolov8l.pt")
model.export(format="engine")
#trt_model=YOLO("Modelos/yolo11n.engine")
#results=trt_model("py/testimg1.jpg")
#print(results)