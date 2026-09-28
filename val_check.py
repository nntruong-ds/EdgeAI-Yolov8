from ultralytics import YOLO 
model = YOLO('/home/nntruong/YOLOv8_Traffic_Density_Estimation/models/best_int8.onnx') 
metrics = model.val(data='/home/nntruong/Bản tải về/archive (4)/Vehicle_Detection_Image_Dataset/data.yaml') 
print('mAP50:', metrics.box.map50) 
print('mAP50-95:', metrics.box.map)
