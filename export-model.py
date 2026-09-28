from ultralytics import YOLO

YOLO("models/best_fp32.pt").export(format="onnx")
# -> tự sinh ra models/best_fp32.onnx

YOLO("models/best_int8.pt").export(
    format="onnx", int8=True,
    data="/home/nntruong/Bản tải về/archive (4)/Vehicle_Detection_Image_Dataset/data.yaml"
)
# -> tự sinh ra models/best_int8.onnx
