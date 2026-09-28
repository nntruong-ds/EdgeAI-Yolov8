# Edge-Optimized Real-Time Traffic Density Estimation with YOLOv8

Real-time vehicle detection and per-lane traffic density estimation that runs fully on-device (CPU), with no cloud calls. This project extends an existing YOLOv8 traffic-density demo with model export, benchmarking, and pipeline optimizations aimed at edge deployment.

## What I added

- **Model variants:** exported the fine-tuned YOLOv8 model to ONNX (FP32) and ONNX INT8 (post-training quantization).
- **Non-blocking capture:** a separate thread reads frames into a size-1 queue and drops stale frames, so slow inference never makes the video lag behind real time.
- **Frame skipping:** YOLO runs once every N frames (`--skip N`); in-between frames reuse the latest detections for drawing and lane counting.
- **Benchmark:** compared accuracy (mAP) and throughput (FPS) across model variants and skip levels.

## Results

Hardware: Intel Core i5-12450H, CPU only. Input: 1280x720 video at 30 FPS. FPS is the average processing rate over about 20 s of real-time playback.

| Model | mAP50 | mAP50-95 | FPS (skip=1) | FPS (skip=3) | FPS (skip=5) |
|---|---|---|---|---|---|
| `best.pt` (PyTorch) | 0.975 | 0.727 | 17.8 | 27.2 | 25.9 |
| `best_fp32.onnx` | 0.955 | 0.712 | 9.1 | 17.4 | 19.3 |
| `best_int8.onnx` | 0.950 | 0.696 | 7.4 | 15.5 | 17.4 |

mAP is measured on the validation split with `model.val()` and does not depend on the skip setting.

**Findings**

- Frame skipping gave the largest speedup: roughly 1.5-2x for the PyTorch model and about 2x for the ONNX models when going from skip=1 to skip=3.
- On this CPU, ONNX Runtime was slower than PyTorch, and INT8 was not faster than FP32. Quantization speedups depend on the target hardware and runtime, so they need to be measured on the actual device rather than assumed.
- Frame skipping only reuses old boxes, so counts can lag slightly behind fast-moving traffic at higher skip values.

## Usage

```bash
pip install ultralytics opencv-python onnxruntime

# Run the real-time demo (press q to quit)
python3 edge_optimized_traffic_analysis.py --model models/best.pt --skip 3
python3 edge_optimized_traffic_analysis.py --model models/best_int8.onnx --skip 5 --source sample_video.mp4
```

Export ONNX variants:

```python
from ultralytics import YOLO
YOLO("models/best.pt").export(format="onnx")                      # FP32
YOLO("models/best.pt").export(format="onnx", int8=True, data="path/to/data.yaml")  # INT8
```

Evaluate mAP:

```python
from ultralytics import YOLO
m = YOLO("models/best.onnx").val(data="path/to/data.yaml")
print(m.box.map50, m.box.map)
```

## Files

- `edge_optimized_traffic_analysis.py`: optimized real-time pipeline (this project's main script)
- `real_time_traffic_analysis.py`: original baseline script
- `models/`: PyTorch and ONNX weights
- `sample_video.mp4`: demo video

## Acknowledgements

Based on [YOLOv8_Traffic_Density_Estimation](https://github.com/FarzadNekouee/YOLOv8_Traffic_Density_Estimation) by Farzad Nekouee, including the fine-tuned model, the lane-counting logic, and the demo video. The ONNX/INT8 export, benchmarking, capture thread, and frame skipping are my additions. See `LICENSE.txt` for the original license.
