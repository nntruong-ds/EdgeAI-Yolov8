"""
Bản edge-optimized của real_time_traffic_analysis.py gốc.

Đã thêm đúng 2 kỹ thuật, KHÔNG đụng vào logic đếm làn / polygon / ngưỡng gốc:

  1) Non-blocking camera thread + queue frame-dropping (class FrameGrabber)
     -> Đọc video ở luồng riêng, không để CPU chờ cap.read().

  2) Frame skipping (--skip)
     -> Cứ mỗi --skip frame mới gọi best_model.predict() 1 lần (bước nặng).
     -> Các frame ở giữa dùng lại kết quả bbox của lần predict() gần nhất
        (vẽ + đếm lại trên đúng vị trí cũ) thay vì bỏ trắng màn hình.

Cách dùng:
    python3 real_time_traffic_analysis_edge.py --model models/best.pt --skip 3
    python3 real_time_traffic_analysis_edge.py --model models/best_fp32.onnx --skip 5
"""

import argparse
import queue
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO


# ------------------------------------------------------------------
# 1) NON-BLOCKING CAMERA THREAD + FRAME-DROPPING QUEUE
# ------------------------------------------------------------------
class FrameGrabber:
    """Luồng riêng liên tục đọc video, luôn giữ đúng 1 frame mới nhất trong
    queue. Nếu nguồn là file video, tự giả lập đúng nhịp FPS gốc (nếu không,
    OpenCV sẽ đọc hết cả video trong chưa đầy 1 giây, không phản ánh đúng
    tình huống camera thực tế)."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được nguồn video: {source}")

        self.is_file_source = isinstance(source, str)
        src_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_interval = (1.0 / src_fps) if (self.is_file_source and src_fps > 0) else 0.0

        self.q = queue.Queue(maxsize=1)
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        next_deadline = time.time()
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                self.stopped = True
                break

            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

            if self.frame_interval > 0:
                next_deadline += self.frame_interval
                sleep_time = next_deadline - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_deadline = time.time()

    def read(self, timeout=1.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_running(self):
        return not self.stopped

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


# ------------------------------------------------------------------
# CODE GỐC TỪ ĐÂY -- GIỮ NGUYÊN LOGIC, CHỈ THAY VÒNG LẶP ĐỌC/PREDICT
# ------------------------------------------------------------------

def run(model_path, source_path, skip_n):
    # Load the best fine-tuned YOLOv8 model
    best_model = YOLO(model_path)

    # Define the threshold for considering traffic as heavy
    heavy_traffic_threshold = 10

    # Define the vertices for the quadrilaterals
    vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
    vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)

    # Define the vertical range for the slice and lane threshold
    x1, x2 = 325, 635
    lane_threshold = 609

    # Define the positions for the text annotations on the image
    text_position_left_lane = (10, 50)
    text_position_right_lane = (820, 50)
    intensity_position_left_lane = (10, 100)
    intensity_position_right_lane = (820, 100)

    # Define font, scale, and colors for the annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)    # White color for text
    background_color = (0, 0, 255)  # Red background for text

    # Mở video qua FrameGrabber (non-blocking) thay vì cv2.VideoCapture trực tiếp
    grabber = FrameGrabber(source_path).start()

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        'processed_sample_video.avi', fourcc, 20.0,
        (int(grabber.cap.get(3)), int(grabber.cap.get(4)))
    )

    frame_idx = 0
    n_processed = 0
    t_start = time.time()
    cached_boxes_xyxy = np.empty((0, 4))  # bbox của lần predict() gần nhất, dùng lại ở frame bị skip

    while grabber.is_running():
        frame = grabber.read()
        if frame is None:
            break

        processed_frame = frame.copy()
        run_yolo_this_frame = (frame_idx % skip_n == 0)

        if run_yolo_this_frame:
            # --- Frame "nặng": chạy YOLO thật, giống hệt code gốc ---
            detection_frame = frame.copy()
            detection_frame[:x1, :] = 0
            detection_frame[x2:, :] = 0

            results = best_model.predict(detection_frame, imgsz=640, conf=0.4)
            processed_frame = results[0].plot(line_width=1)

            processed_frame[:x1, :] = frame[:x1, :].copy()
            processed_frame[x2:, :] = frame[x2:, :].copy()

            cached_boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        else:
            # --- Frame "nhẹ": không gọi YOLO, chỉ vẽ lại bbox của lần predict() gần nhất ---
            for box in cached_boxes_xyxy:
                bx1, by1, bx2, by2 = map(int, box)
                cv2.rectangle(processed_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 1)

        # Draw the quadrilaterals on the processed frame
        cv2.polylines(processed_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(processed_frame, [vertices2], isClosed=True, color=(255, 0, 0), thickness=2)

        # Đếm xe theo làn -- dùng cached_boxes_xyxy (vừa predict thật, hoặc tái sử dụng khi skip)
        vehicles_in_left_lane = 0
        vehicles_in_right_lane = 0
        for box in cached_boxes_xyxy:
            if box[0] < lane_threshold:
                vehicles_in_left_lane += 1
            else:
                vehicles_in_right_lane += 1

        traffic_intensity_left = "Heavy" if vehicles_in_left_lane > heavy_traffic_threshold else "Smooth"
        traffic_intensity_right = "Heavy" if vehicles_in_right_lane > heavy_traffic_threshold else "Smooth"

        cv2.rectangle(processed_frame, (text_position_left_lane[0] - 10, text_position_left_lane[1] - 25),
                      (text_position_left_lane[0] + 460, text_position_left_lane[1] + 10), background_color, -1)
        cv2.putText(processed_frame, f'Vehicles in Left Lane: {vehicles_in_left_lane}', text_position_left_lane,
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        cv2.rectangle(processed_frame, (intensity_position_left_lane[0] - 10, intensity_position_left_lane[1] - 25),
                      (intensity_position_left_lane[0] + 460, intensity_position_left_lane[1] + 10), background_color, -1)
        cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_left}', intensity_position_left_lane,
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        cv2.rectangle(processed_frame, (text_position_right_lane[0] - 10, text_position_right_lane[1] - 25),
                      (text_position_right_lane[0] + 460, text_position_right_lane[1] + 10), background_color, -1)
        cv2.putText(processed_frame, f'Vehicles in Right Lane: {vehicles_in_right_lane}', text_position_right_lane,
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        cv2.rectangle(processed_frame, (intensity_position_right_lane[0] - 10, intensity_position_right_lane[1] - 25),
                      (intensity_position_right_lane[0] + 460, intensity_position_right_lane[1] + 10), background_color, -1)
        cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_right}', intensity_position_right_lane,
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        cv2.imshow('Real-time Traffic Analysis', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1
        n_processed += 1

    elapsed = time.time() - t_start
    src_fps = grabber.cap.get(cv2.CAP_PROP_FPS)
    print(f"Đã xử lý {n_processed} frame trong {elapsed:.2f}s "
          f"(~{n_processed / elapsed:.1f} FPS xử lý trung bình, skip={skip_n}, "
          f"FPS gốc của video: {src_fps:.1f})")

    grabber.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best.pt", help="Đường dẫn model .pt/.onnx")
    parser.add_argument("--source", default="sample_video.mp4", help="Đường dẫn video hoặc chỉ số webcam (0)")
    parser.add_argument("--skip", type=int, default=3, help="Cứ N frame mới gọi predict() 1 lần")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run(args.model, src, args.skip)