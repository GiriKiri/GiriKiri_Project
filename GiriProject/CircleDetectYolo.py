import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import cv2
import numpy as np
import pyrealsense2 as rs
import torch

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

        # Dictionary to store information about detected objects
        self.detected_objects = {}

    def initUI(self):
        self.setGeometry(100, 100, 1200, 400)
        self.setWindowTitle('Object Detection and Distance Measurement')

        self.label_color_image = QLabel(self)
        self.label_ir_image = QLabel(self)
        self.label_depth_image = QLabel(self)
        self.label_distance = QLabel(self)

        layout_images = QHBoxLayout()
        layout_images.addWidget(self.label_color_image)
        layout_images.addWidget(self.label_ir_image)
        layout_images.addWidget(self.label_depth_image)
 
        layout = QVBoxLayout()
        layout.addLayout(layout_images)
        layout.addWidget(self.label_distance)

        self.setLayout(layout)

        # YOLOv5 모델 로드
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        # RealSense D455 카메라 설정
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)


        # yolov5 깊이 스케일 작성
        self.depth_scale = 0.0010000000474974513

        self.k = 1

        # 타이머 설정
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30 milliseconds (33 fps)

        self.show()
    
    def get_stream(self, stream_type):
        # Get stream from the RealSense pipeline
        return self.pipeline.get_active_profile().get_stream(stream_type)

    def update_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        ir_frame = frames.get_infrared_frame()
        depth_frame = frames.get_depth_frame()

        color_image = np.asanyarray(color_frame.get_data())
        ir_image = np.asanyarray(ir_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        depth_image = depth_image * self.depth_scale * 1.333

        results = self.model(color_image)

        self.detected_objects.clear()

        intr = self.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        for idx, result in enumerate(results.xyxy[0], start=1):
            x1, y1, x2, y2, confidence, class_id = result
            center_x = (int)((x1 + x2) / 2)
            center_y = (int)((y1 + y2) / 2)
            radius = int(max(abs(x2 - x1) / 2, abs(y2 - y1) / 2))

            # Ensure coordinates are within the image dimensions
            max_x, max_y = color_image.shape[1] - 1, color_image.shape[0] - 1
            distRight_x = min(max(center_x + radius, 0), max_x)
            distRight_y = min(max(center_y, 0), max_y)
            distLeft_x = min(max(center_x - radius, 0), max_x)
            distLeft_y = min(max(center_y, 0), max_y)

            distRight = depth_frame.get_distance(distRight_x, distRight_y)
            distLeft = depth_frame.get_distance(distLeft_x, distLeft_y)

            realRightX = distRight * (distRight_x - intr.ppx) / intr.fx
            realRightY = distRight * (distRight_y - intr.ppy) / intr.fy
            realRightZ = distRight

            realLeftX = distLeft * (distLeft_x - intr.ppx) / intr.fx
            realLeftY = distLeft * (distLeft_y - intr.ppy) / intr.fy
            realLeftZ = distLeft

            length = np.sqrt((realRightX-realLeftX)**2+(realRightY-realLeftY)**2+(realRightZ-realLeftZ)**2)

            self.detected_objects[idx] = {'class': self.model.names[int(class_id)], 'radius': radius, 'center': (center_x, center_y), 'length': length}

            cv2.putText(color_image, f"{idx}", (int(center_x), int(center_y) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (252, 119, 30), 2)

            #     # 거리 정보를 QLabel에 표시
        self.label_distance.setText("\n".join([f"{obj['class']} {idx} - {obj['radius']:.2f} - {obj['length']:.2f}" for idx, obj in self.detected_objects.items()]))

        self.show_images(color_image, ir_image, depth_image)

    def show_images(self, color_image, ir_image, depth_image):
        # Resize images for display
        resize_factor = 0.8
        color_image = cv2.resize(color_image, (0, 0), fx=resize_factor, fy=resize_factor)
        ir_image = cv2.resize(ir_image, (0, 0), fx=resize_factor, fy=resize_factor)
        depth_image = cv2.resize(depth_image, (0, 0), fx=resize_factor, fy=resize_factor)

        h, w, ch = color_image.shape
        bytes_per_line = ch * w

        # 컬러 이미지를 PyQt에 표시
        q_image_color = QImage(color_image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap_color = QPixmap.fromImage(q_image_color)
        self.label_color_image.setPixmap(pixmap_color)

        # IR 이미지를 PyQt에 표시
        q_image_ir = QImage(ir_image.data, w, h, w, QImage.Format_Grayscale8)
        pixmap_ir = QPixmap.fromImage(q_image_ir)
        self.label_ir_image.setPixmap(pixmap_ir)

        # 깊이 이미지를 PyQt에 표시
        q_image_depth = QImage(depth_image.data, w, h, w * 2, QImage.Format_Grayscale16)
        pixmap_depth = QPixmap.fromImage(q_image_depth)
        self.label_depth_image.setPixmap(pixmap_depth)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    sys.exit(app.exec_())