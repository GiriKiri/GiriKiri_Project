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
        # 카메라에서 최신 프레임 가져오기
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        ir_frame = frames.get_infrared_frame()
        depth_frame = frames.get_depth_frame()

        # 프레임을 넘파이 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())
        ir_image = np.asanyarray(ir_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 컬러 이미지를 그레이스케일로 변환
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # 깊이 이미지를 미터로 변환
        depth_image = depth_image * self.depth_scale

        # YOLOv5로 객체 감지
        results = self.model(color_image)

        # Clear the dictionary of detected objects
        self.detected_objects.clear()

        # get camera intrinsics
        intr = self.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()




        # 결과 처리
        for idx, result in enumerate(results.xyxy[0], start=1):
            x1, y1, x2, y2, confidence, class_id = result            

            # 상단과 하단 가장자리의 중점 계산
            upper_midpoint = ((x1 + x2) / 2, y1)
            lower_midpoint = ((x1 + x2) / 2, y2)

            distUp = depth_frame.get_distance((x1+x2)/2,y1)
            distDown = depth_frame.get_distance((x1+x2)/2,y2)




            realUpX = distUp*((x1+x2)/2-intr.ppx)/intr.fx
            realUpY = distUp*(y1-intr.ppy)/intr.fy
            realUpZ = distUp

            realDownX = distDown*((x1+x2)/2-intr.ppx)/intr.fx
            realDownY = distDown*(y2-intr.ppy)/intr.fy
            realDownz = distUp

            real_between_points = np.sqrt((realUpX-realDownX)**2+(realDownY-realUpY)**2+(realDownz-realUpZ)**2)


            # 중점에서의 깊이 계산
            upper_depth = 0.0
            if 0 <= int(upper_midpoint[1]) < 480 and 0 <= int(upper_midpoint[0]) < 640:
                upper_depth = np.median(depth_image[int(upper_midpoint[1]), int(upper_midpoint[0])])

            lower_depth = 0.0
            if 0 <= int(lower_midpoint[1]) < 480 and 0 <= int(lower_midpoint[0]) < 640:
                lower_depth = np.median(depth_image[int(lower_midpoint[1]), int(lower_midpoint[0])])

            # 좌표에 스케일 적용
            upper_midpoint_scaled = (
            upper_midpoint[0] * 20.50249535474523 * self.k, upper_midpoint[1] * 20.50249535474523 * self.k)
            lower_midpoint_scaled = (
            lower_midpoint[0] * 20.50249535474523 * self.k, lower_midpoint[1] * 20.50249535474523 * self.k)

            # 두 점 사이의 거리 계산
            distance_between_points = np.sqrt((upper_midpoint_scaled[0] - lower_midpoint_scaled[0]) ** 2 +
                                              (upper_midpoint_scaled[1] - lower_midpoint_scaled[1]) ** 2 + (
                                              upper_depth - lower_depth) ** 2)

             # 객체 정보를 저장
            self.detected_objects[idx] = {'class': self.model.names[int(class_id)], 'distance': distance_between_points, 'realDistance':real_between_points}
            
  

            # 객체 주위에 사각형 그리기
            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (252, 119, 30), 2)

            # 중점 그리기
            cv2.circle(color_image, (int(upper_midpoint[0]), int(upper_midpoint[1])), 5, (0, 255, 0), -1)
            cv2.circle(color_image, (int(lower_midpoint[0]), int(lower_midpoint[1])), 5, (0, 255, 0), -1)

            # 바운딩 박스에 객체 번호 출력
            cv2.putText(color_image, f"{idx}", (int(x1), int(y1) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (252, 119, 30), 2)

        # 거리 정보를 QLabel에 표시
        self.label_distance.setText("\n".join([f"{obj['class']} {idx} - {obj['distance']:.2f} - {obj['realDistance']:.4f}" for idx, obj in self.detected_objects.items()]))

        # 이미지 표시
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