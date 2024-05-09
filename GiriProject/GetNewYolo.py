import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.plotting import Annotator
import torch
from ultralytics.utils.instance import Bboxes

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

        # YOLO 학습 모델 로드
        model_path = "C:/Users/Jimin Lee/OneDrive/바탕 화면/AIproject/2023-02-Giri/GiriProject/new_data_yolo.pt"        
        self.model = YOLO(model_path)

        # RealSense D455 카메라 설정
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        
        self.setLayout(layout)
        
        # yolov8 깊이 스케일 작성
        self.depth_scale = 0.0010000000474974513

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

        # YOLOv8로 객체 감지
        results = self.model(color_image, conf=0.25)
        annotator = Annotator(color_image, example=self.model.names) #이미지에 자동으로 주석달기
        
        for box in results[0].boxes.xyxy.cpu():
            width, height, area = annotator.get_bbox_dimension(box)
            print("Bounding Box Width {}, Height {}, Area {}".format(
                width.item(), height.item(), area.item()))

        for result in results :
            boxes = result.boxes.xyxy.tolist()
        
        # Clear the dictionary of detected objects
        self.detected_objects.clear()

        # get camera intrinsics
        intr = self.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        # 이미지 표시
        self.show_images(color_image, ir_image, depth_image)
        
        # 결과 처리
        '''
        for *xyxy, conf, cls in results:
            xywh = xyxy2xywh(xyxy)
            print(xywh)'''
            #    x1, y1, x2, y2, confidence, class_id = obj         

            # 중심점과 반지름 계산
            #    center = ((int)(x1 + x2) // 2, (int)(y1 + y2) // 2)
            #    radius = int(max(abs(x2 - x1) // 2, abs(y2 - y1) // 2))
            #    centerDepth = depth_frame.get_distance((x1+x2)/2,(y1+y2)/2)

                # 객체 정보를 저장
            #    self.detected_objects[idx] = {'class': self.model.names[int(class_id)], 'center': center, 'radius':radius, 'centerDepth':centerDepth}

                # 객체 주위에 사각형 그리기
            #    cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (252, 119, 30), 2)

                # 바운딩 박스에 객체 번호 출력
            #    cv2.putText(color_image, f"{idx}", (int(center[0]), int(center[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (252, 119, 30), 2)

        # 결과 처리
        #for *xyxy, conf, cls in results: 
            #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
  
            #if save_img or view_img:  # Add bbox to image 
             #   label = '%s %.2f' % (names[int(cls)], conf) 
              #  plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3) 

        # 거리 정보를 QLabel에 표시
        '''
        for idx, obj in self.detected_objects.items():
            i=1
            if obj['class'] == "head" & idx == i:
                headCenterX = obj['center'][0]
                headCenterY = obj['center'][1]
                headCenterDepth = obj['centerDepth']
                headRadius = obj['radius']

            elif obj['class'] == "tail" & idx == i+1:
                tailCenterX = obj['center'][0]
                tailCenterY = obj['center'][1]
                tailCenterDepth = obj['centerDepth']
                tailRadius = obj['radius']

            i+=2
            fishCM = headRadius + np.sqrt(abs(headCenterX - tailCenterX)**2 + abs(headCenterY - tailCenterY)**2 + abs(headCenterDepth-tailCenterDepth)**2) 
            + tailRadius
        
            
        self.label_distance.setText("\n".join([f"{obj['class']} {idx} - {fishCM}" for idx, obj in self.detected_objects.items()]))
'''
        #객체 저장한 것을 매개로 인덱스 구분해서 각 바운딩박스 길이 출력 구현
        # flouder 0 head 1 tail 2 

        # 이미지 표시
        #self.show_images(color_image, ir_image, depth_image)

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