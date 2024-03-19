import cv2
import numpy as np
import pyrealsense2 as rs
import torch

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# RealSense D455 카메라 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# yolov5 깊이 스케일 작성
depth_scale = 0.0010000000474974513

k = 1


# 주 루프
while True:

    # 카메라에서 최신 프레임 가져오기
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    # 프레임을 넘파이 배열로 변환
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # 컬러 이미지를 그레이스케일로 변환
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # 깊이 이미지를 미터로 변환
    depth_image = depth_image * depth_scale

    # YOLOv5로 객체 감지
    results = model(color_image)

    # 결과 처리
    for result in results.xyxy[0]:
        x1, y1, x2, y2, confidence, class_id = result

        # 상단과 하단 가장자리의 중점 계산
        upper_midpoint = ((x1 + x2) / 2, y1)
        lower_midpoint = ((x1 + x2) / 2, y2)

        # 중점에서의 깊이 계산
        upper_depth = 0.0
        if 0 <= int(upper_midpoint[1]) < 480 and 0 <= int(upper_midpoint[0]) < 640:
         upper_depth = np.median(depth_image[int(upper_midpoint[1]), int(upper_midpoint[0])])

        lower_depth = 0.0
        if 0 <= int(lower_midpoint[1]) < 480 and 0 <= int(lower_midpoint[0]) < 640:
            lower_depth = np.median(depth_image[int(lower_midpoint[1]), int(lower_midpoint[0])])

        # 좌표에 스케일 적용
        upper_midpoint_scaled = (upper_midpoint[0] * 20.50249535474523 * k, upper_midpoint[1] * 20.50249535474523 * k)
        lower_midpoint_scaled = (lower_midpoint[0] * 20.50249535474523 * k, lower_midpoint[1] * 20.50249535474523 * k)

        # 두 점 사이의 거리 계산
        distance_between_points = np.sqrt((upper_midpoint_scaled[0] - lower_midpoint_scaled[0])**2 +
                                          (upper_midpoint_scaled[1] - lower_midpoint_scaled[1])**2 + (upper_depth-lower_depth)**2)

        print(f"두 점 사이의 거리: {distance_between_points:.2f}")
    
        
        # 객체 주위에 사각형 그리기
        cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (252, 119, 30), 2)

        # 중점 그리기
        cv2.circle(color_image, (int(upper_midpoint[0]), int(upper_midpoint[1])), 5, (0, 255, 0), -1)
        cv2.circle(color_image, (int(lower_midpoint[0]), int(lower_midpoint[1])), 5, (0, 255, 0), -1)

        # 바운딩 박스 그리기
        cv2.putText(color_image, f"{model.names[int(class_id)]}", (int(x1), int(y1) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (252, 119, 30), 2)

    # 이미지 표시
    cv2.imshow("Color Image", color_image)
    cv2.waitKey(1)