import cv2
import pyrealsense2 as rs
import numpy as np

class Depth_Camera():

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.align_to = None
        self.depth_sum = 0
        

        context = rs.context()
        connect_device = None
        if context.devices[0].get_info(rs.camera_info.name).lower() != 'platform camera':
            connect_device = context.devices[0].get_info(rs.camera_info.serial_number)

        print(" > Serial number : {}".format(connect_device))
        self.config.enable_device(connect_device)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
        self.config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 6)

    def __del__(self):
        print("Collecting process is done.\n")

    def execute(self):
        print('Collecting depth information...')
        try:
            self.pipeline.start(self.config)
        except:
            print("There is no signal sended from depth camera.")
            print("Check connection status of camera.")
            return
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        try:
            while True:

                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                depth_info = depth_frame.as_depth_frame()
                ir_frame = aligned_frames.get_infrared_frame()

                self.rgb_sum = np.zeros(3)


                depth_sum = 0
                for x in range(640):
                    for y in range(480):
                        depth_sum += round(depth_info.get_distance(x, y) * 100,2)

                print("AvgDepth : ", round(depth_sum/307200,2))
                

                color_image = np.asanyarray(color_frame.get_data())
                ir_image = np.asanyarray(ir_frame.get_data())

                #RGB 평균값 계산
                self.rgb_sum += np.sum(color_image, axis=(0, 1))
                average_rgb = self.rgb_sum/307200

                #RGB 평균값 출력
                print(f"Average RGB: {average_rgb}")

                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                )

                ir_image_color = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)

                images = np.hstack((color_image, depth_colormap,ir_image_color))

                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense',images)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline.stop()
        print("┌──────────────────────────────────────┐")
        print('│ Collecting of depth info is stopped. │')
        print("└──────────────────────────────────────┘")

if __name__ == "__main__":
    depth_camera = Depth_Camera()
    depth_camera.execute()