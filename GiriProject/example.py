from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import sys
import cv2
import numpy as np
# 평균값 구하는 함수 class import
from DepthPoint_information import Depth_Camera

# from_class = uic.loadUiType("D:\\AIproject\\2023-02-Giri\\GiriProject\\ImageQT.ui")[0]

# open cv 비디오 실행
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False

        #self.wait()
    

# 위젯 실행
class App(QWidget):
    def __init__(self):
        super().__init__()
        # self.setupUi(self)
        
        self.setWindowTitle("Qt live label demo")
        self.display_width = 640
        self.display_height = 480
        # 이미지를 만드는 곳에 라벨 생성
        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)
        # 텍스트 생성
        self.textLabel = QLabel('Webcam')
        
        # 위에 두 라벨을 만들 레이아웃 생성
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        # vbox레이아웃으로 위젯레이아웃 생성
        self.setLayout(vbox)

        # 비디오 캡처 쓰레드 생성
        self.thread = VideoThread()
        # 시그널 slot 메커니즘을 이용해서 안전하게 이미지 업로드 (쓰레싱)
        self.thread.change_pixmap_signal.connect(self.update_image)
        # 쓰레드 시작
        self.thread.start()
        
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()
        
        
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
        
       
if __name__ == '__main__':
    app = QApplication(sys.argv)
    dashboard = App()
    dashboard.show()
    sys.exit(app.exec_())
