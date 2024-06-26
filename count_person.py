from ultralytics import YOLO
from PyQt5 import QtCore, QtWidgets, QtGui
import cv2
import os
import time
from threading import Thread
import sys

from openvino.runtime import Core
from PIL import Image
import numpy as np
from api import api1
import qss
from load import load_model2 as load_model
# 不然每次YOLO处理都会输出调试信息
os.environ['YOLO_VERBOSE'] = 'False'


class Person_detect(QtWidgets.QMainWindow):

    def __init__(self):

        super().__init__()

        # 设置界面
        self.setupUI()
        self.w = 768
        self.h = 576
        self.iscamera = True
        self.videoBtn.clicked.connect(self.startVideo)
        self.camBtn.clicked.connect(self.startCamera)
        self.imgBtn.clicked.connect(self.img_detect)
        self.stopBtn.clicked.connect(self.stop)

        # 定义定时器，用于控制显示视频的帧率
        self.timer_camera = QtCore.QTimer()
        # 定时到了，回调 self.show_camera
        self.timer_camera.timeout.connect(self.show_camera)

        # 加载 YOLO nano 模型，第一次比较耗时，要20秒左右
        # self.model = YOLO('yolov8n.pt')
        # ------------------------------------------------------
        # self.model = 'xml_models/yolov8s_nncf_int8.xml'
        # self.core = Core()
        # self.det_ov_model = self.core.read_model(self.model)
        # self.device = 'GPU'
        # self.det_compiled_model = self.core.compile_model(
        #     self.det_ov_model, self.device)
        # self.label_map = ['person']
        # ---------------------单例模式--------------------------#
        self.det_compiled_model = load_model.det_compiled_model
        self.label_map = load_model.label_map
        # ------------------------------------------------------
        # 要处理的视频帧图片队列，目前就放1帧图片
        self.frameToAnalyze = []

        # 启动处理视频帧独立线程
        Thread(target=self.frameAnalyzeThreadFunc, daemon=True).start()

    def resizeEvent(self, event):
        # 获取新的窗口大小
        new_size = event.size()
        # 更新标签的文本内容
        print(f"Window size: {new_size.width()} x {new_size.height()}")
        self.w = new_size.width()
        self.h = new_size.height()

    def setupUI(self):

        self.resize(1250, 750)

        self.setWindowTitle('人体计数')

        # central Widget
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)
        centralWidget.setStyleSheet(
            qss.qss_background)
        # central Widget 里面的 主 layout
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        # 界面的上半部分 : 图形展示部分
        topLayout = QtWidgets.QHBoxLayout()
        self.label_treated = QtWidgets.QLabel("请选择输入源")
        self.label_treated.setAlignment(QtCore.Qt.AlignCenter)
        self.label_treated.setMinimumSize(768, 576)
        self.label_treated.setStyleSheet(
            'border:1px solid #D7E2F9;color:white;font-size:30px;')

        # topLayout.addWidget(self.label_ori_video)
        topLayout.addWidget(self.label_treated)

        mainLayout.addLayout(topLayout)

        # 界面下半部分： 输出框 和 按钮
        groupBox = QtWidgets.QGroupBox(self)

        bottomLayout = QtWidgets.QHBoxLayout(groupBox)
        self.textLog = QtWidgets.QTextBrowser()
        bottomLayout.addWidget(self.textLog)

        mainLayout.addWidget(groupBox)

        btnLayout = QtWidgets.QVBoxLayout()
        self.videoBtn = QtWidgets.QPushButton('🎞️视频文件')
        self.camBtn = QtWidgets.QPushButton('📹摄像头')
        self.imgBtn = QtWidgets.QPushButton('🆕图片')
        self.stopBtn = QtWidgets.QPushButton('🛑停止')
        self.videoBtn.setStyleSheet(qss.qss_btn)
        self.camBtn.setStyleSheet(qss.qss_btn)
        self.imgBtn.setStyleSheet(qss.qss_btn)
        self.stopBtn.setStyleSheet(qss.qss_btn)
        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.camBtn)
        btnLayout.addWidget(self.imgBtn)
        btnLayout.addWidget(self.stopBtn)

        bottomLayout.addLayout(btnLayout)

    def startCamera(self):

        # 参考 https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html

        # 在 windows上指定使用 cv2.CAP_DSHOW 会让打开摄像头快很多，
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.iscamera = True
        # self.cap.set(3, 768)
        # self.cap.set(4, 576)
        if not self.cap.isOpened():
            print("0号摄像头不能打开")
            return ()

        if self.timer_camera.isActive() == False:  # 若定时器未启动
            self.timer_camera.start(50)

    def is_video_file(self, file_path):
        VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.wmv', '.flv']
        extension = os.path.splitext(file_path)[-1].lower()
        return extension in VIDEO_EXTENSIONS

    def startVideo(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择文件", "", "All Files (*)")
        if file_path:
            print(file_path)
        if self.is_video_file(file_path):
            self.cap = cv2.VideoCapture(file_path)
            self.iscamera = False
            if not self.cap.isOpened():
                print("视频文件不能打开")
                return ()

            if self.timer_camera.isActive() == False:  # 若定时器未启动
                self.timer_camera.start(50)

    def show_camera(self):

        ret, frame = self.cap.read()  # 从视频流中读取
        if not ret:
            return

        # 把读到的16:10帧的大小重新设置
        if self.iscamera:
            frame = cv2.resize(frame, (768, 576))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            w = frame.shape[1]
            h = frame.shape[0]
            if w > h:
                h = int(h*768/w)
                w = 768
            else:
                w = int(w*576/h)
                h = 576
            frame = cv2.resize(frame, (w, h))
            # 视频色彩转换回RGB，OpenCV images as BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # qImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
        #                       QtGui.QImage.Format_RGB888)  # 变成QImage形式
        # 往显示视频的Label里 显示QImage
        # self.label_ori_video.setPixmap(QtGui.QPixmap.fromImage(qImage))

        # 如果当前没有处理任务
        if not self.frameToAnalyze:
            self.frameToAnalyze.append(frame)

    def frameAnalyzeThreadFunc(self):

        while True:
            if not self.frameToAnalyze:
                time.sleep(0.01)
                continue

            frame = self.frameToAnalyze.pop(0)
            t1 = time.time()
            detections = api1.detect(
                frame, self.det_compiled_model, 1)[0]
            image_with_boxes, count = api1.draw_results(
                detections, frame, self.label_map)
            t2 = time.time()
            ms = int((t2-t1)*1000)
            cv2.putText(
                image_with_boxes, f'FPS:{int(1000/ms)}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            cv2.putText(
                image_with_boxes, f'Person:{count}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            # img = cv2.resize(image_with_boxes, (768, 576))
            img = image_with_boxes
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
# ----------------------------------------------------
            # self.textLog.append(str(count))
            qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0],
                                  QtGui.QImage.Format_RGBA8888)  # 变成QImage形式

            self.label_treated.setPixmap(
                QtGui.QPixmap.fromImage(qImage))  # 往显示Label里 显示QImage
            # print(self.width(), self.height())
            time.sleep(0.01)

    def is_image_file(self, file_path):
        IMAGE_EXTENSIONS = ['.jpg', '.png']
        extension = os.path.splitext(file_path)[-1].lower()
        return extension in IMAGE_EXTENSIONS

    def img_detect(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择文件", "", "All Files (*)")
        if file_path:
            print(file_path)
        if self.is_image_file(file_path):
            img = cv2.imread(file_path)
            w = img.shape[1]
            h = img.shape[0]
            if w > h:
                h = int(h*768/w)
                w = 768
            else:
                w = int(w*576/h)
                h = 576
            img = cv2.resize(img, (w, h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = api1.detect(
                img, self.det_compiled_model, 1)[0]
            image_with_boxes, count = api1.draw_results(
                detections, img, self.label_map)
            cv2.putText(
                image_with_boxes, f'Person:{count}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            img = image_with_boxes
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0],
                                  QtGui.QImage.Format_RGBA8888)

            self.label_treated.setPixmap(
                QtGui.QPixmap.fromImage(qImage))

    def stop(self):
        self.timer_camera.stop()  # 关闭定时器
        self.cap.release()  # 释放视频流
        self.label_treated.clear()  # 清空视频显示区域
        self.label_treated.setText('请选择输入源')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Person_detect()
    window.show()
    app.exec()
