from PyQt5 import QtCore, QtWidgets, QtGui
import cv2
import os
import time
from threading import Thread
import sys

from openvino.runtime import Core
import numpy as np
from api import api2

import dlib
from imutils import face_utils
import qss
from load import load_model3, load_model1
# 不然每次YOLO处理都会输出调试信息
os.environ['YOLO_VERBOSE'] = 'False'


class Head_detect(QtWidgets.QMainWindow):

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

        self.det_compiled_model = load_model1.det_compiled_model
        self.label_map = load_model1.label_map
        self.face_landmark_path = './models/dlib_models/shape_predictor_68_face_landmarks.dat'
        self.predictor = dlib.shape_predictor(self.face_landmark_path)

        self.cam_matrix = load_model3.cam_matrix
        self.dist_coeffs = load_model3.dist_coeffs

        self.object_pts = load_model3.object_pts

        self.reprojectsrc = load_model3.reprojectsrc

        self.line_pairs = load_model3.line_pairs
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

        self.setWindowTitle('恒星不见')

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

    def get_head_pose(self, shape):
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])

        _, rotation_vec, translation_vec = cv2.solvePnP(
            self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs)

        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec, self.cam_matrix,
                                            self.dist_coeffs)

        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

        # calc euler angle
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        return reprojectdst, euler_angle

    def startCamera(self):

        # 参考 https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html

        # 在 windows上指定使用 cv2.CAP_DSHOW 会让打开摄像头快很多，
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.iscamera = True
        # self.predictor = dlib.shape_predictor(self.face_landmark_path)
        self.cap.set(3, 768)
        self.cap.set(4, 576)
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
                h = int(h*768//w)
                w = 768
            else:
                w = int(w*576//h)
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
            detections = api2.detect(frame, self.det_compiled_model, 1)[0]
            image_with_boxes, count, boxes = api2.draw_results(
                detections, frame, self.label_map)
            down = 0
            for idx, (*xyxy, conf, lbl) in enumerate(boxes):
                face_rect = dlib.rectangle(int(xyxy[0]), int(
                    xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                shape = self.predictor(frame, face_rect)
                shape = face_utils.shape_to_np(shape)

                reprojectdst, euler_angle = self.get_head_pose(shape)
                if euler_angle[0, 0] < 10:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                    down += 1
                cv2.rectangle(frame, (int(xyxy[0]), int(
                    xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 1, cv2.LINE_AA)
                # for (x, y) in shape:
                #     cv2.circle(frame, (x, y), 1, color, -1)
                # for start, end in self.line_pairs:
                #     cv2.line(frame, reprojectdst[start],
                #              reprojectdst[end], color)
                print("X: " + "{:7.2f}".format(euler_angle[0, 0]),
                      "Y: " + "{:7.2f}".format(euler_angle[1, 0]),
                      "Z: " + "{:7.2f}".format(euler_angle[2, 0]))

            t2 = time.time()
            ms = int((t2-t1)*1000)
            cv2.putText(
                image_with_boxes, f'FPS:{int(1000/ms)}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            cv2.putText(
                image_with_boxes, f'FACE:{count}__DOWN:{down}__per{down/(count+0.1)*100:.1f}% ', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 1)
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
            # self.predictor = dlib.shape_predictor(self.face_landmark_path)
            img = cv2.imread(file_path)
            w = img.shape[1]
            h = img.shape[0]
            print(w, h)
            if w >= h:
                h = int(h*768/w)
                w = 768
            else:
                w = int(w*576/h)
                h = 576
            img = cv2.resize(img, (w, h))
            # 视频色彩转换回RGB，OpenCV images as BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = api2.detect(
                img, self.det_compiled_model, 1)[0]
            image_with_boxes, count, boxes = api2.draw_results(
                detections, img, self.label_map)
            down = 0
            for idx, (*xyxy, conf, lbl) in enumerate(boxes):
                face_rect = dlib.rectangle(int(xyxy[0]), int(
                    xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                print(face_rect)
                shape = self.predictor(img, face_rect)
                shape = face_utils.shape_to_np(shape)

                reprojectdst, euler_angle = self.get_head_pose(shape)
                if euler_angle[0, 0] < 10:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                    down += 1
                cv2.rectangle(img, (int(xyxy[0]), int(
                    xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 1, cv2.LINE_AA)
                # for (x, y) in shape:
                #     cv2.circle(img, (x, y), 1, color, -1)

                # for start, end in self.line_pairs:
                #     cv2.line(img, reprojectdst[start],
                #              reprojectdst[end], color)

            cv2.putText(
                image_with_boxes, f'FACE:{count}__DOWN:{down}__per{down/(count+0.1)*100:.1f}% ', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 1)

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
    window = Head_detect()
    window.show()
    app.exec()
