
from PyQt5 import QtCore, QtWidgets, QtGui
import os
from threading import Thread
import sys
from openvino.runtime import Core
import cv2
import time
import csv
import dlib
from imutils import face_utils
import numpy as np
from api import api, api2
from load import load_model1, load_model2, load_model3, load_model4
import qss


class Com(QtWidgets.QMainWindow):

    def __init__(self):

        super().__init__()

        # 设置界面
        self.setupUI()
        # -----------------------------------------#
        self.det_compiled_model_1 = load_model1.det_compiled_model
        self.label_map_1 = load_model1.label_map

        self.predictor = load_model3.predictor
        self.cam_matrix = load_model3.cam_matrix
        self.dist_coeffs = load_model3.dist_coeffs
        self.object_pts = load_model3.object_pts
        self.reprojectsrc = load_model3.reprojectsrc
        self.line_pairs = load_model3.line_pairs
        # --------------------------------------------#
        self.sep_time = 0
        self.save_type = 0
        self.s_500ms = 0
        self.count = 0
        self.down = 0
        self.emotion = 0
        self.w = 768
        self.h = 576
        self.iscamera = True
        self.ispause = True
        self.start = False
        self.save_file = None
        self.save_cur = 0
        self.save_img_state = False
        self.root = os.getcwd()
        self.videoBtn.clicked.connect(self.startVideo)
        self.camBtn.clicked.connect(self.startCamera)
        self.imgBtn.clicked.connect(self.img_detect)
        self.stopBtn.clicked.connect(self.stop)
        self.d_Btn.clicked.connect(self.pause)
        # ------------------------QTimer--------------------#
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.getFrame)

        self.timer_time = QtCore.QTimer()
        self.timer_time.timeout.connect(self.showTime)
        # --------------------------------------------------#
        self.frameToAnalyze = []

        Thread(target=self.frameAnalyzeThreadFunc, daemon=True).start()

    def setupUI(self):

        self.resize(1250, 750)

        self.setWindowTitle('恒星不见')

        # central Widget
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)
        centralWidget.setStyleSheet(qss.qss_background)
        # central Widget 里面的 主 layout
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)
        # 界面的上半部分 : 图形展示部分
        topLayout = QtWidgets.QHBoxLayout()

        self.label_treated = QtWidgets.QLabel("请选择输入源")
        self.label_treated.setAlignment(QtCore.Qt.AlignCenter)
        self.label_treated.setMinimumSize(768, 576)
        self.label_treated.setStyleSheet(
            'border:1px solid #D7E2F9;color:white;font-size:30px;')
        topLayout.addWidget(self.label_treated)
        mainLayout.addLayout(topLayout)

        # 界面下半部分： 输出框 和 按钮
        groupBox = QtWidgets.QGroupBox(self)

        bottomLayout = QtWidgets.QHBoxLayout(groupBox)
        self.labal_log = QtWidgets.QLabel("恒星不见，夜中星陨如雨")
        self.labal_log.setMinimumSize(300, 100)
        self.labal_log.setStyleSheet(qss.qss_label)
        infoLayout = QtWidgets.QVBoxLayout()
        self.info1 = QtWidgets.QLabel()
        self.info1.setMinimumSize(350, 50)
        self.info1.setStyleSheet(qss.qss_label_0)
        self.info2 = QtWidgets.QLabel()
        self.info2.setMinimumSize(350, 50)
        self.info2.setStyleSheet(qss.qss_label)
        infoLayout.addWidget(self.info1)
        infoLayout.addWidget(self.info2)
        bottomLayout.addWidget(self.labal_log)
        bottomLayout.addLayout(infoLayout)
        mainLayout.addWidget(groupBox)

        btnLayout = QtWidgets.QVBoxLayout()
        self.videoBtn = QtWidgets.QPushButton('🎞️视频文件')
        self.camBtn = QtWidgets.QPushButton('📹摄像头')
        self.imgBtn = QtWidgets.QPushButton('🆕图片')
        self.stopBtn = QtWidgets.QPushButton('🛑停止')

        aLayout = QtWidgets.QHBoxLayout()
        self.a_comboBox = QtWidgets.QComboBox()
        self.a_comboBox.addItem("")
        self.a_comboBox.addItem("")
        self.a_comboBox.addItem("")
        self.a_comboBox.addItem("")
        self.a_comboBox.setItemText(0, "间隔1s")
        self.a_comboBox.setItemText(1, "间隔2s")
        self.a_comboBox.setItemText(2,  "间隔5s")
        self.a_comboBox.setItemText(3, "间隔10s")
        self.a_comboBox.setStyleSheet(qss.qss_label)
        self.a_comboBox.setFixedWidth(280)
        self.a_label = QtWidgets.QLabel('设置间隔时间')
        bLayout = QtWidgets.QHBoxLayout()
        self.b_comboBox = QtWidgets.QComboBox()
        self.b_comboBox.addItem("")
        self.b_comboBox.addItem("")
        self.b_comboBox.addItem("")
        self.b_comboBox.setItemText(0, "保存数据,不保存图片")
        self.b_comboBox.setItemText(1, "保存数据,并保存图片")
        self.b_comboBox.setItemText(2,  "都不保存")
        self.b_comboBox.setStyleSheet(qss.qss_label)
        self.b_comboBox.setFixedWidth(280)
        self.b_label = QtWidgets.QLabel('设置保存类型')
        # --------------------------------------------#
        cLayout = QtWidgets.QHBoxLayout()
        self.c_lineEdit = QtWidgets.QLineEdit()
        self.c_lineEdit.setPlaceholderText('恒星不见')
        self.c_lineEdit.setFixedWidth(280)
        self.c_label = QtWidgets.QLabel('确定')
        dLayout = QtWidgets.QHBoxLayout()
        self.d_lineEdit = QtWidgets.QLineEdit()
        self.d_lineEdit.setPlaceholderText('恒星不见')
        self.d_lineEdit.setFixedWidth(280)
        self.d_Btn = QtWidgets.QPushButton()
        self.d_Btn.setMinimumWidth(100)
        self.d_Btn.setMaximumWidth(150)
        self.d_Btn.setText('暂停')

        aLayout.addWidget(self.a_comboBox)
        aLayout.addWidget(self.a_label)
        bLayout.addWidget(self.b_comboBox)
        bLayout.addWidget(self.b_label)
        cLayout.addWidget(self.c_lineEdit)
        cLayout.addWidget(self.c_label)
        dLayout.addWidget(self.d_lineEdit)
        dLayout.addWidget(self.d_Btn)
        dLayout.addStretch(1)
        setLayout = QtWidgets.QVBoxLayout()
        setLayout.addLayout(aLayout)
        setLayout.addLayout(bLayout)
        setLayout.addLayout(cLayout)
        setLayout.addLayout(dLayout)

        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.camBtn)
        btnLayout.addWidget(self.imgBtn)
        btnLayout.addWidget(self.stopBtn)
        self.videoBtn.setStyleSheet(qss.qss_btn)
        self.camBtn.setStyleSheet(qss.qss_btn)
        self.imgBtn.setStyleSheet(qss.qss_btn)
        self.stopBtn.setStyleSheet(qss.qss_btn)
        bottomLayout.addLayout(setLayout)
        bottomLayout.addLayout(btnLayout)

        self.c_lineEdit.setStyleSheet(qss.qss_lineEdit)
        self.d_lineEdit.setStyleSheet(qss.qss_lineEdit)
        self.a_label.setStyleSheet(qss.qss_label_min)
        self.b_label.setStyleSheet(qss.qss_label_min)
        self.c_label.setStyleSheet(qss.qss_label_min)
        self.d_Btn.setStyleSheet(qss.qss_btn)

        # ----------------------------------------#

    def resizeEvent(self, event):
        # 获取新的窗口大小
        new_size = event.size()
        # 更新标签的文本内容
        print(f"Window size: {new_size.width()} x {new_size.height()}")
        self.w = new_size.width()
        self.h = new_size.height()

    def getComboxBoxValue(self):
        a_select_value = self.a_comboBox.currentText()
        b_select_value = self.b_comboBox.currentText()
        if a_select_value == '间隔1s':
            self.sep_time = 1
        elif a_select_value == '间隔2s':
            self.sep_time = 2
        elif a_select_value == '间隔5s':
            self.sep_time = 5
        elif a_select_value == '间隔10s':
            self.sep_time = 10
        else:
            self.sep_time = 1
        if b_select_value == '保存数据,不保存图片':
            self.save_type = 0
        elif b_select_value == '保存数据,并保存图片':
            self.save_type = 1
        elif b_select_value == '都不保存':
            self.save_type = 2
        else:
            self.save_type = 0
        # QtWidgets.QMessageBox.information(self, "消息框", "%s," % (
        #     select_value,), QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

    def start_init(self):
        if self.timer_camera.isActive() is False:  # 若定时器未启动
            self.timer_camera.start(50)
        if self.timer_time.isActive() is False:  # 若定时器未启动
            self.timer_time.start(500)
        self.getComboxBoxValue()
        self.start = True
        self.save_cur = 0
        if (self.save_type == 0) or (self.save_type == 1):
            os.chdir(self.root)
            os.chdir('results')
            time = QtCore.QDateTime.currentDateTime().toString('yyyy-MM-dd-hh-mm-ss')
            path = time
            os.makedirs(path)
            self.save_file = open(path+'/data.csv', "a",
                                  encoding="utf-8", newline="")
            self.csv_writer = csv.writer(self.save_file)
            # 3. 构建列表头
            name = ['time', 'count', 'down', 'emotion']
            self.csv_writer.writerow(name)
        if self.save_type == 1:
            os.chdir(path)
            os.makedirs('img_with_label')

    def startCamera(self):
        # 在 windows上指定使用 cv2.CAP_DSHOW 会让打开摄像头快很多，
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("0号摄像头不能打开")
            return ()
        self.iscamera = True
        self.start_init()

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
            self.start_init()

    def getFrame(self):

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
        if not self.frameToAnalyze:
            self.frameToAnalyze.append(frame)

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

    def frameAnalyzeThreadFunc(self):

        while True:
            if not self.frameToAnalyze:
                time.sleep(0.01)
                continue

            frame = self.frameToAnalyze.pop(0)
            t1 = time.time()
            detections = api2.detect(frame, self.det_compiled_model_1, 1)[0]
            image_with_boxes, self.count, boxes = api2.draw_results(
                detections, frame, self.label_map_1)
            self.down = 0
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
                    self.down += 1
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
            fps = str(int(1000/ms))
            self.info1.setText(
                f"FPS:{fps}  COUNT:{str(self.count)} DOWN:{str(self.down)}\nRATE:{self.down/(self.count+0.001)*100:.1f}% ")
            # img = cv2.resize(image_with_boxes, (768, 576))
            img = image_with_boxes

            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            # self.textLog.append(str(count))
            qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0],
                                  QtGui.QImage.Format_RGBA8888)  # 变成QImage形式
            self.label_treated.setPixmap(
                QtGui.QPixmap.fromImage(qImage))
# -----------------------------------------------------------
            if self.save_img_state:
                cv2.putText(
                    img, f'FPS:{fps}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
                cv2.putText(
                    img, f'FACE:{self.count}__DOWN:{self.down}__per{self.down/(self.count+0.001)*100:.1f}% ', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 1)
                cv2.imwrite('img_with_label/'+self.timeDisplay +
                            '.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                self.save_img_state = False
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
            if w >= h:
                h = int(h*768/w)
                w = 768
            else:
                w = int(w*576/h)
                h = 576
            img = cv2.resize(img, (w, h))
            # 视频色彩转换回RGB，OpenCV images as BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = api.detect(
                img, self.det_compiled_model, 1)[0]
            image_with_boxes, count = api.draw_results(
                detections, img, self.label_map)
            cv2.putText(
                image_with_boxes, f'Face:{count}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            img = image_with_boxes
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0],
                                  QtGui.QImage.Format_RGBA8888)

            self.label_treated.setPixmap(
                QtGui.QPixmap.fromImage(qImage))

    def pause(self):
        if self.start:
            if self.ispause is True:
                self.ispause = False
                self.timer_camera.stop()
                self.timer_time.stop()
                self.d_Btn.setText('继续')
            else:
                self.ispause = True
                self.timer_camera.start(50)
                self.timer_time.start(500)
                self.d_Btn.setText('暂停')

    def showTime(self):
        # 获取系统当前时间
        time = QtCore.QDateTime.currentDateTime()
        self.timeDisplay = time.toString('yyyy-MM-dd-hh-mm-ss')
        self.info2.setText(time.toString('yyyy-MM-dd hh:mm:ss'))
        if (self.save_type == 0) or (self.save_type == 1):
            self.s_500ms += 1
            if self.s_500ms == self.sep_time*2:          # septime/500ms
                self.savadata()
                if self.save_type == 1:
                    self.save_img_state = True
                self.save_cur += 1
                self.s_500ms = 0

    def savadata(self):
        self.labal_log.setText('已记录'+str(self.save_cur)+'帧')
        z = [self.timeDisplay, self.count, self.down, 0]
        self.csv_writer.writerow(z)

    def stop(self):
        if self.start:
            self.start = False
            self.ispause = True
            self.d_Btn.setText('暂停')
            self.timer_camera.stop()  # 关闭定时器
            self.timer_time.stop()
            self.cap.release()  # 释放视频流
            self.label_treated.clear()  # 清空视频显示区域
            self.label_treated.setText('请选择输入源')
            time.sleep(0.01)
            self.label_treated.clear()  # 清空视频显示区域
            self.label_treated.setText('请选择输入源')
            if (self.save_type == 0) or (self.save_type == 1):
                self.save_file.close()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Com()
    window.show()
    app.exec()
