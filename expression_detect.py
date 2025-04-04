from ultralytics import YOLO
from PyQt5 import QtCore, QtWidgets, QtGui
import cv2
import os
import time
from threading import Thread
import sys
from cv2 import dnn
from math import ceil
from PIL import Image
import numpy as np
from api import api2
import qss
from load import load_model4, load_model1
# 不然每次YOLO处理都会输出调试信息
os.environ['YOLO_VERBOSE'] = 'False'



class Expression_detect(QtWidgets.QMainWindow):

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
        # ------------------------------------------------------
        self.det_compiled_model = load_model1.det_compiled_model
        self.label_map = load_model1.label_map
        self.image_mean = load_model4.image_mean
        self.image_std = load_model4.image_std
        self.iou_threshold = load_model4.iou_threshold
        self.center_variance = load_model4.center_variance
        self.size_variance = load_model4.size_variance
        self.min_boxes = load_model4.min_boxes
        self.strides = load_model4.strides
        self.threshold = load_model4.threshold
        self.emotion_dict = load_model4.emotion_dict
        self.model = load_model4.model

        self.model_path = load_model4.model_path
        self.proto_path = load_model4.proto_path
        self.net = load_model4.net
        self.input_size = load_model4.input_size
        self.width = load_model4.width
        self.height = load_model4.height
        self.priors = load_model4.priors
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
# ------------------------------------------------------------------------

    def define_img_size(self, image_size):
        shrinkage_list = []
        feature_map_w_h_list = []
        for size in image_size:
            feature_map = [int(ceil(size / stride)) for stride in self.strides]
            feature_map_w_h_list.append(feature_map)

        for i in range(0, len(image_size)):
            shrinkage_list.append(self.strides)
        priors = self.generate_priors(
            feature_map_w_h_list, shrinkage_list, image_size, self.min_boxes
        )
        return priors

    def generate_priors(self,
                        feature_map_list, shrinkage_list, image_size, min_boxes
                        ):
        priors = []
        for index in range(0, len(feature_map_list[0])):
            scale_w = image_size[0] / shrinkage_list[0][index]
            scale_h = image_size[1] / shrinkage_list[1][index]
            for j in range(0, feature_map_list[1][index]):
                for i in range(0, feature_map_list[0][index]):
                    x_center = (i + 0.5) / scale_w
                    y_center = (j + 0.5) / scale_h

                    for min_box in min_boxes[index]:
                        w = min_box / image_size[0]
                        h = min_box / image_size[1]
                        priors.append([
                            x_center,
                            y_center,
                            w,
                            h
                        ])
        print("priors nums:{}".format(len(priors)))
        return np.clip(priors, 0.0, 1.0)

    def hard_nms(self, box_scores, iou_threshold, top_k=-1, candidate_size=200):
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        indexes = np.argsort(scores)
        indexes = indexes[-candidate_size:]
        while len(indexes) > 0:
            current = indexes[-1]
            picked.append(current)
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = self.iou_of(
                rest_boxes,
                np.expand_dims(current_box, axis=0),
            )
            indexes = indexes[iou <= iou_threshold]
        return box_scores[picked, :]

    def area_of(self, left_top, right_bottom):
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]

    def iou_of(self, boxes0, boxes1, eps=1e-5):
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = self.area_of(overlap_left_top, overlap_right_bottom)
        area0 = self.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    def predict(self,
                width,
                height,
                confidences,
                boxes,
                prob_threshold,
                iou_threshold=0.3,
                top_k=-1
                ):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate(
                [subset_boxes, probs.reshape(-1, 1)], axis=1
            )
            box_probs = self.hard_nms(box_probs,
                                      iou_threshold=iou_threshold,
                                      top_k=top_k,
                                      )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return (
            picked_box_probs[:, :4].astype(np.int32),
            np.array(picked_labels),
            picked_box_probs[:, 4]
        )

    def convert_locations_to_boxes(self, locations, priors, center_variance,
                                   size_variance):
        if len(priors.shape) + 1 == len(locations.shape):
            priors = np.expand_dims(priors, 0)
        return np.concatenate([
            locations[..., :2] * center_variance *
            priors[..., 2:] + priors[..., :2],
            np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
        ], axis=len(locations.shape) - 1)

    def center_form_to_corner_form(self, locations):
        return np.concatenate(
            [locations[..., :2] - locations[..., 2:] / 2,
             locations[..., :2] + locations[..., 2:] / 2],
            len(locations.shape) - 1
        )

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
            start_time = time.time()
            detections = api2.detect(frame, self.det_compiled_model, 1)[0]
            image_with_boxes, count, boxes = api2.draw_results(
                detections, frame, self.label_map)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            handle_face = 0
            for idx, (*xyxy, conf, lbl) in enumerate(boxes):
                # handle_face += 1
                # if handle_face > 2:
                #     break
                x1 = int(xyxy[0])
                y1 = int(xyxy[1])
                x2 = int(xyxy[2])
                y2 = int(xyxy[3])
                w = x2 - x1
                h = y2 - y1
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                resize_frame = cv2.resize(
                    gray[y1:y1 + h, x1:x1 + w], (64, 64)
                )
                resize_frame = resize_frame.reshape(1, 1, 64, 64)
                self. model.setInput(resize_frame)
                output = self.model.forward()
                pred = self.emotion_dict[list(output[0]).index(max(output[0]))]
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (215, 5, 247), 2, lineType=cv2.LINE_AA)
                cv2.putText(frame, pred, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (215, 5, 247), 2, lineType=cv2.LINE_AA)
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            print(f"FPS: {fps:.1f}")

            cv2.putText(
                frame, f'FPS:{int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
# ----------------------------------------------------
            # self.textLog.append(str(count))
            qImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
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
            frame = cv2.imread(file_path)
            w = frame.shape[1]
            h = frame.shape[0]
            if w >= h:
                h = int(h*768/w)
                w = 768
            else:
                w = int(w*576/h)
                h = 576
            frame = cv2.resize(frame, (w, h))
            img_ori = frame
            # print("frame size: ", frame.shape)
            rect = cv2.resize(img_ori, (self.width, self.height))
            rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
            self.net.setInput(dnn.blobFromImage(
                rect, 1 / self.image_std, (self.width, self.height), 127)
            )
            boxes, scores = self.net.forward(["boxes", "scores"])
            boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
            scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
            boxes = self.convert_locations_to_boxes(
                boxes, self.priors, self.center_variance, self.size_variance
            )
            boxes = self.center_form_to_corner_form(boxes)
            boxes, labels, probs = self.predict(
                img_ori.shape[1],
                img_ori.shape[0],
                scores,
                boxes,
                self.threshold
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for (x1, y1, x2, y2) in boxes:
                w = x2 - x1
                h = y2 - y1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                resize_frame = cv2.resize(
                    gray[y1:y1 + h, x1:x1 + w], (64, 64)
                )
                resize_frame = resize_frame.reshape(1, 1, 64, 64)
                self. model.setInput(resize_frame)
                output = self.model.forward()

                pred = self.emotion_dict[list(output[0]).index(max(output[0]))]
                cv2.rectangle(
                    img_ori,
                    (x1, y1),
                    (x2, y2),
                    (215, 5, 247),
                    2,
                    lineType=cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    pred,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (215, 5, 247),
                    2,
                    lineType=cv2.LINE_AA
                )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            qImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
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
    window = Expression_detect()
    window.show()
    app.exec()
