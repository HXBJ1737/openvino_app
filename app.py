from api import api
import numpy as np
from PIL import Image
from openvino.runtime import Core
from threading import Thread
import time
import os
import cv2
from PyQt5 import QtCore, QtWidgets, QtGui
from ultralytics import YOLO
from count_face import Face_detect
from count_person import Person_detect
from head_pose import Head_detect
from expression_detect import Expression_detect
from face_id_app.face_id_detect import CoreUI
from com import Com
import sys
from PyQt5.QtWidgets import QTabWidget, QApplication
import qss


class App(QTabWidget):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)
        self.resize(1250, 750)
        self.setMinimumSize(1200, 700)
        self.tab1 = Face_detect()
        self.tab2 = Person_detect()
        self.tab3 = Head_detect()
        self.tab4 = Expression_detect()
        self.tab5 = Com()
        self.addTab(self.tab1, "人脸检测计数")
        self.addTab(self.tab2, "人体检测计数")
        self.addTab(self.tab3, "头部姿态检测")
        self.addTab(self.tab4, "面部表情检测")
        self.addTab(self.tab5, "综合应用")
        self.setStyleSheet(qss.qss_tab+qss.qss_tab_selected +
                           qss.qss_tab_hover+qss.qss_tab_width)

    def resizeEvent(self, event):
        # 获取新的窗口大小
        new_size = event.size()
        # 更新标签的文本内容
        print(f"Window size: {new_size.width()} x {new_size.height()}")
        qss_tab_width = qss.qss_tab_width.replace(
            '100px', f"{new_size.width()/5.45}px")
        self.setStyleSheet(qss_tab_width+qss.qss_tab+qss.qss_tab_selected +
                           qss.qss_tab_hover)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = App()
    demo.show()
    sys.exit(app.exec_())
