
from openvino.runtime import Core
import numpy as np
# import dlib
import cv2
from cv2 import dnn
from math import ceil

class model1:
    def __init__(self):
        self.model = 'xml_models/yolov8n_100e.xml'
        self.core = Core()
        self.det_ov_model = self.core.read_model(self.model)
        self.device = 'GPU'
        self.det_compiled_model = self.core.compile_model(
            self.det_ov_model, self.device)
        self.label_map = ['face']



class model2:
    def __init__(self):
        self.model = 'xml_models/yolov8s_nncf_int8.xml'
        self.core = Core()
        self.det_ov_model = self.core.read_model(self.model)
        self.device = 'GPU'
        self.det_compiled_model = self.core.compile_model(
            self.det_ov_model, self.device)
        self.label_map = ['person']


class model3:
    def __init__(self):
        # self.face_landmark_path = './models/dlib_models/shape_predictor_68_face_landmarks.dat'
        # self.predictor = dlib.shape_predictor(self.face_landmark_path)

        K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
             0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
             0.0, 0.0, 1.0]
        D = [7.0834633684407095e-002, 6.9140193737175351e-002,
             0.0, 0.0, -1.3073460323689292e+000]

        self.cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                                      [1.330353, 7.122144, 6.903745],
                                      [-1.330353, 7.122144, 6.903745],
                                      [-6.825897, 6.760612, 4.402142],
                                      [5.311432, 5.485328, 3.987654],
                                      [1.789930, 5.393625, 4.413414],
                                      [-1.789930, 5.393625, 4.413414],
                                      [-5.311432, 5.485328, 3.987654],
                                      [2.005628, 1.409845, 6.165652],
                                      [-2.005628, 1.409845, 6.165652],
                                      [2.774015, -2.080775, 5.048531],
                                      [-2.774015, -2.080775, 5.048531],
                                      [0.000000, -3.116408, 6.097667],
                                      [0.000000, -7.415691, 4.070434]])

        self.reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                        [10.0, 10.0, -10.0],
                                        [10.0, -10.0, -10.0],
                                        [10.0, -10.0, 10.0],
                                        [-10.0, 10.0, 10.0],
                                        [-10.0, 10.0, -10.0],
                                        [-10.0, -10.0, -10.0],
                                        [-10.0, -10.0, 10.0]])

        self.line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                           [4, 5], [5, 6], [6, 7], [7, 4],
                           [0, 4], [1, 5], [2, 6], [3, 7]]


class model4:
    def __init__(self):
        self.image_mean = np.array([127, 127, 127])
        self.image_std = 128.0
        self.iou_threshold = 0.3
        self.center_variance = 0.1
        self.size_variance = 0.2
        self.min_boxes = [
            [10.0, 16.0, 24.0],
            [32.0, 48.0],
            [64.0, 96.0],
            [128.0, 192.0, 256.0]
        ]
        self.strides = [8.0, 16.0, 32.0, 64.0]
        self.threshold = 0.5
        self.emotion_dict = {
            0: 'neutral',
            1: 'happiness',
            2: 'surprise',
            3: 'sadness',
            4: 'anger',
            5: 'disgust',
            6: 'fear'
        }
        self.model = cv2.dnn.readNetFromONNX(
            './models/RFB-320/emotion-ferplus-8.onnx')
        # Read the Caffe face detector.
        self.model_path = './models/RFB-320/RFB-320.caffemodel'
        self.proto_path = './models/RFB-320/RFB-320.prototxt'
        self.net = dnn.readNetFromCaffe(self.proto_path, self.model_path)
        self.input_size = [320, 240]
        self.width = self.input_size[0]
        self.height = self.input_size[1]
        self.priors = self.define_img_size(self.input_size)
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


load_model1 = model1()
load_model2 = model2()
load_model3 = model3()
load_model4 = model4()
