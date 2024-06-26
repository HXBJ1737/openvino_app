from api.api import *
from openvino.runtime import Core
import cv2
import time
from PIL import Image
import numpy as np


def predict(model, cls_maps: dict, obj_path, video=False):
    """
    使用Open_Vino量化后预测(目标检测）
    :param model: xml模型（量化后的模型）
    :param cls_maps: 输入dict类型，{编号:对应标签}
    :param obj_path: 图片或视频路径，设置为'0'且cap为True的时候使用摄像头
    :param cap: 是否使用摄像头,bool类型
    :return: 图片模式返回预测后的图片，视频模式返回预测后的视频，摄像头模式返回摄像头录制的视频
    """
    label_map = cls_maps
    # 使用xml进行预测
    core = Core()
    det_ov_model = core.read_model(model)
    device = 'GPU'
    det_compiled_model = core.compile_model(det_ov_model, device)

    if video:
        print('open cam or video')
        cm = cv2.VideoCapture(obj_path)
        while True:
            a, frame = cm.read()
            if a:
                t1 = time.time()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = detect(frame, det_compiled_model, 1)[0]  # num
                image_with_boxes, count = draw_results(
                    detections, frame, label_map)
                t2 = time.time()
                ms = int((t2-t1)*1000)
                cv2.putText(
                    image_with_boxes, f'FPS:{int(1000/ms)}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(
                    image_with_boxes, f'FACE:{count}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                image_with_boxes = cv2.cvtColor(
                    image_with_boxes, cv2.COLOR_BGR2RGB)
                cv2.imshow('cm', image_with_boxes)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
                else:
                    continue
            else:
                print('摄像头未打开成功！')
                break
    else:
        print('图片模式或其他')
        frame = Image.open(obj_path)
        frame = np.array(frame)
        detections = detect(frame, det_compiled_model, 80)[0]
        image_with_boxes = draw_results(detections, frame, label_map)
        # cv2.imshow('images', image_with_boxes)
        # cv2.waitKey(0)
        img = Image.fromarray(image_with_boxes)
        img.show()
        # cv2.imwrite('result/001.jpg',image_with_boxes)


if __name__ == '__main__':
    cls_maps = ['face']
    predict('xml_models/yolov8n_100e.xml', cls_maps, 0, video=True)
