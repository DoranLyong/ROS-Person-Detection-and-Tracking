"""
(ref) https://wikidocs.net/28
(ref) https://github.com/DoranLyong/yolov4-tiny-tflite-for-person-detection/blob/main/run_webcam_person_detector.py
"""
import core.utils as utils
from core.yolov4 import filter_boxes


class YOLO_INFER(object):
    def __init__(self, ConfigProto, InteractiveSession):
        self.config = ConfigProto()
        self.config.gpu_options.allow_growth = True 
        self.session = InteractiveSession(config=self.config)

    def model_inference(image_input, interpreter, input_details, output_details ):

        interpreter.set_tensor(input_details[0]['index'], image_input)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([FLAGS.size, FLAGS.size]))

        return  boxes, pred_conf 






class YOLO_CFG(object):
    def __init__(self, cfg):
        self.framework = cfg['YOLO4_TINY']['FRAMEWORK']
        self.weights = cfg['YOLO4_TINY']['WEIGHTS']
        self.size = cfg['YOLO4_TINY']['SIZE']
        self.tiny = cfg['YOLO4_TINY']['TINY']
        self.model = cfg['YOLO4_TINY']['MODEL']
        self.iou = cfg['YOLO4_TINY']['IoU']
        self.score = cfg['YOLO4_TINY']['SCORE']


