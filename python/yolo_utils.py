"""
(ref) https://wikidocs.net/28
(ref) https://github.com/DoranLyong/yolov4-tiny-tflite-for-person-detection/blob/main/run_webcam_person_detector.py
"""

import time 

import cv2
import numpy as np
import matplotlib.pyplot as plt
from colorama import Back, Style
import tensorflow as tf 

import core.utils as utils
from core.config import cfg
from core.yolov4 import filter_boxes
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet



#%%
class YOLO_INFER(object):
    def __init__(self, ConfigProto, InteractiveSession, FLAGS):
        print(f"{Back.GREEN}YOLO init...{Style.RESET_ALL}")
        self.config = ConfigProto()
        self.config.gpu_options.allow_growth = True 
        self.session = InteractiveSession(config=self.config)


        """ setup yolov4-tiny model 
        """
        self.yolo_interpreter = tf.lite.Interpreter(model_path=FLAGS.detector_weights)
        self.yolo_interpreter.allocate_tensors()
        self.input_details = self.yolo_interpreter.get_input_details()
        self.output_details = self.yolo_interpreter.get_output_details()

        self.input_size = FLAGS.size  # image resizing
        self.iou = FLAGS.iou
        self.score = FLAGS.score
        


        """ setup DeepSORT model 
        """
        self.encoder = gdet.create_box_encoder(FLAGS.tracker_weights, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", FLAGS.max_cosine_distance, FLAGS.nn_budget) # calculate cosine distance metric
        self.DeepSORT_tracker = Tracker(self.metric ) # initialize tracker
        self.count = FLAGS.count
        self.info = FLAGS.info
        self.dont_show = FLAGS.dont_show
        self.nms_max_overlap = FLAGS.nms_max_overlap 



    def model_inference(self, image_input, interpreter, input_details, output_details ):

        interpreter.set_tensor(input_details[0]['index'], image_input)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([self.input_size, self.input_size]))

        return  boxes, pred_conf 


    def __call__(self, input_frame):
        image_data = cv2.resize(input_frame, (self.input_size, self.input_size))
        image_data = image_data / 255. # normalization ; [0, 255] -> [0, 1]
        image_data = image_data[np.newaxis, ...].astype(np.float32)


        prev_time = time.time()  # stop-watch on 


        """ detection inference 
        """
        boxes, pred_conf =  self.model_inference(image_data, self.yolo_interpreter, self.input_details, self.output_details)

        """ post-processing 
        """
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class = 50,
            max_total_size = 50,
            iou_threshold = self.iou,
            score_threshold=self.score
            ) 


        
        """ Convert data to numpy arrays and slice out unused elements
        """
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores_npy = scores.numpy()[0]
        scores_npy = scores_npy[0:int(num_objects)]
        classes_npy = classes.numpy()[0]
        classes_npy = classes_npy[0:int(num_objects)]

#        print(f"The number of objects: {num_objects}")

        """ format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        """
        original_h, original_w, _ = input_frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)        


        """ store all predictions in one parameter for simplicity when calling functions
        """
        pred_bbox = [bboxes, scores_npy, classes_npy , num_objects]


        """ read in all class names from config
        """
        class_names = utils.read_class_names(cfg.YOLO.CLASSES) 



        """loop through objects and use class index to get class name, allow only classes in allowed_classes list
        """
        names = []

        for i in range(num_objects):
            cls_indx = int(classes_npy[i])
            cls_name = class_names[cls_indx]

            names.append(cls_name)

        names = np.array(names)
        count = len(names)


        """ tracking inference 
        """

        if self.count:
            cv2.putText(input_frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
#            print(f"Objects being tracked: {count}")
            

        # encode yolo detections and feed to tracker
        features = self.encoder(input_frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores_npy, names, features)]


        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]


        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices] 


        # Call the tracker
        self.DeepSORT_tracker.predict()
        self.DeepSORT_tracker.update(detections)

        # update tracks
        bbox_list = []

        for track in self.DeepSORT_tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()

            """ Draw bbox on screen 
            """
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(input_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(input_frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(input_frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            bbox_list.append((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))) # (x1, y1, x2, y2)


            # if enable info flag then print details about each track
            if self.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - prev_time) # stop-watch off 
#        print(f"FPS: {fps:.2f}")        

        result = np.asarray(input_frame)
        result = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR)         


        if not self.dont_show :    
            cv2.imshow("Output", result)   


        return bbox_list
            




#%%
class CFG_FLAGS(object):
    def __init__(self, cfg):
        self.framework = cfg['YOLO4_TINY']['FRAMEWORK']
        self.detector_weights = cfg['YOLO4_TINY']['WEIGHTS']
        self.size = cfg['YOLO4_TINY']['SIZE']
        self.tiny = cfg['YOLO4_TINY']['TINY']
        self.model = cfg['YOLO4_TINY']['MODEL']
        self.iou = cfg['YOLO4_TINY']['IoU']
        self.score = cfg['YOLO4_TINY']['SCORE']
        
        # DeepSORT tracker 
        self.tracker_weights = cfg['DeepSORT']['WEIGHTS']
        self.max_cosine_distance = cfg['DeepSORT']['MAX_COSINE_DISTANCE']
        self.nn_budget = cfg['DeepSORT']['NN_BUDGET']
        self.nms_max_overlap = cfg['DeepSORT']['NMS_MAX_OVERLAP']
        self.count = cfg['DeepSORT']['COUNT']
        self.info = cfg['DeepSORT']['INFO']
        self.dont_show = cfg['DeepSORT']['DONT_SHOW']


