import tensorflow as tf
import numpy as np
from PIL import ImageColor
import time
import cv2

class Lights_Detector(object):

    #
    # Utility funcs
    #
    
    def __init__(self,model_name):
        
        self.confidence_cutoff = 0.2
        self.traffic_lights_class_id=10   # id for traffic lights in labeled COCO data
        
        self.detection_graph = None
        self.image_input = None
        self.detection_boxes = None
        self.detection_scores = None
        self.detection_classes = None
        self.sess = None
        
        self.classifier_w = 32
        self.classifier_h = 64
        
#         self.detection_graph = self.load_graph('detector_model/frozen_inference_graph.pb')
        self.load_graph(model_name)
    
            # The classification of the object (integer id).
    
    def filter_boxes(self, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if classes[i] == self.traffic_lights_class_id and scores[i] >= self.confidence_cutoff:
                idxs.append(i)
        
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes
    
    def to_image_coords(self,boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].
        
        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
        
        return box_coords
    
    
    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        
        # We load the protobuf file from the disk and parse it to retrieve the 
        # unserialized graph_def
        with tf.gfile.GFile(graph_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
     
        # Then, we import the graph_def into a new Graph and returns it 
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="")
                 
            # The input placeholder for the image.
            # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
            self.image_input = graph.get_tensor_by_name('image_tensor:0')
     
            # Each box represents a part of the image where a particular object was detected.
            self.detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
     
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.detection_scores = graph.get_tensor_by_name('detection_scores:0')
     
            # The classification of the object (integer id).
            self.detection_classes = graph.get_tensor_by_name('detection_classes:0')
             
            self.detection_graph = graph
        
        #session should be saved for future use, otherwise will affect efficiency!!!
        self.sess = tf.Session(graph=self.detection_graph)
            
    
    def detect(self, img):
      
#         with tf.Session(graph=self.detection_graph) as sess: 
        with self.detection_graph.as_default():               
            # Actual detection.
            t0 = time.time()
            (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                                feed_dict={self.image_input: img})
            t1 = time.time()
            detect_time = (t1 - t0) * 1000
    
            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
    
            return boxes, scores, classes, detect_time

    

    
    def detect_traffic_lights(self,img):
        
        width = img.shape[1]
        height = img.shape[0]

        img = np.expand_dims(np.asarray(img, dtype=np.uint8), 0)
        boxes,scores,classes,detect_time = self.detect(img)
        
        print("Detect time:{}".format(detect_time))

        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores,classes = self.filter_boxes(boxes, scores, classes)
        box_coords = self.to_image_coords(boxes, height, width)
        crop_imgs = []
#         print(box_coords)
        for box in box_coords:
            top,left,bottom,right = box
            crop_img = img[0,int(top):int(bottom), int(left):int(right)]
            # prepare the format for classifer
            crop_img = cv2.resize(crop_img,(self.classifier_w,self.classifier_h))
            crop_imgs.append(crop_img/255.)
        return np.asarray(crop_imgs), img
        
    
    
    
