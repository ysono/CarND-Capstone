import glob
import cv2
import numpy as np

from lights_detector import Lights_Detector
from lights_classifier import Lights_Classifier

class Pipeline(object):
    
    def __init__(self):
        
        self.detector = Lights_Detector('detector_model/frozen_inference_graph_mobilenetV1.pb')
        self.classifier = Lights_Classifier('tl_classifier_real.h5')
    
    def detect(self,img):
        cropped_lights,_ = self.detector.detect_traffic_lights(img)
        
        return self.classifier.classify_traffic_lights(cropped_lights)
    


def load_test_data():
    
    test_img = []
    test_label = []
    test_file = []

    for img_class, directory in enumerate(['Red', 'Yellow', 'Green', 'Unknown']):
        for i, file_name in enumerate(glob.glob("/home/andcircle/FunDriving/Term3/Final_Proj/Ysono_Use_Data/data/real_training_data/{}/*.jpg".format(directory))):
            img = cv2.imread(file_name)   
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
      
            test_img.append(img)
            test_label.append(img_class)
            test_file.append(file_name)
     
    test_img = np.array(test_img)
    test_label = np.array(test_label)
    
    return test_img, test_label, test_file
    
    

test_img, test_label, test_file = load_test_data()
pipe = Pipeline()

counter = 0
total = len(test_label)

for i in range(total):
    p = pipe.detect(test_img[i])
    if test_label[i] == 0:
        r = 0
    else:
        r = 2
        
    if p == r:
        counter=counter+1
    else:
        print("Error File Name:{}".format(test_file[i]))

print("Total number of pic: {}".format(total))
print("Accuracy:{}".format(float(counter)/total))
        











