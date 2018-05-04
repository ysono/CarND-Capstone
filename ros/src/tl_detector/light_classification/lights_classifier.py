from keras.models import load_model
import time
import rospy
import numpy as np

class Lights_Classifier(object):
    def __init__(self,model_name):
        
        self.model = load_model(model_name)
        self.model.predict(np.zeros((1, 64, 32, 3)))
        
        
    def classify_traffic_lights(self,cropped_lights):
        
        red_num = 0;
        
        prediction=[]
        if len(cropped_lights)>0:
            t0 = time.time()
            rospy.logdebug("Shape of Classifier Input: {}".format(cropped_lights.shape))
            prediction = self.model.predict(cropped_lights,verbose=0)
            t1 = time.time()
            classify_time = (t1-t0)*1000
            rospy.logdebug("Classify time:{}".format(classify_time))
        
        for r in prediction:
            if r[0]>0.5:
                red_num += 1
        
        if (red_num>0 and red_num==len(cropped_lights)) or red_num>=2:
            return 0
        
        return 2     # for now, no red light is green light

    