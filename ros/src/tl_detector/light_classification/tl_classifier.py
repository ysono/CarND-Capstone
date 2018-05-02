from styx_msgs.msg import TrafficLight
from lights_detector import Lights_Detector
from lights_classifier import Lights_Classifier
import rospy

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.detector = Lights_Detector('light_classification/frozen_inference_graph_mobilenetV1.pb')
        self.classifier = Lights_Classifier('light_classification/tl_classifier_simulator.h5')

    def get_classification(self, img):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        cropped_lights,_ = self.detector.detect_traffic_lights(img)
        
        return self.classifier.classify_traffic_lights(cropped_lights)

#         return TrafficLight.UNKNOWN
