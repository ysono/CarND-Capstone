from styx_msgs.msg import TrafficLight
from lights_detector import Lights_Detector
from lights_classifier import Lights_Classifier
import rospy

class TLClassifier(object):
    def __init__(self):
        self.detector = Lights_Detector(rospy.get_param('~detector_model_path'))
        self.classifier = Lights_Classifier(rospy.get_param('~classifier_model_path'))

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
