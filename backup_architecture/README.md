
This method is using 2 seperate DNNs, responsible for detecting traffic lights and recognizing the color of traffic lights respectively.  

A pre-trained model from model-zoo is used for traffic ligts detection, since there's no need to train the model, we can simply switch to any model we wanna try.

A second DNN is built to recognize the color of traffic lights, it's relatively small, so the training process is very fast. Training data can be found here https://carnd.slack.com/files/U6WJWH7S7/F836LL4G1/tl_classifier_data.zip  
