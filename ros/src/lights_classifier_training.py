import cv2
import glob
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from keras.utils.np_utils import to_categorical
from keras import losses, optimizers, regularizers


X_train = []
x_label = []
for img_class, directory in enumerate(['Red', 'Yellow', 'Green', 'NoTrafficLight']):
    for i, file_name in enumerate(glob.glob("simulator_lights/{}/*.png".format(directory))):
#     for i, file_name in enumerate(glob.glob("/home/andcircle/FunDriving/Term3/Final_Proj/tl_classifier_exceptsmall/real/{}/*.png".format(directory))):
        img = cv2.imread(file_name)
  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
        resized = cv2.resize(img, (32,64))
  
        X_train.append(resized/255.)
        x_label.append(img_class)
  
  
X_train = np.array(X_train)
x_label = np.array(x_label)
  
categorical_labels = to_categorical(x_label)
  
  
num_classes = 4
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 32, 3), padding='same', activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(2,2))
Dropout(0.5)
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(2,2))
Dropout(0.5)
model.add(Flatten())
  
model.add(Dense(8, activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(num_classes, activation='softmax'))
  
loss = losses.categorical_crossentropy
optimizer = optimizers.Adam()
  
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
  
model.fit(X_train, categorical_labels, batch_size=32, epochs=10, verbose=True, validation_split=0.1, shuffle=True)
score = model.evaluate(X_train, categorical_labels, verbose=0)
print(score)
  
model.save('tl_classifier_simulator.h5')
# model.save('tl_classifier_real.h5')
#--------------------------------------------------------------- model.summary()





