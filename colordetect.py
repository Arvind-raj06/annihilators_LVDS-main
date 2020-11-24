import cv2
import numpy as np
from keras import models
def cdc(x_test):
    network = models.load_model("vehicle_color_haze_free_model.h5")
    network.load_weights("vehicle_color_haze_free_model.h5")
    d={'black':0,'blue':1,'cyan':2,'gray':3,'green':4,'red':5,'white':6,'yellow':7}
    d_b={0: 'black',1:'blue',2:'cyan',3:'gray',4:'green',5:'red',6:'white',7:'yellow'}
    layer_outputs = [layer.output for layer in network.layers[:12]]
    activation_model = models.Model(inputs=network.input, outputs=layer_outputs)
    x_test=cv2.imread('DVLA-number-plates-2017-67-new-car-847566.jpg')
    resized = cv2.resize(x_test,(100,100), interpolation = cv2.INTER_AREA)
    activations = activation_model.predict(resized.reshape(1,100,100,3)) # Returns a list of five Numpy arrays: one array per layer activation
    return (d_b[np.argmax(activations[len(activations)-1])])