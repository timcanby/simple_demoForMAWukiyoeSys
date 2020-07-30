from keras.applications.vgg19 import VGG19
import glob
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
from keras import backend as K

import pickle

K.clear_session()
global base_model
base_model = VGG19(weights='imagenet')

model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
def pretreatment(ima):


    img_path = ima

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    block5_pool_features = model.predict(x)

    return block5_pool_features.flatten()