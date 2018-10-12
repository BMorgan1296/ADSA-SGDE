import cv2
import numpy as np
#from model.augmentations import randomHueSaturationValue, randomShiftScaleRotate, randomHorizontalFlip
import model.u_net as unet
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = unet.get_unet_128()
#model.summary()

SIZE = (256, 256)

def validation_process(data):
    img, mask = data
    img = img[:,:,:3]
    mask = mask[:, :, :3]
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    fix_mask(mask)
    img = cv2.resize(img, SIZE)
    mask = cv2.resize(mask, SIZE)
    fix_mask(mask)
    img = img/255.
    mask = mask/255.
    mask = np.expand_dims(mask, axis=2)
    return (img, mask)