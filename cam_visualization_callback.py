from keras import backend as K
from keras.callbacks import Callback

import numpy as np
from sklearn import preprocessing
from skimage.transform import resize
from skimage.io import imsave
import matplotlib as plt
from matplotlib.pyplot import imshow
from matplotlib.pyplot import show
from scipy.misc import imresize
from PIL import Image

from .util import stitch_pil_images

class CAMVisualizationCallback(Callback):

    def __init__(self, generator, dense_softmax_layer_name = None, last_conv_layer_name = None):
        self.generator = generator
        self.dense_softmax_layer_name = dense_softmax_layer_name
        self.last_conv_layer_name = last_conv_layer_name
        self.cam_threshold_pct = 0.5
        self.cam_transparency_pct = 0.75

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):

        batch_x, batch_y = self.generator.next()

        # TODO: depends on 'tf' or 'th'
        width = batch_x.shape[2]
        height = batch_x.shape[3]

        dense_softmax_layer = self.model.get_layer(self.dense_softmax_layer_name)
        dense_softmax_layer_weights = dense_softmax_layer.get_weights()
        class_weights = dense_softmax_layer_weights[0]

        last_conv_layer = self.model.get_layer(self.last_conv_layer_name)

        get_output = K.function([self.model.layers[0].input, K.learning_phase()], [last_conv_layer.output, dense_softmax_layer.output])
        [conv_outputs_batch, predictions] = get_output([batch_x, 1])

        pil_img_list = []
        for i in range(batch_x.shape[0]):
            conv_outputs = conv_outputs_batch[i, :, :, :]

            # Create the class activation map.
            cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[1:3])
            for j, w in enumerate(class_weights[:, 1]):
                cam += w * conv_outputs[j, :, :]

            # scale cam from 0 to 1 and resize to same dims as batch images
            cam = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(cam)
            cam = imresize(cam, (height,width))

            # apply color map to cam, clip values under threshold, apply transparency to layer
            cm_cam = plt.cm.jet(cam)
            cm_cam[np.where(cam < self.cam_threshold_pct)] = 0
            cam = np.uint8(cm_cam * 255)
            cam[:,:,3] = np.uint((1-self.cam_transparency_pct)*255)
            # get as PIL RGBA image to composite
            cam_pil_img_rgba = Image.fromarray(cam, mode='RGBA')

            # (3, width, height) -> (width, height, 3)
            x = np.uint8(np.rollaxis(batch_x[i], 0, 3) * 255)
            # get as PIL RGBA image to composite
            x_pil_img_rgba = Image.fromarray(x, mode='RGB').convert(mode='RGBA')

            # composite cam layer over image layer and save
            pil_img = Image.alpha_composite(x_pil_img_rgba, cam_pil_img_rgba)
            pil_img_list.append(pil_img)

        stitched_pil_images = stitch_pil_images(pil_img_list)

        stitched_pil_images.save('images/epoch_%d_cam.png' % epoch)

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        #imshow(self.model.output)
        #show(True)
        pass

    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass