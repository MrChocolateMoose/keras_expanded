from keras import backend as K
from keras.callbacks import Callback
from keras.utils.np_utils import to_categorical
from keras.layers import Convolution2D

import numpy as np
from sklearn import preprocessing
from skimage.transform import resize
from skimage.io import imsave
import matplotlib as plt
from matplotlib.pyplot import imshow
from matplotlib.pyplot import show
from scipy.misc import imresize
from PIL import Image
import os

from .util import stitch_pil_images, stitch_pil_image_from_values

class CAMVisualizationCallback(Callback):

    def __init__(self, generator, directory, cam_funcs = None, base_file_name = 'cam', model = None, **kwargs):
        self.generator = generator
        self.directory = directory
        self.base_file_name = base_file_name
        self.__is_training = False
        self.__batch_number = 0

        if cam_funcs is None:
            cam_funcs = [lambda layer: issubclass(type(layer), Convolution2D)]

        self.cam_funcs = cam_funcs

        self.cam_threshold_pct = kwargs.get("cam_threshold_pct", 0.5)
        self.cam_transparency_pct = kwargs.get("cam_transparency_pct", 0.75)
        self.image_border_thickness = kwargs.get("image_border_thickness", 5)
        self.image_margin = kwargs.get("image_margin", 5)
        self.nb_classes = kwargs.get("nb_classes", 2)
        self.every_n_epochs = kwargs.get("every_n_epochs", None)
        self.skip_initial_epoch = kwargs.get("skip_initial_epoch", False)
        self.model = model

        # make sure the directory exists before writing filter images to it
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def visualize(self, x, y):
        # TODO: make batch of size (1,1)
        pass

    def visualize_batch(self, batch_x = None, batch_y = None):

        if batch_x == None or batch_y == None:
            print("batch index: %s batch_size: %s" % (str(self.generator.batch_index), str(self.generator.batch_size)))
            batch_x, batch_y = self.generator.next()

        if K.image_dim_ordering() == 'th':
            width = batch_x.shape[2]
            height = batch_x.shape[3]
        else:
            width = batch_x.shape[1]
            height = batch_x.shape[2]

        def global_average_pooling(x):
            return np.mean(x, axis=(2,3))

        for current_layer in (layer for layer in reversed(self.model.layers) if any(cam_func(layer) for cam_func in self.cam_funcs)):

            layer_name = current_layer.name

            get_output = K.function([self.model.layers[0].input, K.learning_phase()], [current_layer.get_output_at(0)])
            [conv_outputs_batch] = get_output([batch_x, 1])

            class_weights = global_average_pooling(conv_outputs_batch)

            for k in range(1): # for each filter in filters
            #for k in range(class_weights.shape[1]): # for each filter in filters

                pil_img_list = []
                for i in range(batch_x.shape[0]): # for each img in batch
                    conv_outputs = conv_outputs_batch[i, :, :, :]

                    # Create the class activation map.
                    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[1:3])

                    #class_weights = np.sum(class_weights, axis=1, keepdims=True)
                    #class_weights_k = class_weights[:, k]

                    # 32 samples
                    # 8 filters

                    class_weights_k = np.sum(class_weights, axis=0) # 1 , 8 or 8?

                    for j, w in enumerate(class_weights_k):
                        cam += w * conv_outputs[j, :, :]

                    # scale cam from 0 to 1 and resize to same dims as batch images
                    cam = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(cam)
                    cam = imresize(cam, (width, height))

                    # apply color map to cam, clip values under threshold, apply transparency to layer
                    cm_cam = plt.cm.jet(cam) # TODO: Allow COLORMAP to be configured via string paramter
                    #cm_cam = plt.cm.Blues(cam) # TODO: Allow COLORMAP to be configured via string paramter

                    cm_cam[np.where(cam < self.cam_threshold_pct)] = 0
                    cam = np.uint8(cm_cam * 255)
                    cam[:,:,3] = np.uint((1-self.cam_transparency_pct)*255)
                    # get as PIL RGBA image to composite
                    cam_pil_img_rgba = Image.fromarray(cam, mode='RGBA')
                    cam_pil_img_rgba = cam_pil_img_rgba.transpose(Image.ROTATE_90)

                    # (color_channels, width, height) -> (width, height, color_channels)
                    x = np.uint8(np.rollaxis(batch_x[i], 0, 3) * 255)

                    # get as PIL RGBA image to composite
                    if x.shape[2] == 4:
                        x_pil_img_rgba = Image.fromarray(x, mode='RGBA')
                    else:
                        x_pil_img_rgba = Image.fromarray(x, mode='RGB').convert(mode='RGBA')


                    x_pil_img_rgba = x_pil_img_rgba.transpose(Image.ROTATE_90)


                    # composite cam layer over image layer and save
                    pil_img = Image.alpha_composite(x_pil_img_rgba, cam_pil_img_rgba)
                    #pil_img = pil_img.transpose(Image.ROTATE_90)

                    pil_img_list.append(pil_img)

                try:

                    accuracy_img_size = (pil_img_list[0].size[0] + self.image_border_thickness * 2,
                                         pil_img_list[0].size[1] + self.image_border_thickness * 2)
                    accuracy_list = self.get_accuracy_list(batch_x, batch_y).astype(dtype=np.int32)


                    stitched_pil_cam_images = stitch_pil_images(pil_img_list, margin = self.image_margin, border_thickness=self.image_border_thickness)
                    stitched_pil_accuracy_images = stitch_pil_image_from_values(accuracy_list, accuracy_img_size, margin=self.image_margin, cmap_name='seismic_r')

                    stitched_pil_output_image = Image.alpha_composite(stitched_pil_accuracy_images, stitched_pil_cam_images)
                except:
                    stitched_pil_output_image = stitch_pil_images(pil_img_list, margin=self.image_margin,
                                                                border_thickness=self.image_border_thickness)


                if self.__is_training:
                    full_file_name = os.path.join(self.directory, self.base_file_name +  '_epoch_' + str(self.epoch) + "_" +layer_name + '_k_' + str(k) +'.png')
                else:
                    full_file_name = os.path.join(self.directory, self.base_file_name + '_batch_' + str(self.__batch_number) + '.png')
                    self.__batch_number = self.__batch_number + 1


                stitched_pil_output_image.save(full_file_name)

        return True

    def on_train_begin(self, logs={}):
        self.__is_training = True
        pass

    def on_train_end(self, logs={}):
        self.__is_training = False
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):

        self.epoch = epoch

        if self.every_n_epochs is not None and (epoch % self.every_n_epochs != 0 or (self.skip_initial_epoch == True and epoch == 0)):
            return

        self.visualize_batch()

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    #TODO: bugged for certain metrics
    def get_accuracy_list(self, batch_x, batch_y):

        y_pred = self.model.predict_on_batch(batch_x)
        y_true = batch_y
        #y_true = to_categorical(batch_y, nb_classes=self.nb_classes)

        accuracy_list = np.argmax(y_pred, axis = 1) == np.argmax(y_true, axis = 1)

        return accuracy_list

        #loss_function_name = 'sparse_categorical_crossentropy'

        #if loss_function_name == 'sparse_categorical_crossentropy':


        '''
        if metric == 'accuracy' or metric == 'acc':
            # custom handling of accuracy (because of class mode duality)
            output_shape = self.internal_output_shapes[i]
            if output_shape[-1] == 1 or self.loss_functions[i] == objectives.binary_crossentropy:
                # case: binary accuracy
                self.metrics_tensors.append(metrics_module.binary_accuracy(y_true, y_pred))
            elif self.loss_functions[i] == objectives.sparse_categorical_crossentropy:
                # case: categorical accuracy with sparse targets
                self.metrics_tensors.append(
                    metrics_module.sparse_categorical_accuracy(y_true, y_pred))
            else:
                # case: categorical accuracy with dense targets
                self.metrics_tensors.append(metrics_module.categorical_accuracy(y_true, y_pred))
            if len(self.output_names) == 1:
                self.metrics_names.append('acc')
            else:
                self.metrics_names.append(self.output_layers[i].name + '_acc')
        self.metrics_names.append(self.output_layers[i].name + '_' + metric_fn.__name__)
        '''