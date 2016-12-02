from keras import backend as K
from keras.callbacks import Callback
from keras.utils.np_utils import to_categorical

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

    def __init__(self, generator, directory, dense_softmax_layer_name = None, last_conv_layer_name = None, base_file_name = 'cam', **kwargs):
        self.generator = generator
        self.directory = directory
        self.base_file_name = base_file_name
        self.dense_softmax_layer_name = dense_softmax_layer_name
        self.last_conv_layer_name = last_conv_layer_name

        self.cam_threshold_pct = kwargs.get("cam_threshold_pct", 0.5)
        self.cam_transparency_pct = kwargs.get("cam_transparency_pct", 0.75)
        self.image_border_thickness = kwargs.get("image_border_thickness", 5)
        self.image_margin = kwargs.get("image_margin", 5)
        self.nb_classes = kwargs.get("nb_classes", 2)
        self.every_n_epochs = kwargs.get("every_n_epochs", None)
        self.skip_initial_epoch = kwargs.get("skip_initial_epoch", False)

        # make sure the directory exists before writing filter images to it
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def on_train_begin(self, logs={}):
        
        pass

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):

        if self.every_n_epochs is not None and (epoch % self.every_n_epochs != 0 or (self.skip_initial_epoch == True and epoch == 0)):
            return

        batch_x, batch_y = self.generator.next()

        # TODO: depends on 'tf' or 'th'
        width = batch_x.shape[2]
        height = batch_x.shape[3]

        def global_average_pooling(x):
            return np.mean(x, axis=(2,3))


        last_conv_layer = self.model.get_layer(self.last_conv_layer_name)

        get_output = K.function([self.model.layers[0].input, K.learning_phase()], [last_conv_layer.output])
        [conv_outputs_batch] = get_output([batch_x, 1])

        class_weights = global_average_pooling(conv_outputs_batch)

        pil_img_list = []
        for i in range(batch_x.shape[0]):
            conv_outputs = conv_outputs_batch[i, :, :, :]

            # Create the class activation map.
            cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[1:3])
            for j, w in enumerate(class_weights[:, 1]):
                cam += w * conv_outputs[j, :, :]

            # scale cam from 0 to 1 and resize to same dims as batch images
            cam = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(cam)
            cam = imresize(cam, (width, height))

            # apply color map to cam, clip values under threshold, apply transparency to layer
            cm_cam = plt.cm.jet(cam) # TODO: Allow COLORMAP to be configured via string paramter
            cm_cam[np.where(cam < self.cam_threshold_pct)] = 0
            cam = np.uint8(cm_cam * 255)
            cam[:,:,3] = np.uint((1-self.cam_transparency_pct)*255)
            # get as PIL RGBA image to composite
            cam_pil_img_rgba = Image.fromarray(cam, mode='RGBA')
            cam_pil_img_rgba = cam_pil_img_rgba.transpose(Image.ROTATE_90)

            # (3, width, height) -> (width, height, 3)
            x = np.uint8(np.rollaxis(batch_x[i], 0, 3) * 255)
            # get as PIL RGBA image to composite
            x_pil_img_rgba = Image.fromarray(x, mode='RGB').convert(mode='RGBA')

            x_pil_img_rgba = x_pil_img_rgba.transpose(Image.ROTATE_90)


            # composite cam layer over image layer and save
            pil_img = Image.alpha_composite(x_pil_img_rgba, cam_pil_img_rgba)
            #pil_img = pil_img.transpose(Image.ROTATE_90)

            pil_img_list.append(pil_img)

        accuracy_img_size = (pil_img_list[0].size[0] + self.image_border_thickness * 2,
                             pil_img_list[0].size[1] + self.image_border_thickness * 2)
        accuracy_list = self.get_accuracy_list(batch_x, batch_y).astype(dtype=np.int32)

        stitched_pil_cam_images = stitch_pil_images(pil_img_list, margin = self.image_margin, border_thickness=self.image_border_thickness)
        stitched_pil_accuracy_images = stitch_pil_image_from_values(accuracy_list, accuracy_img_size, margin=self.image_margin, cmap_name='seismic_r')

        stitched_pil_output_image = Image.alpha_composite(stitched_pil_accuracy_images, stitched_pil_cam_images)

        full_file_name = os.path.join(self.directory, self.base_file_name + '_epoch_' + str(epoch) + '.png')

        stitched_pil_output_image.save(full_file_name)


    '''

        def on_epoch_end(self, epoch, logs={}):

            batch_x, batch_y = self.generator.next()

            # TODO: depends on 'tf' or 'th'
            width = batch_x.shape[2]
            height = batch_x.shape[3]

            def global_average_pooling(x):
                return K.mean(x, axis=(2, 3))

            def global_average_pooling_shape(input_shape):
                return input_shape[0:2]



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
                cam = imresize(cam, (width, height))

                # apply color map to cam, clip values under threshold, apply transparency to layer
                cm_cam = plt.cm.jet(cam) # TODO: Allow COLORMAP to be configured via string paramter
                cm_cam[np.where(cam < self.cam_threshold_pct)] = 0
                cam = np.uint8(cm_cam * 255)
                cam[:,:,3] = np.uint((1-self.cam_transparency_pct)*255)
                # get as PIL RGBA image to composite
                cam_pil_img_rgba = Image.fromarray(cam, mode='RGBA')
                cam_pil_img_rgba = cam_pil_img_rgba.transpose(Image.ROTATE_90)

                # (3, width, height) -> (width, height, 3)
                x = np.uint8(np.rollaxis(batch_x[i], 0, 3) * 255)
                # get as PIL RGBA image to composite
                x_pil_img_rgba = Image.fromarray(x, mode='RGB').convert(mode='RGBA')

                x_pil_img_rgba = x_pil_img_rgba.transpose(Image.ROTATE_90)


                # composite cam layer over image layer and save
                pil_img = Image.alpha_composite(x_pil_img_rgba, cam_pil_img_rgba)
                #pil_img = pil_img.transpose(Image.ROTATE_90)

                pil_img_list.append(pil_img)

            accuracy_img_size = (pil_img_list[0].size[0] + self.image_border_thickness * 2,
                                 pil_img_list[0].size[1] + self.image_border_thickness * 2)
            accuracy_list = self.get_accuracy_list(batch_x, batch_y).astype(dtype=np.int32)

            stitched_pil_cam_images = stitch_pil_images(pil_img_list, margin = self.image_margin, border_thickness=self.image_border_thickness)
            stitched_pil_accuracy_images = stitch_pil_image_from_values(accuracy_list, accuracy_img_size, margin=self.image_margin, cmap_name='seismic_r')

            stitched_pil_output_image = Image.alpha_composite(stitched_pil_accuracy_images, stitched_pil_cam_images)

            full_file_name = os.path.join(self.directory, self.base_file_name + '_epoch_' + str(epoch) + '.png')

            stitched_pil_output_image.save(full_file_name)
    '''

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

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