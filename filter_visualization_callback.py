from __future__ import print_function
import time
import math
import os
from scipy.misc import imsave
import numpy as np
from skimage.filter import gaussian
from keras import backend as K
from keras.callbacks import Callback

class FilterVisualizationCallback(Callback):

    def __init__(self, directory, conv_layer_prefix = "conv", base_filename = "filters"):
        self.base_filename = base_filename
        self.directory = directory
        self.conv_layer_prefix = conv_layer_prefix

        # make sure the directory exists before writing filter images to it
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def on_train_begin(self, logs={}):
        # this is the placeholder for the input images
        self.input_img = self.model.input

        if K.image_dim_ordering() == 'th':
            self.img_channels = self.model.layers[0].input_shape[1]
            self.img_width = self.model.layers[0].input_shape[2]
            self.img_height = self.model.layers[0].input_shape[3]
        else:
            self.img_channels = self.model.layers[0].input_shape[3]
            self.img_width = self.model.layers[0].input_shape[1]
            self.img_height = self.model.layers[0].input_shape[2]


        # get the symbolic outputs of each "key" layer (we gave them unique names).
        self.conv_layer_dict = dict([(layer.name, layer)
                                     for layer
                                     in self.model.layers[1:]
                                     if self.conv_layer_prefix in layer.name])
        self.conv_size_dict = dict([(layer_name, self.__get_conv_size_details(layer))
                                     for layer_name, layer
                                     in self.conv_layer_dict.items()])

    # TODO: probably could remove and calculate in stitching: len(filter)**0.5
    def __get_conv_size_details(self, layer):

        filter_length = layer.output_shape[1]

        val = int(math.ceil(math.sqrt(filter_length)))

        return (filter_length, val)


    def on_epoch_end(self, epoch, logs={}):

        for layer_name, layer in self.conv_layer_dict.items():
            self.base_filename = 'filters_reg'

            filters = self.__get_visualized_layer_filters(layer_name, False)

            stitched_filters = self.__stitch_visualized_layer_filters(layer_name, filters)
            self.__save_stitched_visualized_layer(layer_name, epoch, stitched_filters)

            self.base_filename = 'filters_decay'

            filters = self.__get_visualized_layer_filters(layer_name, True)

            stitched_filters = self.__stitch_visualized_layer_filters(layer_name, filters)
            self.__save_stitched_visualized_layer(layer_name, epoch, stitched_filters)

    def __save_stitched_visualized_layer(self, layer_name, epoch, stitched_filters):
        imsave(os.path.join(self.directory, layer_name + '_' + str(epoch) + '_' + self.base_filename + '.png'), stitched_filters)


    def __stitch_visualized_layer_filters(self, layer_name, filters):

        filter_length, filter_grid_dim = self.conv_size_dict[layer_name]

        # build a black picture with enough space for
        # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
        margin = 5
        width = filter_grid_dim * self.img_width + (filter_grid_dim - 1) * margin
        height = filter_grid_dim * self.img_height + (filter_grid_dim - 1) * margin
        stitched_filters = np.zeros((width, height, 3))

        # fill the picture with our saved filters
        for i in range(filter_grid_dim):
            for j in range(filter_grid_dim):

                index = i * filter_grid_dim + j

                if index >= filter_length:
                    continue

                img, loss = filters[index]

                if img.shape[2] == 1:
                    img = np.asarray(np.dstack((img, img, img)), dtype=np.uint8)


                stitched_filters[(self.img_width + margin) * i: (self.img_width + margin) * i + self.img_width,
                (self.img_height + margin) * j: (self.img_height + margin) * j + self.img_height, :] = img


        return stitched_filters


    def __get_visualized_layer_filters(self, layer_name, extra):

        filter_length, _ = self.conv_size_dict[layer_name]

        kept_filters = []
        for filter_index in range(0, filter_length):

            #print('Processing filter %d' % filter_index)
            start_time = time.time()

            # we build a loss function that maximizes the activation
            # of the nth filter of the layer considered
            layer_output = self.conv_layer_dict[layer_name].output
            if K.image_dim_ordering() == 'th':
                loss = K.mean(layer_output[:, filter_index, :, :])
            else:
                loss = K.mean(layer_output[:, :, :, filter_index])

            # we compute the gradient of the input picture wrt this loss
            grads = K.gradients(loss, self.input_img)[0]

            # normalization trick: we normalize the gradient
            grads = self.__normalize(grads)

            # this function returns the loss and grads given the input picture
            iterate = K.function([self.input_img], [loss, grads])

            # step size for gradient ascent
            step = 1.

            # we start from a gray image with some random noise
            if K.image_dim_ordering() == 'th':
                input_img_data = np.random.random((1, self.img_channels, self.img_width, self.img_height))
            else:
                input_img_data = np.random.random((1, self.img_width, self.img_height, self.img_channels))
            input_img_data = (input_img_data - 0.5) * 20 + 128

            # we run gradient ascent for 20 steps
            for i in range(20):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step

                if extra == True:
                    input_img_data = self.__decay_regularization(input_img_data)
                    input_img_data = self.__clip_weak_pixel_regularization(input_img_data)

                #print('Current loss value:', loss_value)
                #if loss_value <= 0.:
                #    # some filters get stuck to 0, we can skip them
                #    break

            # decode the resulting input image
            #if loss_value > 0:
            img = self.__deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
            end_time = time.time()
            #print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

        # the filters that have the highest loss are assumed to be better-looking.
        # we will only keep the top 64 filters.
        #kept_filters.sort(key=lambda x: x[1], reverse=True)
        #kept_filters = kept_filters[:n * n]
        return kept_filters

    def __normalize(self, x):
        '''
        Utility function to normalize a tensor by its L2 norm
        '''

        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


    def __deprocess_image(self, x):
        '''
        Utility function to convert a tensor into a valid image
        '''

        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        if K.image_dim_ordering() == 'th':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    # Define regularizations:
    # TODO: convert size=(3, 3) to sigma
    def __blur_regularization(img, grads, sigma=1):
        return gaussian(img, sigma)

    def __decay_regularization(self, img, decay=0.8):
        return decay * img

    def __clip_weak_pixel_regularization(self, img, percentile=1):
        clipped = img
        threshold = np.percentile(np.abs(img), percentile)
        clipped[np.where(np.abs(img) < threshold)] = 0
        return clipped
