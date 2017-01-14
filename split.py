from base_experiment import BaseExperiment

import tensorflow as tf #TODO: Try to remove and use keras backend instead
from keras import backend as K

import numpy as np
import numpy.core.numeric as _nx #TODO: find better way to do this, can i just use np.array instead of _nx.array??

from keras.layers import Input, Lambda, Merge
from keras.models import Model

def multi_split(layers, indices_or_sections_list_with_axis_tuple_list = [(None, 0)]):

    if not isinstance(layers, list):
        layers = [layers]

    for (indices_or_sections, axis) in indices_or_sections_list_with_axis_tuple_list:

        layers = [layer for layer in split(layers, indices_or_sections, axis)]


    return layers


def split(layers, indices_or_sections = None, axis=0):


    if not isinstance(layers, list):
        layers = [layers]

    split_layers = []

    for layer in layers:

        try:
            layer = layer.output
        except:
            pass

        dims = K.ndim(layer)
        Ntotal = K.int_shape(layer)[axis+1]
        try:
            # handle scalar case.
            Nsections = len(indices_or_sections) + 1
            div_points = [0] + list(indices_or_sections) + [Ntotal]
        except TypeError:
            # indices_or_sections is a scalar, not an array.
            Nsections = int(indices_or_sections)
            if Nsections <= 0:
                raise ValueError('number sections must be larger than 0.')
            Neach_section, extras = divmod(Ntotal, Nsections)
            section_sizes = ([0] +
                             extras * [Neach_section+1] +
                             (Nsections-extras) * [Neach_section])
            div_points = _nx.array(section_sizes).cumsum()



        for i in range(Nsections):

            def split_func(array, st, end, sub_array_axis, sub_array_dims):
                split_dims = np.arange(sub_array_dims)
                split_dims[sub_array_axis + 1] = 1
                split_dims[1] = sub_array_axis + 1

                trans_array = tf.transpose(array, split_dims)

                if sub_array_dims == 2:
                    split_array = trans_array[:, st:end]
                elif sub_array_dims == 3:
                    split_array = trans_array[:, st:end, :]
                else:
                    split_array = trans_array[:] # TODO

                sub_array = tf.transpose(split_array, split_dims)

                return sub_array


            split_lambda = Lambda(split_func,
                                  arguments={'st' :  div_points[i], 'end' : div_points[i + 1], 'sub_array_axis' : axis, 'sub_array_dims' : dims })

            split_lambda(layer)

            split_layers.append(split_lambda)

    return split_layers




class SplitMergeExperiment(BaseExperiment):

    def __init__(self, **kwargs):
        super().__init__("SplitMergeExperiment", None, None, None)

    def setup(self):
        pass

    def get_model(self, input_shape):
        inputs = Input(shape=input_shape)
        split_layers = multi_split(inputs, [(2, 0), (2, 1)])
        tile12 = Merge(split_layers[0:2], mode='concat', concat_axis=2)
        tile34 = Merge(split_layers[2:4], mode='concat', concat_axis=2)
        tile1234 = Merge([tile12, tile34], mode='concat', concat_axis=1)

        model = Model(input=inputs, output=tile1234.output)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return model


    def train(self):

        tile1 = np.reshape(np.tile(np.arange(10), 10), (10, 10))
        tile2 = np.reshape(np.tile(np.arange(10) * 2, 10), (10, 10))
        tile3 = np.reshape(np.tile(np.arange(10) * 3, 10), (10, 10))
        tile4 = np.reshape(np.tile(np.arange(10) * 4, 10), (10, 10))

        v1 = np.vstack((tile1, tile3))
        v2 = np.vstack((tile2, tile4))

        x = np.hstack((v1, v2))
        x = np.expand_dims(x, axis=0)

        model = self.get_model(input_shape=(20, 20))

        model.fit(x=[x], y=[x], batch_size=1, nb_epoch=10)


class SplitMergeExperiment2(BaseExperiment):

    def __init__(self, **kwargs):
        super().__init__("SplitMergeExperiment2", None, None, None)

    def setup(self):
        pass

    def get_model(self, input_shape):
        inputs = Input(shape=input_shape)
        split_layers = multi_split(inputs, [(2, 0), (2, 1)])




        tile12 = Merge(split_layers[0:2], mode='concat', concat_axis=2)
        tile34 = Merge(split_layers[2:4], mode='concat', concat_axis=2)
        tile1234 = Merge([tile12, tile34], mode='concat', concat_axis=1)

        model = Model(input=inputs, output=tile1234.output)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return model


    def train(self):

        tile1 = np.reshape(np.ones((10,10) * 3), (10, 10))
        tile2 = np.reshape(np.ones((10,10) * 15), (10, 10))
        tile3 = np.reshape(np.ones((10,10) * 7), (10, 10))
        tile4 = np.reshape(np.ones((10,10) * 14), (10, 10))

        v1 = np.vstack((tile1, tile3))
        v2 = np.vstack((tile2, tile4))

        x = np.hstack((v1, v2))
        x = np.expand_dims(x, axis=0)

        model = self.get_model(input_shape=(20, 20))

        model.fit(x=[x], y=[x], batch_size=1, nb_epoch=10)


class SplitSumExperiment(BaseExperiment):
    def __init__(self, **kwargs):
        super().__init__("SplitSumExperiment", None, None, None)

    def setup(self):
        pass

    def get_model(self, input_shape):

        inputs = Input(shape=input_shape)

        outputs = []

        for split_layer in multi_split(inputs, [(2,0), (2,1)]):

            def sum_fn(l):
                s = K.sum(l, keepdims=True)
                return s

            outputs.append(Lambda(sum_fn, output_shape=(1,))(split_layer.output))

        model = Model(input=inputs, output=outputs)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):

        tile1 = np.reshape(np.tile(np.arange(10), 10), (10, 10))
        tile2 = np.reshape(np.tile(np.arange(10) * 2, 10), (10, 10))
        tile3 = np.reshape(np.tile(np.arange(10) * 3, 10), (10, 10))
        tile4 = np.reshape(np.tile(np.arange(10) * 4, 10), (10, 10))

        v1 = np.vstack((tile1, tile3))
        v2 = np.vstack((tile2, tile4))

        x = np.hstack((v1, v2))

        x = np.expand_dims(x, axis=0)
        y0 = np.matrix([[np.sum(tile1)]])
        y1 = np.matrix([[np.sum(tile2)]])
        y2 = np.matrix([[np.sum(tile3)]])
        y3 = np.matrix([[np.sum(tile4)]])

        model = self.get_model(input_shape=(20, 20))

        model.fit(x=[x], y=[y0, y1, y2, y3], batch_size=1, nb_epoch=10)


class SplitSumExperiment2(BaseExperiment):
    def __init__(self, **kwargs):
        super().__init__("SplitSumExperiment2", None, None, None)

    def setup(self):
        pass

    def get_model(self, input_shape):


        inputs = Input(shape=input_shape)

        outputs = []

        for split_layer in multi_split(inputs, [(4,0)]):

            def sum_fn(l):
                s = K.sum(l, keepdims=True)
                return s

            outputs.append(Lambda(sum_fn, output_shape=(1,))(split_layer.output))

        model = Model(input=inputs, output=outputs)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):
        x = np.expand_dims(np.matrix([
            [1, 10], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0],[13, 0], [14, 0], [15, 0], [16, 0]
        ]), axis=0)
        y0 = np.matrix([[1 + 2 + 3 + 4 + 10]])
        y1 = np.matrix([[5 + 6 + 7 + 8]])
        y2 = np.matrix([[9 + 10 + 11 + 12]])
        y3 = np.matrix([[13 + 14 + 15 + 16]])

        model = self.get_model(input_shape=(16, 2))

        model.fit(x=[x], y=[y0, y1, y2, y3], batch_size=1, nb_epoch=10)


if __name__ == "__main__":

    SplitMergeExperiment().train()
    SplitMergeExperiment2().train()
    SplitSumExperiment().train()
    SplitSumExperiment2().train()
