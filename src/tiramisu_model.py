# @author : Abhishek R S

import os
import sys
import h5py
import numpy as np
import tensorflow as tf

"""
Tiramisu

# Reference
- [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
- [The One Hundred Layers Tiramisu - Fully Convolutional DenseNets for Semantic Segmentation]
  (https://arxiv.org/pdf/1611.09326.pdf)
"""

class Tiramisu:
    def __init__(self, is_training, data_format="channels_first", num_classes=15, which_model=103):
        self._is_training = is_training
        self._data_format = data_format
        self._num_classes = num_classes
        self._which_model = which_model
        self._padding = "SAME"
        self._feature_map_axis = None
        self._num_dense_blocks = None
        self._growth_rate = None
        self._num_kernels = list()
        self._initializer = tf.contrib.layers.xavier_initializer_conv2d()

        """
        always use channels_first i.e. NCHW as the data format on a GPU
        """

        if self._which_model == 56:
            self._growth_rate = 12
            self._num_dense_blocks = [4, 4, 4, 4, 4, 15]
        elif self._which_model == 67:
            self._growth_rate = 16
            self._num_dense_blocks = [5, 5, 5, 5, 5, 15]
        elif self._which_model == 103:
            self._growth_rate = 16
            self._num_dense_blocks = [4, 5, 7, 10, 12, 15]
        else:
            print("Not a valid option")
            sys.exit(0)

        if data_format == "channels_first":
            self._feature_map_axis = 1
        else:
            self._feature_map_axis = -1

    # build tiramisu network
    def tiramisu_net(self, features, debug=True):
        if self._data_format == "channels_last":
            features = tf.transpose(features, perm=[0, 2, 3, 1])

        # Encoder
        # Stage 0
        self.stage0 = self._initial_block(features, 48, [3, 3], [1, 1], name="initial")

        # Stage 1
        self.dense1 = self._dense_block(self.stage0, 48, self._num_dense_blocks[0], name="dense1")
        self.down1 = self._transition_down_block(self.dense1, self._num_kernels[0], name="transition_down1")

        # Stage 2
        self.dense2 = self._dense_block(self.down1, self._num_kernels[0], self._num_dense_blocks[1], name="dense2")
        self.down2 = self._transition_down_block(self.dense2, self._num_kernels[1], name="transition_down2")

        # Stage 3
        self.dense3 = self._dense_block(self.down2, self._num_kernels[1], self._num_dense_blocks[2], name="dense3")
        self.down3 = self._transition_down_block(self.dense3, self._num_kernels[2], name="transition_down3")

        # Stage 4
        self.dense4 = self._dense_block(self.down3, self._num_kernels[2], self._num_dense_blocks[3], name="dense4")
        self.down4 = self._transition_down_block(self.dense4, self._num_kernels[3], name="transition_down4")

        # Stage 5
        self.dense5 = self._dense_block(self.down4, self._num_kernels[3], self._num_dense_blocks[4], name="dense5")
        self.down5 = self._transition_down_block(self.dense5, self._num_kernels[4], name="transition_down5")

        # Stage 6
        self.dense6 = self._dense_block(self.down5, self._num_kernels[4], self._num_dense_blocks[5], name="dense6")

        # Decoder
        # Stage 1
        self.up1 = self._transition_up_block(self.dense6, self._growth_rate * self._num_dense_blocks[5], name="transition_up1")
        self.concat1 = tf.concat([self.up1, self.dense5], axis=self._feature_map_axis, name="concat1")
        self.up_dense1 = self._dense_block(self.concat1, self._num_kernels[5], self._num_dense_blocks[4], name="up_dense1")

        # Stage 2
        self.up2 = self._transition_up_block(self.up_dense1, self._growth_rate * self._num_dense_blocks[4], name="transition_up2")
        self.concat2 = tf.concat([self.up2, self.dense4], axis=self._feature_map_axis, name="concat2")
        self.up_dense2 = self._dense_block(self.concat2, self._num_kernels[4], self._num_dense_blocks[3], name="up_dense2")

        # Stage 3
        self.up3 = self._transition_up_block(self.up_dense2, self._growth_rate * self._num_dense_blocks[3], name="transition_up3")
        self.concat3 = tf.concat([self.up3, self.dense3], axis=self._feature_map_axis, name="concat3")
        self.up_dense3 = self._dense_block(self.concat3, self._num_kernels[3], self._num_dense_blocks[2], name="up_dense3")

        # Stage 4
        self.up4 = self._transition_up_block(self.up_dense3, self._growth_rate * self._num_dense_blocks[2], name="transition_up4")
        self.concat4 = tf.concat([self.up4, self.dense2], axis=self._feature_map_axis, name="concat4")
        self.up_dense4 = self._dense_block(self.concat4, self._num_kernels[2], self._num_dense_blocks[1], name="up_dense4")

        # Stage 5
        self.up5 = self._transition_up_block(self.up_dense4, self._growth_rate * self._num_dense_blocks[1], name="transition_up5")
        self.concat5 = tf.concat([self.up5, self.dense1], axis=self._feature_map_axis, name="concat5")
        self.up_dense5 = self._dense_block(self.concat5, self._num_kernels[1], self._num_dense_blocks[0], name="up_dense5")

        # logits
        self.logits = self._get_conv2d_layer(self.up_dense5, self._num_classes, [1, 1], [1, 1], name="logits")

    # dense block
    def _dense_block(self, input_layer, num_kernels, num_blocks, name):
        x = input_layer

        for i in range(1, num_blocks + 1):
            x = self._conv_block_dense(x, self._growth_rate, name=name + "_block" + str(i))
            num_kernels = num_kernels + self._growth_rate

        self._num_kernels.append(num_kernels)

        return x

    # convolution dense block
    def _conv_block_dense(self, input_layer, num_kernels, name):
        x = self._conv_block(input_layer, num_kernels, [3, 3], [1, 1], name=name + "x1")
        z = tf.concat([input_layer, x], axis=self._feature_map_axis, name=name + "_concat")

        return z

    # transition down block
    def _transition_down_block(self, input_layer, num_kernels, name):
        x = self._conv_block(input_layer, num_kernels, [1, 1], [1, 1], name=name + "x1")
        x = self._get_maxpool2d_layer(x, [2, 2], [2, 2], name=name + "_maxpool")

        return x

    # transition up block
    def _transition_up_block(self, input_layer, num_kernels, name):
        x = self._get_conv2d_transpose_layer(input_layer, num_kernels, [3, 3], [2, 2], name=name + "_tr_conv")

        return x

    # convolution block
    def _conv_block(self, input_layer, num_kernels, kernel_size, strides, name, use_bias=False):
        x = self._get_relu_activation(input_layer, name=name + "_relu")
        x = self._get_conv2d_layer(x, num_kernels, kernel_size, strides, use_bias=use_bias, name=name + "_conv")
        x = self._get_dropout_layer(x, rate=0.2, name=name + "_dropout")

        return x

    # initial block
    def _initial_block(self, input_layer, num_kernels, kernel_size, strides, name):
        x = self._get_conv2d_layer(input_layer, num_kernels, kernel_size, strides, name=name + "_conv")
        x = self._get_relu_activation(x, name=name + "_relu")

        return x

    # return convolution2d layer
    def _get_conv2d_layer(self, input_layer, num_filters, kernel_size, strides, use_bias=True, name="conv"):
        conv_2d_layer = tf.layers.conv2d(
            inputs=input_layer, filters=num_filters, kernel_size=kernel_size, strides=strides,
            use_bias=use_bias, padding=self._padding, data_format=self._data_format, kernel_initializer=self._initializer, name=name
        )
        return conv_2d_layer

    # return transposed convolution2d upsampling layer
    def _get_conv2d_transpose_layer(self, input_layer, num_filters, kernel_size, strides, name="tr_conv"):
        conv_2d_tr_layer = tf.layers.conv2d_transpose(
            inputs=input_layer, filters=num_filters, kernel_size=kernel_size, strides=strides, padding=self._padding,
            data_format=self._data_format, kernel_initializer=self._initializer, name=name)
        return conv_2d_tr_layer

    # return maxpool2d layer
    def _get_maxpool2d_layer(self, input_layer, pool_size, pool_strides, name="maxpool"):
        pool_layer = tf.layers.max_pooling2d(
            input_layer, pool_size, pool_strides, padding=self._padding, data_format=self._data_format, name=name
        )
        return pool_layer

    # return relu activation function
    def _get_relu_activation(self, input_layer, name="relu"):
        relu_layer = tf.nn.relu(input_layer, name=name)
        return relu_layer

    # return dropout layer
    def _get_dropout_layer(self, input_layer, rate=0.5, name="dropout"):
        dropout_layer = tf.layers.dropout(inputs=input_layer, rate=rate, training=self._is_training, name=name)
        return dropout_layer

    # return batch normalization layer
    def _get_batchnorm_layer(self, input_layer, name="bn"):
        bn_layer = tf.layers.batch_normalization(input_layer, axis=self._feature_map_axis, training=self._is_training, name=name)
        return bn_layer
