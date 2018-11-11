# @author : Abhishek R S

import os
import json
import numpy as np
import tensorflow as tf

# read the json file and return the content
def read_config_file(json_file_name):
    # open and read the json file
    config = json.load(open(json_file_name))

    # return the content
    return config

# create the model directory if not present
def init(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# parse function for tensorflow dataset api
def parse_fn(img_name, lbl_name):
    # read
    img_string = tf.read_file(img_name) 
    lbl_string = tf.read_file(lbl_name)

    # decode
    img = tf.image.decode_png(img_string, channels = 3)
    lbl = tf.image.decode_png(lbl_string, channels = 0)
    lbl = tf.squeeze(lbl)

    # zero padding
    img = tf.pad(img, paddings = [[0, 24], [0, 0], [0, 0]])
    lbl = tf.pad(lbl, paddings = [[0, 24], [0, 0]])

    # datatype casting
    img = tf.image.convert_image_dtype(img, dtype = tf.float32)
    lbl = tf.cast(lbl, dtype = tf.int32)

    # CHW format
    img = tf.transpose(img, perm = [2, 0, 1])

    return img, lbl

# return tf dataset
def get_tf_dataset(images_list, labels_list, num_epochs, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((images_list, labels_list))
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(parse_fn, num_parallel_calls = 8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(batch_size)

    return dataset
