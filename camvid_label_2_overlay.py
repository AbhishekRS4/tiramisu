# @author : Abhishek R S
# create image-mask overlays for visualization

import os
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf
from scipy.misc import imread

tf.enable_eager_execution()

color_maps = np.array([[0, 0, 0], # other
              [192, 128, 0], # bicyclist
              [128, 0, 64], # car
              [64, 128, 192], # person
              [128, 64, 128], # road
              [192, 0, 192], # two wheeler
              [192,  0,  0], # sidewalk
              [128, 128, 192], # sign
              [192, 128, 64], # truck
              [128, 64, 192], # train
              [64,  64, 0], # traffic light
              [64,  64, 128]], # other moving 
              dtype = np.uint8)

def overlay_generator(images_dir, labels_dir, overlays_dir, alpha = 0.75):
    images_list = os.listdir(images_dir)
    print('Number of overlays to generate : ' + str(len(images_list)))

    if not os.path.exists(overlays_dir):
        os.makedirs(overlays_dir)

    for image_file in images_list:
        image = cv2.imread(os.path.join(images_dir, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = imread(os.path.join(labels_dir, 'label_' + image_file))

        label_one_hot = tf.one_hot(label, depth = len(color_maps), dtype = tf.uint8).numpy()
        mask = np.dot(label_one_hot, color_maps) 

        image_mask_overlay = cv2.addWeighted(image, 1, mask, alpha, 0, image)
        image_mask_overlay = cv2.cvtColor(image_mask_overlay, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(overlays_dir, 'overlay_' + image_file), image_mask_overlay)

def main():
    images_dir = '/opt/data/abhishek/cityscapes/resized_images/test/'
    labels_dir = './model_fcn16_100/labels_100/'
    overlays_dir = './model_fcn16_100/overlays_100/'

    parser = argparse.ArgumentParser()
    parser.add_argument('-images_dir', default = images_dir, type = str, help = 'path to load image files')
    parser.add_argument('-labels_dir', default = labels_dir, type = str, help = 'path to load label files')
    parser.add_argument('-overlays_dir', default = overlays_dir, type = str, help = 'path to save image-mask overlay files')

    input_args = vars(parser.parse_args(sys.argv[1:]))

    print('Genrating overlays for images and masks')
    for k in input_args.keys():
        print(k + ': ' + str(input_args[k]))
    print('')
    print('')

    print('Overlay generation started....')
    overlay_generator(images_dir = input_args['images_dir'], labels_dir = input_args['labels_dir'], overlays_dir = input_args['overlays_dir'])
    print('')
    print('Overlay generation completed')

if __name__ == '__main__':
    main()
