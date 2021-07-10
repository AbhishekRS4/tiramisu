# @author : Abhishek R S

import os
import sys
import cv2
import argparse
import numpy as np
import tensorflow as tf
from scipy.misc import imread

tf.enable_eager_execution()

color_maps = np.array([
    [0, 0, 0],  # other
    [0, 128, 192],  # bicyclist
    [64, 0, 128],  # car
    [192, 128, 64],  # person
    [128, 64, 128],  # road
    [192, 0, 192],  # two wheeler
    [0, 0, 192],  # sidewalk
    [192, 128, 128],  # sign
    [64, 128, 192],  # truck
    [192, 64, 128],  # train
    [0, 64, 64],  # traffic light
    [128, 64, 64]],  # other moving
    dtype=np.uint8)

def overlay_generator(FLAGS):
    images_list = os.listdir(FLAGS.images_dir)
    print(f"Number of overlays to generate : {len(images_list)}")

    if not os.path.exists(FLAGS.overlays_dir):
        os.makedirs(FLAGS.overlays_dir)

    for image_file in images_list:
        image = cv2.imread(os.path.join(FLAGS.images_dir, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = imread(os.path.join(FLAGS.labels_dir, "label_" + image_file))

        label_one_hot = tf.one_hot(label, depth=len(color_maps), dtype=tf.uint8).numpy()
        mask = np.dot(label_one_hot, color_maps)

        image_mask_overlay = cv2.addWeighted(image, 1, mask, FLAGS.alpha, 0, image)
        image_mask_overlay = cv2.cvtColor(image_mask_overlay, cv2.COLOR_BGR2RGB)

        cv2.imwrite(
            os.path.join(FLAGS.overlays_dir, "overlay_" + image_file),
            image_mask_overlay
        )

def main():
    images_dir = "/opt/data/abhishek/cityscapes/resized_images/test/"
    labels_dir = "./model_fcn16_100/labels_100/"
    overlays_dir = "./model_fcn16_100/overlays_100/"
    alpha = 0.75

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--images_dir", default=images_dir,
        type=str, help="path to load image files")
    parser.add_argument("--labels_dir", default=labels_dir,
        type=str, help="path to load label files")
    parser.add_argument("--overlays_dir", default=overlays_dir,
        type=str, help="path to save image-mask overlay files")
    parser.add_argument("--alpha", default=alpha,
        type=str, help="alpha to control transparency of mask on overlay")

    FLAGS, unparsed = parser.parse_known_args()
    print("Genrating overlays for images and masks")
    print("Overlay generation started....")
    overlay_generator(FLAGS)
    print("Overlay generation completed")

if __name__ == "__main__":
    main()
