import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


def read_first_frame (path):
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        
        return frame

def plot_range_image_helper(data, name, layout, vmin = 0, vmax=1, cmap='gray'):
    plt.subplot(*layout)
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(name)
    plt.grid(False)
    plt.axis('off')


def get_range_image(frame, laser_name, return_index):
    range_images, camera_projections, seg_labels, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
    return range_images[laser_name][return_index]


def show_range_image(range_image, layout_index_start = 1):
    range_image_tensor = tf.convert_to_tensor(range_image.data)
    range_image_tensor = tf.reshape(range_image_tensor, range_image.shape.dims)
    lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
    range_image_tensor = tf.where(lidar_image_mask, range_image_tensor,
                                    tf.ones_like(range_image_tensor) * 1e10)
    range_image_range = range_image_tensor[...,0] 
    range_image_intensity = range_image_tensor[...,1]
    range_image_elongation = range_image_tensor[...,2]
    plot_range_image_helper(range_image_range.numpy(), 'range',
                    [8, 1, layout_index_start], vmax=75, cmap='gray')
    plot_range_image_helper(range_image_intensity.numpy(), 'intensity',
                    [8, 1, layout_index_start + 1], vmax=1.5, cmap='gray')
    plot_range_image_helper(range_image_elongation.numpy(), 'elongation',
                    [8, 1, layout_index_start + 2], vmax=1.5, cmap='gray')
    frame.lasers.sort(key=lambda laser: laser.name)
    show_range_image(get_range_image(open_dataset.LaserName.TOP, 0), 1)
    show_range_image(get_range_image(open_dataset.LaserName.TOP, 1), 4)

