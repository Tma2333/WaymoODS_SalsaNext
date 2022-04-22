import os

import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
from pathlib import Path
import matplotlib.pyplot as plt

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

def get_3d_seg_data(path):
    frame = get_frame_with_lidar_label (path)
    (range_images, _, segmentation_labels, _) = frame_utils.parse_range_image_and_camera_projection(frame)
    data = {}
    
    ri1_label = segmentation_labels[open_dataset.LaserName.TOP][0]
    ri2_label = segmentation_labels[open_dataset.LaserName.TOP][1]
    ri1_range_image = range_images[open_dataset.LaserName.TOP][0]
    ri2_range_image = range_images[open_dataset.LaserName.TOP][0]
    
    ri1_label = convert_to_numpy(ri1_label)
    ri1_range_image = convert_to_numpy(ri1_range_image)
    ri2_label = convert_to_numpy(ri2_label)
    ri2_range_image = convert_to_numpy(ri2_range_image)

    data['ri1_label'] = ri1_label
    data['ri1_range_image'] = ri1_range_image
    data['ri2_label'] = ri2_label
    data['ri2_range_image'] = ri2_range_image
    data['legend'] = {'label': ['instance id', 'semantic class'], 'range_image': ['range', 'intensity', 'elongation', '?']}
    return data


def get_2d_seg_data(path):
    frame = get_frame_with_lidar_label (path)
    (range_images, camera_projections, segmentation_labels, _) = frame_utils.parse_range_image_and_camera_projection(frame)
    data = {}
    
    ri1_label = segmentation_labels[open_dataset.LaserName.TOP][0]
    ri2_label = segmentation_labels[open_dataset.LaserName.TOP][1]
    ri1_range_image = range_images[open_dataset.LaserName.TOP][0]
    ri2_range_image = range_images[open_dataset.LaserName.TOP][0]

    
    ri1_label = convert_to_numpy(ri1_label)
    ri1_range_image = convert_to_numpy(ri1_range_image)
    ri2_label = convert_to_numpy(ri2_label)
    ri2_range_image = convert_to_numpy(ri2_range_image)

    data['ri1_label'] = ri1_label
    data['ri1_range_image'] = ri1_range_image
    data['ri2_label'] = ri2_label
    data['ri2_range_image'] = ri2_range_image
    data['legend'] = {'label': ['instance id', 'semantic class'], 'range_image': ['range', 'intensity', 'elongation', '?']}
    return data


def convert_to_numpy (pd_data):
    tensor = tf.convert_to_tensor(pd_data.data)
    tensor = tf.reshape(tensor, pd_data.shape.dims)
    return tensor.numpy()

def get_frame_with_lidar_label (path):
    path = Path(path)
    for tfrecord in path.glob('*.tfrecord'):
        dataset = tf.data.TFRecordDataset(str(tfrecord), compression_type='')
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if frame.lasers[0].ri_return1.segmentation_label_compressed:
                break
        break
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

