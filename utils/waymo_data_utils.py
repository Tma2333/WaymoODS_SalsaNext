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
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


def get_3d_seg_data(path=None, frame=None, test=False):
    if frame is None:
        if path is None:
            raise ValueError('When frame is None, the path to directory containt *.tfrecord must be specify')
        frame = get_frame_with_lidar_label(path)
    (range_images, _, segmentation_labels, _) = frame_utils.parse_range_image_and_camera_projection(frame)
    data = {}
    
    if not test:
        ri1_label = segmentation_labels[open_dataset.LaserName.TOP][0]
        ri2_label = segmentation_labels[open_dataset.LaserName.TOP][1]
    ri1_range_image = range_images[open_dataset.LaserName.TOP][0]
    ri2_range_image = range_images[open_dataset.LaserName.TOP][1]
    
    if not test:
        ri1_label = convert_to_numpy(ri1_label)
        ri2_label = convert_to_numpy(ri2_label)
    ri1_range_image = convert_to_numpy(ri1_range_image)
    ri2_range_image = convert_to_numpy(ri2_range_image)

    if not test:
        data['ri1_label'] = ri1_label
        data['ri2_label'] = ri2_label

    data['ri1_range_image'] = ri1_range_image
    data['ri2_range_image'] = ri2_range_image
    data['legend'] = {'label': ['instance id', 'semantic class'], 'range_image': ['range', 'intensity', 'elongation', 'is_in_nlz']}
    return data


def get_2d_seg_data(path=None, frame=None, test=False):
    if frame is None:
        if path is None:
            raise ValueError('When frame is None, the path to directory containt *.tfrecord must be specify')
        frame = get_frame_with_lidar_label(path)
    (range_images, camera_projections, segmentation_labels, _) = frame_utils.parse_range_image_and_camera_projection(frame)
    data = {}
    
    if not test:
        ri1_label = segmentation_labels[open_dataset.LaserName.TOP][0]
        ri2_label = segmentation_labels[open_dataset.LaserName.TOP][1]
        
    ri1_range_image = range_images[open_dataset.LaserName.TOP][0]
    ri2_range_image = range_images[open_dataset.LaserName.TOP][1]
    ri1_proj = camera_projections[open_dataset.LaserName.TOP][0]
    ri2_proj = camera_projections[open_dataset.LaserName.TOP][1]

    if not test:
        ri1_label = convert_to_numpy(ri1_label)
        ri2_label = convert_to_numpy(ri2_label)

    ri1_range_image = convert_to_numpy(ri1_range_image)
    ri1_proj = convert_to_numpy(ri1_proj)

    ri2_range_image = convert_to_numpy(ri2_range_image)
    ri2_proj = convert_to_numpy(ri2_proj)

    data['image'] = {}
    for index, image in enumerate(frame.images):
        im_data = tf.image.decode_jpeg(image.image).numpy().transpose(2, 0, 1)
        data['image'][image.name] = im_data

    if not test:
        data['ri1_label'] = ri1_label
        data['ri2_label'] = ri2_label

    data['ri1_range_image'] = ri1_range_image
    data['ri1_proj'] = ri1_proj
    data['ri2_range_image'] = ri2_range_image
    data['ri2_proj'] = ri2_proj
    
    data['legend'] = {'label': ['instance id', 'semantic class'], 
                      'range_image': ['range', 'intensity', 'elongation', 'is_in_nlz'],
                      'image': ['cam id', '(C,H,W)'],
                      'proj':['first cam id', 'x', 'y', 'second cam id', 'x', 'y']}
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


def plot_range_image_helper(data, vmin=0, vmax=1, cmap='gray', name=None, fig=None, layout=None):
    if fig is None:
        fig = plt.figure(figsize=(200,5))
    if layout is None:
        layout = (1,1,1)
    ax = fig.add_subplot(*layout)
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    if name is not None:
        ax.set_title(name)
        ax.title.set_fontsize(40)
    ax.axis('off')


def show_range_image_data (data, cmap='gray'):
    ri1_range_image = data['ri1_range_image']
    ri2_range_iamge = data['ri2_range_image']
    data_name = data['legend']['range_image']

    fig = plt.figure(figsize=(200, 6*6))
    for i, range_image in enumerate((ri1_range_image, ri2_range_iamge)):
        title_base = f'ri{i}_'
        for j in range(3):
            image_data = range_image[..., j]
            image_data[image_data < 0] = 1e10
            name = data_name[j]
            if name == 'range':
                vmax = 75
            else:
                vmax = 1.5
            plot_range_image_helper(image_data, vmax=vmax, cmap=cmap, name=title_base+name, fig=fig, layout=(6,1,i*3+j+1))
    plt.tight_layout()


def show_range_image_label (data):
    ri1_label = data['ri1_label']
    ri2_label = data['ri2_label']

    fig = plt.figure(figsize=(200, 6*4))
    for i, label in enumerate((ri1_label, ri2_label)):
        title_base = f'ri{i}_'
        instance_id = label[..., 0]
        semantic_label = label[..., 1]

        plot_range_image_helper(instance_id, vmin=-1, vmax=200, cmap='Paired', 
                name=title_base+'instance id', fig=fig, layout=(6,1,i*2+1))
        plot_range_image_helper(semantic_label, vmin=0, vmax=22, cmap='tab20', 
                name=title_base+'semantic label', fig=fig, layout=(6,1,i*2+2))
    plt.tight_layout()


def show_proj_region (data):
    ri1_proj = data['ri1_proj']
    ri2_proj = data['ri2_proj']

    fig = plt.figure(figsize=(200, 6*4))
    for i, proj in enumerate((ri1_proj, ri2_proj)):
        title_base = f'ri{i}_'
        pj_cam1 = proj[..., 0]
        pj_cam2 = proj[..., 3]

        plot_range_image_helper(pj_cam1, vmin=0, vmax=5, cmap='Paired', 
                name=title_base+'first cam match', fig=fig, layout=(4,1,i*2+1))
        plot_range_image_helper(pj_cam2, vmin=0, vmax=5, cmap='Paired', 
                name=title_base+'second cam match', fig=fig, layout=(4,1,i*2+2))
    plt.tight_layout()
        

def plot_image_helper (data, name=None, fig=None, layout=None):
    if fig is None:
        fig = plt.figure(figsize=(200,5))
    if layout is None:
        layout = (1,1,1)
    ax = fig.add_subplot(*layout)
    ax.imshow(data)
    if name is not None:
        ax.set_title(name)
        ax.title.set_fontsize(16)
    ax.axis('off')


def show_images (data):
    images = data['image']

    fig = plt.figure(figsize=(30, 20))
    for cam_id, image in images.items():
        im = image.transpose(1, 2, 0)
        plot_image_helper(im, name=f'cam_{cam_id}', fig=fig, layout=(2, 3,cam_id))
    plt.tight_layout()


def make_projected_pixel_im (ri1_proj, ri2_proj, images):
    H, W, _ = ri1_proj.shape

    ri1_pixel = np.zeros((H, W, 3))
    ri2_pixel = np.zeros((H, W, 3))

    for cam_id in range(1, 6):
        col, row = ri1_proj[ri1_proj[...,  0]==cam_id][..., [1,2]].T
        ri1_pixel[ri1_proj[...,  0]==cam_id] = images[cam_id][:, row, col].T
        
        col, row = ri2_proj[ri2_proj[...,  0]==cam_id][..., [1,2]].T
        ri2_pixel[ri2_proj[...,  0]==cam_id] = images[cam_id][:, row, col].T
    
    return ri1_pixel, ri2_pixel


def show_projected_pixel (data):
    ri1_proj = data['ri1_proj']
    ri2_proj = data['ri2_proj']
    images = data['image']

    ri1_pixel, ri2_pixel = make_projected_pixel_im(ri1_proj, ri2_proj, images)

    fig = plt.figure(figsize=(200, 6*2))
    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(ri1_pixel.astype('int'))
    ax.set_title('ri1 projected pixel')
    ax.axis('off')
    ax.title.set_fontsize(40)
    ax = fig.add_subplot(2, 1, 2)
    ax.imshow(ri2_pixel.astype('int'))
    ax.set_title('ri2 projected pixel')
    ax.axis('off')
    ax.title.set_fontsize(40)
    plt.tight_layout()


def extract_3d_seg_frames (path, txt_path):
    path = Path(path)
    txt_path = Path(txt_path)

    with open(str(txt_path), 'w') as f:
        for tfrecord in path.glob('*.tfrecord'):
            print(f'process {tfrecord}')
            dataset = tf.data.TFRecordDataset(str(tfrecord), compression_type='')
            for data in dataset:
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                if frame.lasers[0].ri_return1.segmentation_label_compressed:
                    context = frame.context.name
                    timestamp = frame.timestamp_micros
                    write_string = f'{context},{timestamp}\n'
                    f.write(write_string)


def show_labels_on_image (data, alpha=0.5, dense_label=True, grid=8, stride=8, nan=-1):
    """
    Projects the labels from the Lidar range images onto the visual images
    to show where on the visual images the Lidar labels are.
    """
    ri1_proj = data['ri1_proj']
    ri2_proj = data['ri2_proj']
    images = data['image']
    H, W = images[1].shape[1], images[1].shape[2]
    ri1_label = data['ri1_label']
    ri2_label = data['ri2_label']
    image_label = {}
    densify_label = {}
    for cam_id in range(1, 6):
        # Project main range lidar labels
        image_label[cam_id] = np.zeros((H, W)) + nan
        col, row = ri1_proj[ri1_proj[...,  0]==cam_id][..., [1,2]].T
        image_label[cam_id][row, col]=ri1_label[ri1_proj[...,  0]==cam_id][:,1]
        # Project secondary range lidar labels
        col, row = ri2_proj[ri2_proj[...,  0]==cam_id][..., [1,2]].T
        image_label[cam_id][row, col] = ri2_label[ri2_proj[...,  0]==cam_id][:,1]

        if dense_label:
            densify_label[cam_id] = densify(image_label[cam_id], grid, stride, nan)
    
    fig = plt.figure(figsize=(20, 35))
    for i in range(1, 6):  # Goes through the 5 images\
        if dense_label:
            ax = fig.add_subplot(5, 2, i*2-1)
        else:
            ax = fig.add_subplot(5, 1, i)
        ax.imshow(images[i].transpose(1,2,0))
        ax.imshow(image_label[i], alpha=alpha, cmap='Paired')
        ax.set_title('Labels projected onto image nr ' + str(i))
        ax.axis('off')
        ax.title.set_fontsize(25)
        if dense_label:
            ax = fig.add_subplot(5, 2, i*2)
            ax.imshow(images[i].transpose(1,2,0))
            ax.imshow(densify_label[i], alpha=alpha, cmap='Paired')
            ax.set_title('Densified Labels projected onto image nr ' + str(i))
            ax.axis('off')
            ax.title.set_fontsize(25)
    plt.tight_layout()



def densify (x, grid=8, stride=8, nan=-1):
    H, W = x.shape
    stride = grid

    uh = H - grid + 1
    nh = (H - grid) // stride + 1
    uw = W - grid + 1
    nw = (W - grid) // stride + 1

    hidx = np.repeat(np.arange(grid), grid)[np.newaxis, :] + np.repeat(np.arange(uh, step=stride), nw)[:,np.newaxis]
    widx = np.tile(np.arange(grid), grid)[np.newaxis, :] + np.tile(np.arange(uw, step=stride), nh)[:,np.newaxis]

    out = x[hidx, widx]
    grid_max = np.max(out, 1)
    grid_max = np.repeat(grid_max[:, np.newaxis], grid**2 ,axis=1)

    im = np.zeros((H, W)) + nan
    im[hidx, widx] = grid_max
    
    return im