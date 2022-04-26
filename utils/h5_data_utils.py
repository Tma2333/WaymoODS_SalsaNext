from pathlib import Path
import io

import numpy as np
from PIL import Image
import h5py
from tqdm import tqdm
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
from utils import get_2d_seg_data, make_projected_pixel_im


def parse_split_frames (txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    split_frames = {}
    for line in lines:
        context, timestamp = line.strip().split(',')
        if context not in split_frames:
            split_frames[context] = []
        split_frames[context].append(int(timestamp))
    return split_frames


def create_empty_h5 (h5_path, H=64, W=2650, image_dset_upperlim=1000000):
    channel_dict = {'range_image':4, 'label':2, 'proj': 6}
    with h5py.File(str(h5_path), 'w') as f:
        for data_name, channel in channel_dict.items():
            init_shape = (1,channel,H,W)
            maxshape = (None,channel,H,W) 
            chunks = (1,channel,H,W)
            for ri in ['ri1', 'ri2']:
                dataset_name = f'/{data_name}/{ri}'
                if data_name == 'range_image':
                    dtype = 'float32'
                else:
                    dtype = 'int32'
                f.create_dataset(dataset_name, init_shape, maxshape=maxshape, chunks=chunks, dtype=dtype)
        f.create_dataset('/proj_pixel', (1, 3, H, W), maxshape=(None,3,H,W), chunks=(1,3,H,W), dtype='uint8')
        f.create_dataset('/image', (1, 5, image_dset_upperlim), maxshape=(None,5, image_dset_upperlim), chunks=(1, 5, image_dset_upperlim), dtype='uint8')
        f.create_dataset('/image_length', (1, 5), maxshape=(None, 5), chunks=(1, 5), dtype='int32')



def write_h5_dataset(idx, h5_file_obj, h5_path, data):
    dset = h5_file_obj[h5_path]
    dset.resize(idx+1, axis=0)
    dset[idx, ...] = data


def tfrecord_to_h5 (txt_path, h5_path, data_dir, test=False):
    frame_dict = parse_split_frames(txt_path)
    print('create a new h5 dataset')
    create_empty_h5(str(h5_path))
    data_dir = Path(data_dir)

    idx = 0
    with  h5py.File(str(h5_path), 'r+') as f:
        image_dset = f['image']
        image_dset_upperlim = image_dset.shape[-1]
        image_length_dset = f['image_length']

        pbar = tqdm(frame_dict.items(), total=len(frame_dict))
        for context, timestamps in pbar:
            file_name = f'segment-{context}_with_camera_labels.tfrecord'
            pbar.set_description(f'process {file_name}')
            file_path = data_dir/file_name

            tfr_dataset = tf.data.TFRecordDataset(str(file_path), compression_type='')
            for data in tfr_dataset:
                # read frames
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                
                if frame.timestamp_micros not in timestamps:
                    continue

                if not frame.lasers[0].ri_return1.segmentation_label_compressed and not test:
                    raise ValueError('Frame does not contain label')

                # read data
                data = get_2d_seg_data(None, frame, test=test)
                
                if not test:
                    ri1_label = data['ri1_label']
                    ri2_label = data['ri2_label']
                    write_h5_dataset(idx, f, '/label/ri1', ri1_label.transpose(2, 0, 1))
                    write_h5_dataset(idx, f, '/label/ri2', ri2_label.transpose(2, 0, 1))

                ri1_range_image = data['ri1_range_image']
                ri1_proj = data['ri1_proj']

                
                write_h5_dataset(idx, f, '/range_image/ri1', ri1_range_image.transpose(2, 0, 1))
                write_h5_dataset(idx, f, '/proj/ri1', ri1_proj.transpose(2, 0, 1))

                
                ri2_range_image = data['ri2_range_image']
                ri2_proj = data['ri2_proj']

                
                write_h5_dataset(idx, f, '/range_image/ri2', ri2_range_image.transpose(2, 0, 1))
                write_h5_dataset(idx, f, '/proj/ri2', ri2_proj.transpose(2, 0, 1))

                ri1_pixel, _ = make_projected_pixel_im(ri1_proj, ri2_proj, data['image'])
                write_h5_dataset(idx, f, 'proj_pixel', ri1_pixel.astype(np.uint8).transpose(2, 0, 1))

                image_dset.resize(idx+1, axis=0)
                image_length_dset.resize(idx+1, axis=0)
                
                for _, image in enumerate(frame.images):
                    cam_id = image.name
                    jpeg_data =  np.frombuffer(image.image, dtype='uint8')
                    jpeg_byte_length = len(jpeg_data)
                    if jpeg_byte_length > image_dset_upperlim:
                        raise ValueError(f'{jpeg_byte_length} greater than upper byte limit {image_dset_upperlim}')
                    image_dset[idx, cam_id-1, :jpeg_byte_length] = jpeg_data
                    image_length_dset[idx, cam_id-1] = jpeg_byte_length

                idx += 1


def get_data_h5 (h5_path, idx):
    data = {}
    with h5py.File(str(h5_path), 'r') as f:
        data['ri1_range_image'] = f['/range_image/ri1'][idx, ...].transpose(1, 2, 0)
        data['ri2_range_image'] = f['/range_image/ri2'][idx, ...].transpose(1, 2, 0)
        data['ri1_label'] = f['/label/ri1'][idx, ...].transpose(1, 2, 0)
        data['ri2_label'] = f['/label/ri2'][idx, ...].transpose(1, 2, 0)
        data['ri1_proj'] = f['/proj/ri1'][idx, ...].transpose(1, 2, 0)
        data['ri2_proj'] = f['/proj/ri2'][idx, ...].transpose(1, 2, 0)
        data['proj_pixel'] = f['proj_pixel'][idx, ...].transpose(1, 2, 0)
        data['image'] = {}
        for i in range(5):
            cam_id = i+1
            byte_length = f['image_length'][idx, i]
            byte_data = f['image'][idx, i, :byte_length]
            image_data = Image.open(io.BytesIO(byte_data))
            image_data = np.array(image_data).transpose(2, 0, 1)
            data['image'][cam_id] = image_data

        data['legend'] = {'label': ['instance id', 'semantic class'], 
                          'range_image': ['range', 'intensity', 'elongation', 'is_in_nlz'],
                          'image': ['cam id', '(C,H,W)'],
                          'proj':['first cam id', 'x', 'y', 'second cam id', 'x', 'y'],
                          'proj_pixel':['R', 'G', 'B']}
    return data


def h5_subset (subset_path, h5_path, n=10, test=False):
    create_empty_h5(subset_path)

    with h5py.File(subset_path, 'r+') as subf, h5py.File(h5_path, 'r') as f:
        length = len(f['proj_pixel'])
        idx = np.random.choice(length, n, False)
        idx = np.sort(idx)
        for key, dataset in f.items():
            if key not in['image', 'image_length', 'proj_pixel']:
                if test and key=='label':
                    continue
                for sub_key, sub_dataset in dataset.items():
                    dataset_path = f'/{key}/{sub_key}'
                    subf[dataset_path].resize(n, axis=0)
                    subf[dataset_path][...] = sub_dataset[idx, ...]
            else:
                dataset_path = f'/{key}'
                subf[dataset_path].resize(n, axis=0)
                subf[dataset_path][...] = dataset[idx, ...]