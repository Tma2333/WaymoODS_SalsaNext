import io
from multiprocessing.sharedctypes import Value
from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
import h5py


KEY_TO_H5PATH = {'ri1_range_image': '/range_image/ri1',
                 'ri2_range_image': '/range_image/ri2',
                 'range_image':['/range_image/ri1', '/range_image/ri2'],
                 'ri1_label': '/label/ri1',
                 'ri2_label': '/label/ri2',
                 'ri1_proj': '/proj/ri1', 
                 'ri2_proj': '/proj/ri2',
                 'proj_pixel': 'proj_pixel'}


class H5Dataset(torch.utils.data.Dataset):
    def __init__ (self, h5_path, data_key_list, test=False):
        self.h5_path = Path(h5_path)

        if test and ('ri1_label' in data_key_list or 'ri2_label' in data_key_list):
            raise ValueError(f'Test set does not contain label.')

        self.data_key_list = data_key_list

        self.to_tensor = T.ToTensor()

        with h5py.File(str(self.h5_path), 'r') as f:
            self.dataset_size = len(f['/range_image/ri1'])


    def __len__ (self):
        return self.dataset_size


    def __getitem__ (self, index):
        return self.get_data(index)
    

    def norm_data (self, data, key):
        if key == 'image':
            return self.to_tensor(data.astype(np.uint8))
        elif key == 'proj_pixel':
            return self.to_tensor(data.transpose(1, 2, 0))
        elif key in ['ri1_range_image', 'ri2_range_image']:
            data[0, ...] = (data[0, ...] + 1) / 76
            data[1, ...] = np.clip(data[1, ...], a_min=-1, a_max=1.5)
            data[1, ...] = (data[1, ...] + 1) / 2.5
            data[2, ...] = (data[2, ...] + 1) / 2.5
            return torch.tensor(data, dtype=torch.float32)
        elif key == 'range_image':
            data[0, ...] = (data[0, ...] + 1) / 76
            data[1, ...] = np.clip(data[1, ...], a_min=-1, a_max=1.5)
            data[1, ...] = (data[1, ...] + 1) / 2.5
            data[2, ...] = (data[2, ...] + 1) / 2.5
            data[3, ...] = (data[3, ...] + 1) / 76
            data[4, ...] = np.clip(data[4, ...], a_min=-1, a_max=1.5)
            data[4, ...] = (data[4, ...] + 1) / 2.5
            data[5, ...] = (data[5, ...] + 1) / 2.5
            return torch.tensor(data, dtype=torch.float32)
        else:
            return torch.tensor(data)


    def get_data (self, index):
        data = {}
        with h5py.File(str(self.h5_path), 'r') as f:
            for key in self.data_key_list:
                if key == 'image':
                    data[key] = {}
                    for i in range(5):
                        cam_id = i+1
                        byte_length = f['image_length'][index, i]
                        byte_data = f['image'][index, i, :byte_length]
                        image_data = Image.open(io.BytesIO(byte_data))
                        image_data = np.array(image_data)
                        
                        im_tensor = self.norm_data(image_data, key)
                        data[key][cam_id] = im_tensor

                elif key == 'range_image':
                    ri1 = f[KEY_TO_H5PATH[key][0]][index, :3, ...]
                    ri2 = f[KEY_TO_H5PATH[key][1]][index, :3, ...]
                    np_data = np.concatenate([ri1, ri2], 0)
                    data[key] = self.norm_data(np_data, key)

                elif key in ['ri1_range_image', 'ri2_range_image']:
                    np_data = f[KEY_TO_H5PATH[key]][index, :3, ...]
                    data[key] = self.norm_data(np_data, key)

                elif key in KEY_TO_H5PATH:
                    np_data = f[KEY_TO_H5PATH[key]][index, ...]
                    data[key] = self.norm_data(np_data, key)

                else:
                    raise ValueError(f'{key} is not a valid key!')
        return data


# class H5Dataset(torch.utils.data.Dataset):
#     def __init__ (self, dataset_root_path, is_train, train_split=0.7, transforms=None):
#         self.root = Path(dataset_root_path)
#         self.data_path = self.root/'data.h5'
#         self.metadata_path = self.root/'metadata.yaml'

#         with open(self.metadata_path) as f:
#             self.metadata = yaml.load(f, Loader=yaml.CLoader)
        
#         mean = []
#         std = []
#         for data_range in self.metadata['data_range']:
#             mean.append((data_range[1]+data_range[0])/2)
#             std.append((data_range[1]-data_range[0])/2)

#         if transforms is None:
#             self.transforms = []
#         else:
#             self.transforms = transforms
#         self.transforms.append(T.ToTensor())
#         self.transforms.append(T.ConvertImageDtype(torch.float))
#         self.transforms.append(T.Normalize(mean, std))
#         self.transforms = T.Compose(self.transforms)

#         with h5py.File(str(self.data_path), 'r') as f:
#             dataset_size = len(f['image'])

#         if train_split < 1:
#             if is_train:
#                 self.length = int(dataset_size * train_split)
#                 self.offset = 0
#             else:
#                 self.length = int(dataset_size - int(dataset_size * train_split))
#                 self.offset = int(dataset_size * train_split)


#     def __len__ (self):
#         return self.length

#     def __getitem__ (self, index):
#         im_np = self._hdf5_loader(index+self.offset)
#         return self.transforms(im_np)


#     def _hdf5_loader (self, index):
#         with h5py.File(str(self.data_path), 'r') as f:
#             dset = f['image']
#             return np.moveaxis(dset[index, ...], 0, -1)
