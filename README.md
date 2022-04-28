# Setup Enviroment
Make sure you have proper Nvidia Driver installed 

This will create `seg3d` enviroment based on the dependencies file. This file could be update later.
```
conda env create -f environment.yml
conda activate seg3d
```

Once finshed, install Waymo-Opendataset API using pip
```
pip3 install waymo-open-dataset-tf-2-6-0
```

# Download data
## Setup 
### Register your Google Account with Waymo
Make sure you register your google account with [Waymo Open Dataset](https://waymo.com/intl/en_us/open/download/).

### Install gcloud tool

**Windows**: 

Download and follow the installer [here](https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe)

**Linux**: 

- Download one of the following file for your architecture

x86_64:
```
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-381.0.0-linux-x86_64.tar.gz
```
Arm_64:
```
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-381.0.0-linux-arm.tar.gz

```
x86_32:
```
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-381.0.0-linux-x86.tar.gz
```

- Extract the file, replace XXXX with coresponding version you downloaded
```
tar -xf google-cloud-cli-381.0.0-linux-XXXX.tar.gz
```
- Run the install script. It will prompt a privacy question and whether to add to `PATH` to enable command completion. 
```
./google-cloud-sdk/install.sh
```

### Initialize gcloud CLI

Open a new terminal and 
```
gcloud init
```
It will prompt you to log in using Goolge user account. After answer `Y`, it will open a webpage. Make sure you log in the same Google use account you register for the dataset and click **Allow**

If you are working on a machine without browser access, it will prompt you a command to run on a machine with a browser. You might have to repeat the above instruction on how to install gcloud tool on a browser-enabled machine. You do not need to `initialize` on the browser-enabled machine. Simply copy and run the command, it will prompt the same webpage as above. Then you just need to copy the output from the CLI back to your browser-disabled machine. 

If you have done it correctly, you will see the following message
```
You are logged in as: [YOUR_EMAIL].
```
If you do not have any project on Google Cloud on your account, it will prompt you to create a project. Simply answer `N`. All following process is for initialization for a `service account` and no relevant to us. 

### Check Authentication 

To check if you have done everything correctly, you can download this 200 Mb file from motion dataset by running the following. 
```
gsutil -m cp "gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/testing/testing.tfrecord-00000-of-00150" ./
```

### (Optional) crcmod

You can install crcmod to accelerate large file downloading
```
pip3 install crcmod
```

## Command Line

Once you register you can access and get download link from Google Cloud Platform [here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_3_0;tab=objects?prefix=&forceOnObjectsSortingFiltering=false). 

Then you can use the following command 
```
gsuitl -m cp FILE_URL DES_PATH
```

## Script

You can use `download_data.py` to help you download and extract the data you want: 

**Example Usage:**

downlaod segment 0 in testing split:
```
python download_data.py download --des PATH/TO/DES --split testing --seg_id 0
```

downlaod mutiple segments in validation split:
```
python download_data.py download --des PATH/TO/DES --split validation --seg_id '[0 1 2]'
```

downlaod all segments in training split:
```
python download_data.py download --des PATH/TO/DES --split training --seg_id -1
```

downlaod segment 0 in both training and validation split:
```
python download_data.py download --des PATH/TO/DES --split '[training, validation]' --seg_id '[0, 0]'
```

downlaod mutiple segments in both training and validation split:
```
python download_data.py download --des PATH/TO/DES --split '[training, validation]' --seg_id '[[0,1,2], [0,1,2]]'
```

downlaod all segments in all split:
```
python download_data.py download --des PATH/TO/DES --split '[training, validation, testing]' --seg_id '[-1, -1, -1]'
```

# Data format
## Extract frames has segmentation label
Extract frames with segmentation label use the following code::
```
data_dir = 'PATH/TO/DATA/SPLIT/DIR'
path_txt = 'PATH/TO/EXTRACT/FRAME/TXT'

extract_3d_seg_frames(data_dir, path_txt)
```
You con find extracted frame txt file in `docs/`. `3d_semseg_test_set_frames.txt` provided by Waymo.

Each lines frame txt file of has the following format:
```
{context id},{timestamp}
ex:
2830680430134047327_1720_000_1740_000,1558034229922468
```

## Convert tfrecord to hdf5 format
Conversion can be done using the following code:
```
from utils import *

h5_path = 'PATH/TO/H5/FILE'
txt_path = 'PATH/TO/EXTRACT/FRAME/TXT'
data_dir = 'PATH/TO/DATA/SPLIT/DIR'

tfrecord_to_h5(txt_path, h5_path, data_dir)
```
Only frames with segmentation data will be converted, the hdf5 file has the following structure:
```
<HDF5 dataset "image": shape (None, 5, 800000), type "|u1">
<HDF5 dataset "image_length": shape (None, 5), type "<i4">
<HDF5 dataset "proj_pixel": shape (None, 3, 64, 2650), type "|u1">
<HDF5 group "/label" (2 members)>
    <HDF5 dataset "ri1": shape (None, 2, 64, 2650), type "<i4">
    <HDF5 dataset "ri2": shape (None, 2, 64, 2650), type "<i4">
proj <HDF5 group "/proj" (2 members)>
    <HDF5 dataset "ri1": shape (None, 6, 64, 2650), type "<i4">
    <HDF5 dataset "ri2": shape (None, 6, 64, 2650), type "<i4">
range_image <HDF5 group "/range_image" (2 members)>
    <HDF5 dataset "ri1": shape (None, 4, 64, 2650), type "<f4">
    <HDF5 dataset "ri2": shape (None, 4, 64, 2650), type "<f4">
```
See how to extract these data in `get_data_h5` in `/utils/h5_data_utils.py`
