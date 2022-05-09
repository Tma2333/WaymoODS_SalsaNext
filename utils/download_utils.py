import os 
from typing import Union, List
from pathlib import Path
import tarfile

NUM_SEG = {'training': 32, 'validation': 8, 'testing': 8}
AVAILABLE_SPLIT = list(NUM_SEG.keys())

def download_segment (urls: List[str], des: Union[str, os.PathLike]):
    des = Path(des)
    des.mkdir(parents=True, exist_ok=True)

    base_download_cmd = 'gsutil -m cp '

    for url in urls:
        file_name = url.split('/')[-1]
        split = url.split('/')[-2]
        file_loc = des/file_name
        if not file_loc.exists():
            cmd = base_download_cmd + url + f' {str(des)}'
            flag = os.system(cmd)
        else:
            flag = 1
            print(f'{file_name} already downloaded')

        if file_loc.exists():
            (des/split).mkdir(parents=True, exist_ok=True)
            with tarfile.open(str(file_loc)) as tar:
                print(f'Extract {file_name} to {des/split}')
                tar.extractall(str(des/split))
        

def gen_urls (split: Union[str, List[str]], seg_id: Union[int, List[Union[List[int], int]]]):
    urls = []
    if isinstance(split, str):
        if isinstance(seg_id, list):
            for id_num in seg_id:
                if id_num == -1:
                    continue
                urls.append(format_url(split, id_num))
        else:
            if seg_id == -1:
                for id_num in range(NUM_SEG[split]):
                    urls.append(format_url(split, id_num))
            else:
                urls.append(format_url(split, seg_id))

    if isinstance(split, list):
        if not isinstance(seg_id, list):
            raise TypeError('If split is a list (mutiple split), seg_id must also be a list' )
        if len(split) != len(seg_id):
            raise ValueError(f'Downlaod from {len(split)} split, but only get {len(seg_id)} segment id')

        for s, ids in zip(split, seg_id):
            if isinstance(ids, list):
                for id_num in ids:
                    if id_num == -1:
                        continue
                    urls.append(format_url(s, id_num))
            else:
                if ids == -1:
                    for id_num in range(NUM_SEG[s]):
                        urls.append(format_url(s, id_num))
                else:
                    urls.append(format_url(s, ids))
    return urls


def format_url (split: str, seg_id: int):
    check_param(split, seg_id)
    base_url = 'gs://waymo_open_dataset_v_1_3_2/archived_files/'
    url = base_url + f'{split}/{split}_{seg_id:04d}.tar'
    return url



def check_param (split: str, seg_id: int):
    if split not in AVAILABLE_SPLIT:
        raise ValueError(f'{split} is not available, availabel split: {AVAILABLE_SPLIT}')
    if seg_id < 0 or seg_id >= NUM_SEG[split]:
        raise ValueError(f'{split} only has {NUM_SEG[split]} segment, but recieve {seg_id}')
    


