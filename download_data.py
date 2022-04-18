import fire

from utils import gen_urls, download_segment

def download (des, split, seg_id):
    '''
    download data from Waymo Open Dataset: Perception

    Parameters
    ----------
    des : str
        Path of save destination
    split : str
        Name of data split
    seg_id: list
        Segment id to download
    '''

    urls = gen_urls(split, seg_id)
    download_segment(urls, des)


if __name__ == '__main__':
    fire.Fire()
