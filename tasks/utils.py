from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt 


def log_spherical_image_helper (data_list, vmin, vmax, cmap):
    num_ax = len(data_list)
    fig = plt.figure(figsize=(120, num_ax*3))
    for i, data in enumerate(data_list):
        data = data.clone().data.cpu().numpy()
        ax = fig.add_subplot(num_ax, 1, i+1)
        ax.imshow(data, cmap=cmap[i], vmin=vmin[i], vmax=vmax[i])
        ax.axis('off')
    plt.ioff()
    return fig
