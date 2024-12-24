import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from tools import mutils

saved_grad = None
saved_name = None

base_url = './results'
os.makedirs(base_url, exist_ok=True)


def normalize_tensor_mm(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def normalize_tensor_sigmoid(tensor):
    return nn.functional.sigmoid(tensor)


def save_image(tensor, name=None, save_path=None, exit_flag=False, timestamp=False, nrow=4, split_dir=None):
    if split_dir:
        _base_url = os.path.join(base_url, split_dir)
    else:
        _base_url = base_url
    os.makedirs(_base_url, exist_ok=True)
    import torchvision.utils as vutils
    grid = vutils.make_grid(tensor.detach().cpu(), nrow=nrow)

    if save_path:
        vutils.save_image(grid, save_path)
    else:
        if timestamp:
            vutils.save_image(grid, f'{_base_url}/{name}_{mutils.get_timestamp()}.png')
        else:
            vutils.save_image(grid, f'{_base_url}/{name}.png')
    if exit_flag:
        exit(0)


def save_feature_heatmap(tensor, name, exit_flag=False):
    save_path = f'{base_url}/{name}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    f = (torch.sum(tensor, dim=1)).squeeze(0).detach().cpu().numpy()
    plt.imshow(f, cmap="rainbow")
    plt.axis('off')
    dpi = max(tensor.shape[-2:]) / 2
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

    if exit_flag:
        exit(0)
