# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""
import cv2
import glob
import os
import re
import random
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

try:
    import pyspng
except ImportError:
    pyspng = None

import legacy
from datasets.mask_generator_512 import RandomMask
from networks.mat import Generator


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    missing = []
    for name, tensor in named_params_and_buffers(dst_module):
        if (name not in src_tensors) and require_all:
            missing.append(name)
            continue
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)
    if require_all and missing:
        raise AssertionError(f'Missing {len(missing)} params/buffers in source checkpoint. Example: {missing[:5]}')


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--dpath', help='the path of the input image', required=True)
@click.option('--mpath', help='the path of the mask')
@click.option('--resolution', type=int, help='resolution of input image', default=512, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--allow-missing-params', is_flag=True, default=False, help='Allow loading checkpoints that miss some current model params/buffers.')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    dpath: str,
    mpath: Optional[str],
    resolution: int,
    truncation_psi: float,
    noise_mode: str,
    allow_missing_params: bool,
    outdir: str,
):
    """
    Generate images using pretrained network pickle.
    """
    seed = 240  # pick up a random number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print(f'Loading data from: {dpath}')
    img_list = sorted(glob.glob(dpath + '/*.png') + glob.glob(dpath + '/*.jpg'))

    if mpath is not None:
        print(f'Loading mask from: {mpath}')
        mask_list = sorted(glob.glob(mpath + '/*.png') + glob.glob(mpath + '/*.jpg'))
        assert len(img_list) == len(mask_list), 'illegal mapping'

    print(f'Loading networks from: {network_pkl}')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
    net_res = 512 if resolution > 512 else resolution
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(device).eval().requires_grad_(False)
    copy_params_and_buffers(G_saved, G, require_all=not allow_missing_params)

    os.makedirs(outdir, exist_ok=True)

    # no Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    def read_image(image_path):
        with open(image_path, 'rb') as f:
            if pyspng is not None and image_path.endswith('.png'):
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
            image = np.repeat(image, 3, axis=2)
        image = image.transpose(2, 0, 1) # HWC => CHW
        image = image[:3]
        return image

    def to_image(image, lo, hi):
        image = np.asarray(image, dtype=np.float32)
        image = (image - lo) * (255 / (hi - lo))
        image = np.rint(image).clip(0, 255).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return image

    if resolution != 512:
        noise_mode = 'random'
    with torch.no_grad():
        for i, ipath in enumerate(img_list):
            iname = os.path.basename(ipath).replace('.jpg', '.png')
            print(f'Prcessing: {iname}')
            image = read_image(ipath)
            image = (torch.from_numpy(image).float().to(device) / 127.5 - 1).unsqueeze(0)

            if mpath is not None:
                mask = cv2.imread(mask_list[i], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0).unsqueeze(0)
            else:
                mask = RandomMask(resolution) # adjust the masking ratio by using 'hole_range'
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0)

            z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
            output = G(image, mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            output = output[0].cpu().numpy()
            PIL.Image.fromarray(output, 'RGB').save(f'{outdir}/{iname}')


if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Colab / Streamlit UI inference helper
# Required by demo_ui/inference_adapter.py:
#     from generate_image import load_generator_for_inference
# -------------------------------------------------------------------------

_GENERATOR_INFERENCE_CACHE = {}

def load_generator_for_inference(
    network_pkl,
    device=None,
    resolution=512,
    allow_missing_params=False,
):
    """
    Load a MAT generator checkpoint for UI/API inference.

    This mirrors generate_images():
    1. load G_ema from the .pkl checkpoint
    2. instantiate networks.mat.Generator
    3. copy params/buffers from saved checkpoint into the current Generator
    4. return eval/frozen Generator

    Args:
        network_pkl: path to .pkl checkpoint
        device: torch.device or string. Defaults to cuda if available.
        resolution: expected image resolution, default 512.
        allow_missing_params: if True, allow partial checkpoint loading.

    Returns:
        Generator ready for inference.
    """
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    network_pkl = str(network_pkl)
    net_res = 512 if int(resolution) > 512 else int(resolution)

    cache_key = (
        network_pkl,
        str(device),
        net_res,
        bool(allow_missing_params),
    )

    if cache_key in _GENERATOR_INFERENCE_CACHE:
        return _GENERATOR_INFERENCE_CACHE[cache_key]

    print(f"[load_generator_for_inference] Loading networks from: {network_pkl}")
    print(f"[load_generator_for_inference] Device: {device}")
    print(f"[load_generator_for_inference] Resolution: {net_res}")

    with dnnlib.util.open_url(network_pkl) as f:
        checkpoint = legacy.load_network_pkl(f)

    if "G_ema" not in checkpoint:
        raise KeyError(
            f"Checkpoint does not contain 'G_ema'. Available keys: {list(checkpoint.keys())}"
        )

    G_saved = checkpoint["G_ema"].to(device).eval().requires_grad_(False)

    G = Generator(
        z_dim=512,
        c_dim=0,
        w_dim=512,
        img_resolution=net_res,
        img_channels=3,
    ).to(device).eval().requires_grad_(False)

    copy_params_and_buffers(
        G_saved,
        G,
        require_all=not allow_missing_params,
    )

    _GENERATOR_INFERENCE_CACHE[cache_key] = G
    return G
