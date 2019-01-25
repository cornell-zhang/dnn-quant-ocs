import torch
import torch.nn as nn
import numpy as np

from .q_utils import *

#-------------------------------------------------------------------------
# For weights we can use numpy since we perform quantization once
# at the beginning of testing
#-------------------------------------------------------------------------
def ocs_wts(weight_np, expand_ratio, axis=1, split_threshold=0.5, w_scale=None, grid_aware=False):
    """ Basic net2net and split the channels into 2 equal parts """
    assert(axis == 1)
    assert(grid_aware == False or w_scale is not None)
    weight_np = weight_np.copy()

    # Identify channel to split
    # weight layout is O x I x h x w
    num_channels = weight_np.shape[axis]
    ocs_channels = int(np.ceil(expand_ratio * num_channels))

    if ocs_channels == 0:
        return weight_np, []

    # Which act channels to copy
    in_channels_to_copy = []
    # Mapping from newly added channels to the orig channels they split from
    orig_idx_dict = {}

    for c in range(ocs_channels):
        # pick the channels with the largest max values
        axes = list(range(weight_np.ndim))
        axes.remove(axis)
        max_per_channel = np.max(np.abs(weight_np), axis=tuple(axes))
        # Sort and compute which channel to split
        idxs = np.flip(np.argsort(max_per_channel), axis=0)
        split_idx = idxs[0]

        # Split channel
        ch_slice = weight_np[:, split_idx:(split_idx+1), :, :].copy()

        ch_slice_half = ch_slice / 2.
        ch_slice_zero = np.zeros_like(ch_slice)
        split_value = np.max(ch_slice) * split_threshold

        if not grid_aware:
            ch_slice_1 = np.where(np.abs(ch_slice) > split_value, ch_slice_half, ch_slice)
            ch_slice_2 = np.where(np.abs(ch_slice) > split_value, ch_slice_half, ch_slice_zero)
        else:
            ch_slice_half *= w_scale
            ch_slice_1 = np.where(np.abs(ch_slice) > split_value, ch_slice_half-0.25, ch_slice*w_scale) / w_scale
            ch_slice_2 = np.where(np.abs(ch_slice) > split_value, ch_slice_half+0.25, ch_slice_zero)    / w_scale

        weight_np[:, split_idx:(split_idx+1), :, :] = ch_slice_1
        weight_np = np.concatenate((weight_np, ch_slice_2), axis=axis)

        # Record which channel was split
        if split_idx < num_channels:
            in_channels_to_copy.append(split_idx)
            orig_idx_dict[num_channels+c] = split_idx
        else:
            idx_to_copy = orig_idx_dict[split_idx]
            in_channels_to_copy.append(idx_to_copy)
            orig_idx_dict[num_channels+c] = idx_to_copy

    return weight_np, in_channels_to_copy
