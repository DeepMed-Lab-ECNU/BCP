import os
import torch


def view_augment(arr, view_mode):
    pp = [(0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2)]
    new = torch.zeros_like(arr)
    for x in range(arr.shape[0]):
        new[x] = arr[x].permute(pp[view_mode])
    return new


def view_re_augment(arr, view_mode):
    pp = [(0, 1, 2, 3), (0, 3, 1, 2), (0, 2, 3, 1)]
    new = torch.zeros_like(arr)
    for i in range(arr.shape[0]):
        new[i] = arr[i].permute(pp[view_mode])
    return new

