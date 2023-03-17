from turtle import pd
import numpy as np
import torch
import torch.nn.functional as F
import math
import pdb

from medpy import metric


def predict_patch(net, image, dim_patch=6):
    """
    dim_patch: crop image into n*n patches. (3, 6, 8...)
    """
    batch_size, _, w, h, d = image.shape
    patch_size = int(96 / dim_patch)
    score_map0 = torch.zeros(batch_size, 2, w, h, d).cuda()
    score_map1 = torch.zeros_like(score_map0).cuda()
    score_map2 = torch.zeros_like(score_map0).cuda()
    for x in range(0, 96, patch_size):
        for y in range(0, 96, patch_size):
            for z in range(0, 96, patch_size):
                batch_patch = image[:, :, x:x + patch_size, y:y + patch_size, z:z + patch_size]
                y1 = net(batch_patch)
                score_map0[:, :, x:x + patch_size, y:y + patch_size, z:z + patch_size] += y1[0]
                score_map1[:, :, x:x + patch_size, y:y + patch_size, z:z + patch_size] += y1[1]
                score_map2[:, :, x:x + patch_size, y:y + patch_size, z:z + patch_size] += y1[2]
                #pdb.set_trace()
    score_map = [score_map0, score_map1, score_map2]
    return score_map




