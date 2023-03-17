from locale import normalize
from multiprocessing import reduction
import pdb
from turtle import pd
import numpy as np
import torch.nn as nn
import torch
import random
import torch.nn.functional as F
from utils1.loss import DiceLoss, softmax_mse_loss
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

DICE = DiceLoss(nclass=2)
CE = nn.CrossEntropyLoss(reduction='none')
MSE_loss = torch.nn.MSELoss(reduce=False, size_average=False)
# aa = torch.randn(4, 1, 32, 32, 32)
# dim_patch = 4
# patch_size = int(32 / dim_patch)
# for x in range(0, dim_patch):
#     for y in range(0, dim_patch):
#         for z in range(0, dim_patch):
#             batch_patch = aa[:, :, x:x + patch_size, y:y + patch_size, z:z + patch_size]
#             pdb.set_trace()
# dim_patch = 6
# patch_size = int(96 / dim_patch)
# tensor_one = torch.ones(2, 1, 96, 96, 96)
# tensor_zero = torch.zeros_like(tensor_one)
# for x in range(0, 96, patch_size):
#     for y in range(0, 96, patch_size):
#         for z in range(0, 96, patch_size):
#             tensor_zero[:, :, x:x+patch_size, y:y+patch_size, z:z+patch_size] = tensor_one[:, :, x:x+patch_size, y:y+patch_size, z:z+patch_size]
# pdb.set_trace()
def random_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*2/3), int(img_y*2/3), int(img_z*2/3)
    mask_num = 27
    mask_size_x, mask_size_y, mask_size_z = int(patch_pixel_x/3)+1, int(patch_pixel_y/3)+1, int(patch_pixel_z/3)
    size_x, size_y, size_z = int(img_x/3), int(img_y/3), int(img_z/3)
    for xs in range(3):
        for ys in range(3):
            for zs in range(3):
                w = np.random.randint(xs*size_x, (xs+1)*size_x - mask_size_x - 1)
                h = np.random.randint(ys*size_y, (ys+1)*size_y - mask_size_y - 1)
                z = np.random.randint(zs*size_z, (zs+1)*size_z - mask_size_z - 1)
                mask[w:w+mask_size_x, h:h+mask_size_y, z:z+mask_size_z] = 0
                loss_mask[:, w:w+mask_size_x, h:h+mask_size_y, z:z+mask_size_z] = 0
    return mask.long(), loss_mask.long()

def context_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*2/3), int(img_y*2/3), int(img_z*2/3)
    w = np.random.randint(0, 112 - patch_pixel_x)
    h = np.random.randint(0, 112 - patch_pixel_y)
    z = np.random.randint(0, 80 - patch_pixel_z)
    mask[w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    loss_mask[:, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    return mask.long(), loss_mask.long()

def concate_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    z_length = int(img_z * 8 / 27)
    z = np.random.randint(0, img_z - z_length -1)
    mask[:, :, z:z+z_length] = 0
    loss_mask[:, :, :, z:z+z_length] = 0
    return mask.long(), loss_mask.long()

def generate_mask(img, patch_ratio):
    batch_l = img.shape[0]
    #batch_unlab = unimg.shape[0]
    loss_mask = torch.ones(batch_l, 96, 96, 96).cuda()
    patch_size = int(96 * patch_ratio)
    #loss_mask_unlab = torch.ones(batch_unlab, 96, 96, 96).cuda()
    mask = torch.ones(96, 96, 96).cuda()
    w = np.random.randint(0, 96 - patch_size)
    h = np.random.randint(0, 96 - patch_size)
    z = np.random.randint(0, 96 - patch_size)
    mask[w:w+patch_size, h:h+patch_size, z:z+patch_size] = 0
    loss_mask[:, w:w+patch_size, h:h+patch_size, z:z+patch_size] = 0
    #loss_mask_unlab[:, w:w+patch_size, h:h+patch_size, z:z+patch_size] = 0
    #cordi = [w, h, z]
    return mask.long(), loss_mask.long()

def generate_LA_mask(img):
    patch_size_x, patch_size_y, patch_size_z = 76, 76, 53
    batch_l = img.shape[0]
    #batch_unlab = unimg.shape[0]
    loss_mask_l = torch.ones(batch_l, 112, 112, 80).cuda()
    #loss_mask_unlab = torch.ones(batch_unlab, 112, 112, 80).cuda()
    mask = torch.ones(112, 112, 80).cuda()
    w = np.random.randint(0, 112 - patch_size_x)
    h = np.random.randint(0, 112 - patch_size_y)
    z = np.random.randint(0, 80 - patch_size_z)
    mask[w:w+patch_size_x, h:h+patch_size_y, z:z+patch_size_z] = 0
    loss_mask_l[:, w:w+patch_size_x, h:h+patch_size_y, z:z+patch_size_z] = 0
    #loss_mask_unlab[:, w:w+patch_size_x, h:h+patch_size_y, z:z+patch_size_z] = 0
    #cordi = [w, h, z]
    return mask.long(), loss_mask_l.long()

def generate_kits_mask(img):
    b, c, d, w, h = img.shape
    pd, pw, ph = int(d*2/3), int(w*2/3), int(h*2/3)
    loss_mask = torch.ones(b, d, w, h).cuda()
    mask = torch.ones(d, w, h).cuda()
    s_d = np.random.randint(0, d-pd)
    s_w = np.random.randint(0, w-pw)
    s_h = np.random.randint(0, h-ph)
    mask[s_d:s_d+pd, s_w:s_w+pw, s_h:s_h+ph] = 0
    loss_mask[:, s_d:s_d+pd, s_w:s_w+pw, s_h:s_h+ph] = 0
    return mask, loss_mask

def mix_input(image_a, image_b, mask, cordi, one_train=True):
    patch_size = 64
    if one_train:
        image_a = image_a * mask + image_b * (1-mask)
    else:
        image_a = image_a * mask
        image_a[:, :, cordi[0]:cordi[0]+patch_size, cordi[1]:cordi[1]+patch_size, cordi[2]:cordi[2]+patch_size] += image_b
    return image_a

def mix_loss(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    dice_loss = DICE(net3_output, img_l, mask) * image_weight 
    dice_loss += DICE(net3_output, patch_l, patch_mask) * patch_weight
    loss_ce = image_weight * (CE(net3_output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(net3_output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    loss = (dice_loss + loss_ce) / 2
    return loss

def mix_loss_DCM(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    """
    DCM: D_Dice, C_CrossEntropy, M_MSE
    """
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    # dice_loss = DICE(net3_output, img_l, mask) * image_weight 
    # dice_loss += DICE(net3_output, patch_l, patch_mask) * patch_weight
    loss_ce = image_weight * (CE(net3_output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(net3_output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    n3_mse = F.softmax(net3_output, dim=1)
    n3_mse = n3_mse[:, 1, ...].squeeze()
    loss_mse = image_weight * (MSE_loss(n3_mse, img_l.float()) * mask).sum() / (mask.sum() + 1e-16)
    loss_mse += patch_weight * (MSE_loss(n3_mse, patch_l.float()) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    loss = (loss_mse + loss_ce) / 2
    return loss

def mix_kits_loss(net3_output, img_l, patch_l, mask, criterion, u_weight=0.5, unlab=False):
    image_weight, patch_weight = 1, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, 1
    patch_mask = 1 - mask
    ce_loss = criterion[0](net3_output, img_l, mask, num_classes=2) * image_weight
    ce_loss += criterion[0](net3_output, patch_l, patch_mask, num_classes=2) * patch_weight
    dice_loss = criterion[1](net3_output, img_l, mask) * image_weight
    dice_loss += criterion[1](net3_output, patch_l, patch_mask) * patch_weight
    loss = (dice_loss + ce_loss) /2
    return loss

def mix_supervise_loss(net3_output, img_l, patch_l, mask):
    patch_mask = 1 - mask
    dice_loss = DICE(net3_output, img_l, mask)
    dice_loss += DICE(net3_output, patch_l, patch_mask)
    loss_ce = (CE(net3_output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += (CE(net3_output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    loss = (dice_loss + loss_ce) / 2
    return loss

def crop_96_2_64(tensor_96, cordi):
    batch_size = tensor_96.shape[0]
    channel_size = tensor_96.shape[1]
    tensor_64 = torch.zeros(batch_size, channel_size, 64, 64, 64).cuda()
    tensor_64 = tensor_64 + tensor_96[:, :, cordi[0]:cordi[0]+64, cordi[1]:cordi[1]+64, cordi[2]:cordi[2]+64]
    return tensor_64

def crop_lab(lab, cordi):
    batch_size = lab.shape[0]
    lab_64 = torch.zeros(batch_size, 64, 64, 64).cuda()
    lab_64 = lab_64 + lab[:, cordi[0]:cordi[0]+64, cordi[1]:cordi[1]+64, cordi[2]:cordi[2]+64]
    return lab_64

def compute_sdf(img_gt, out_shape):
    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis - np.min(negdis)) / (np.max(negdis - np.min(negdis))) - (posdis - np.min(posdis)) / (np.max(posdis) - np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b] = sdf
    
    gt_dis = torch.from_numpy(normalized_sdf).float().cuda()
    return gt_dis

