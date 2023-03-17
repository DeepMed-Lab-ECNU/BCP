#from cProfile import label
from skimage.measure import label
from multiprocessing.spawn import import_main_path
import os
from random import shuffle
from tkinter.tix import Tree
from turtle import Turtle
import torch
import logging
import sys
import time
import itertools
import numpy as np
import pdb
import cv2

from torch.nn import functional as F
from pathlib import Path
from utils1 import statistic
from vnet import one_out_VNet, one_VNet, TMI_VNet
from torch.utils.data import DataLoader
from dataset.pancreas import Pancreas, PancreasSTDataset, cutmix_Pancreas, data_collate
from torch import import_ir_module, nn as nn, optim as optim
from tensorboardX import SummaryWriter


def cutmix_config_log(save_path, tensorboard=False):
    writer = SummaryWriter(str(save_path), filename_suffix=time.strftime('_%Y-%m-%d_%H-%M-%S')) if tensorboard else None

    save_path = str(Path(save_path) / 'log.txt')
    formatter = logging.Formatter('%(levelname)s [%(asctime)s] %(message)s')

    logger = logging.getLogger(save_path.split('/')[-2])
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(save_path)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    sh = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(sh)

    return logger, writer


class Measures():
    def __init__(self, keys, writer, logger):
        self.keys = keys
        self.measures = {k: AverageMeter() for k in self.keys}
        self.writer = writer
        self.logger = logger

    def reset(self):
        [v.reset() for v in self.measures.values()]


class CutPreMeasures(Measures):
    def __init__(self, writer, logger):
        keys = ['ce_loss', 'dice_loss', 'loss_all', 'train_dice']
        super(CutPreMeasures, self).__init__(keys, writer, logger)

    def update(self, out, lab, *args):
        args = list(args)
        masks = get_mask(out)
        train_dice1 = statistic.dice_ratio(masks, lab)
        args.append(train_dice1)

        dict_variables = dict(zip(self.keys, args))
        for k, v in dict_variables.items():
            self.measures[k].update(v)

    def log(self, epoch, step):
        # self.logger.info('epoch : %d, step : %d, train_loss: %.4f, train_dice: %.4f' % (
        #     epoch, step, self.measures['loss_all'].avg, self.measures['train_dice'].avg))

        log_string, params = 'Epoch : {}', []
        for k in self.keys:
            log_string += ', ' + k + ': {:.4f}'
            params.append(self.measures[k].val)
        self.logger.info(log_string.format(epoch, *params))

        for k, measure in self.measures.items():
            k = 'pretrain/' + k
            self.writer.add_scalar(k, measure.avg, step)
        self.writer.flush()
        

class SingleMeasures(Measures):
    def __init__(self, writer, logger):
        keys = ['loss_dice', 'lodd_ce', 'loss_all', 'train_dice']
        super(SingleMeasures, self).__init__(keys, writer, logger)

    def update(self, a_out, a_lab, *args):
        args = list(args)
        a_masks = get_mask(a_out)
        #b_masks = get_mask(b_out)
        train_dice1 = statistic.dice_ratio(a_masks, a_lab)
        #train_dice2 = statistic.dice_ratio(b_masks, b_lab)
        args.append(train_dice1)
        #args.append(train_dice2)

        dict_variables = dict(zip(self.keys, args))
        for k, v in dict_variables.items():
            self.measures[k].update(v)

    def log(self, epoch, step):
        # self.logger.info('epoch : %d, step : %d, train_loss: %.4f, train_dice: %.4f' % (
        #     epoch, step, self.measures['loss_all'].avg, self.measures['train_dice'].avg))

        log_string, params = 'Epoch : {}', []
        for k in self.keys:
            log_string += ', ' + k + ': {:.4f}'
            params.append(self.measures[k].val)
        self.logger.info(log_string.format(epoch, *params))

        for k, measure in self.measures.items():
            k = 'pretrain/' + k
            self.writer.add_scalar(k, measure.avg, step)
        self.writer.flush()


def get_mask(out, thres=0.5):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).float()
    masks = masks[:, 1, :, :].contiguous()
    return masks


def get_cut_mask(out, thres=0.5, nms=True, connect_mode=1):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms==True:
        masks = LargestCC_pancreas(masks, connect_mode=connect_mode)
    return masks


def LargestCC_pancreas(segmentation, connect_mode=1):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob, connectivity=connect_mode)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    
    return torch.Tensor(batch_list).cuda()


def get_conf_thres_mask(t_out1, t_out2, thres=0.5):
    t_probs1, t_probs2 = F.softmax(t_out1, 1), F.softmax(t_out2, 1)
    probs = t_probs1 * t_probs2
    probs = F.softmax(probs, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    return masks
    

def get_soft_mask(out, temp=0.1):
    probs = F.softmax(out, 1)
    masks = torch.pow(probs,  1/temp) / (torch.pow(probs, 1/temp) + torch.pow((1-probs), 1/temp))
    masks = masks.type(torch.long)
    masks = masks[:, 1, :, :].contiguous()
    return masks


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return self

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        return self

def create_Vnet(ema=False):
    net = one_out_VNet()
    net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def get_ema_model_and_dataloader(data_root, split_name, batch_size, lr, labelp=20):
    print("Initialize ema cutmix: network, optimizer and datasets...")
    """Net & optimizer"""
    net = create_Vnet()
    ema_net = create_Vnet(ema=True).cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    trainset_lab_a = Pancreas(data_root, split_name, split='train_lab')
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = Pancreas(data_root, split_name, split='train_lab', reverse=True)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    
    trainset_unlab_a = Pancreas(data_root, split_name, split='train_unlab')
    unlab_loader_a = DataLoader(trainset_unlab_a, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_b = Pancreas(data_root, split_name, split='train_unlab', reverse=True)
    unlab_loader_b = DataLoader(trainset_unlab_b, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    
    testset = Pancreas(data_root, split_name, split='test')
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    return net, ema_net, optimizer, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader


def get_optimizer(net_list, lr):
    """Net & optimizer"""
    params = []
    for net in net_list:
        params.append({'params': net.parameters()})
    optimizer = optim.Adam(params, lr=lr)
    return optimizer


def extend_patch(image, patch, cordi, patch_size):
    ext_patch = torch.zeros_like(image).cuda()
    ext_patch[:, cordi[0]:cordi[0]+patch_size, cordi[1]:cordi[1]+patch_size, cordi[2]:cordi[2]+patch_size] += patch
    return ext_patch