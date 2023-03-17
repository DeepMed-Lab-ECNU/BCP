import logging
import random
from struct import pack
import sys
import time
import os
from tkinter import N

import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn as nn, optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset.pancreas import Pancreas, PancreasSTDataset
from resnet18_3d import AHNet
from utils1 import statistic
from vnet import VNet
from utils1.patch_utils import predict_patch

def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = False


def seed_reproducer(seed=2022):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2020)
        Radnom seed.

    Example
    -------
    seed_reproducer(seed=2020).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)#set all gpus seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False#if input data type and channels' changes arent' large use it improve train efficient
        torch.backends.cudnn.enabled = True



def to_cuda(tensors, device=None):
    res = []
    if isinstance(tensors, (list, tuple)):
        for t in tensors:
            res.append(to_cuda(t, device))
        return res
    elif isinstance(tensors, (dict,)):
        res = {}
        for k, v in tensors.items():
            res[k] = to_cuda(v, device)
        return res
    else:
        if isinstance(tensors, torch.Tensor):
            if device is None:
                return tensors.cuda()
            else:
                return tensors.to(device)
        else:
            return tensors


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


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_current_consistency_weight(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)


def create_model(res18=False, ema=False):
    net = AHNet() if res18 else VNet()
    net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def get_model_and_dataloader(data_root, split_name, batch_size, lr, res18=False):
    print("Initialize network, optimizer and datasets...")
    """Net & optimizer"""
    net = create_model(res18)
    ema_net = create_model(res18, ema=True).cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))

    trainset_lab = Pancreas(data_root, split_name, split='train_lab')
    lab_loader = DataLoader(trainset_lab, batch_size=batch_size, shuffle=False, num_workers=0)

    trainset_unlab = Pancreas(data_root, split_name, split='train_unlab', no_crop=True)
    unlab_loader = DataLoader(trainset_unlab, batch_size=1, shuffle=False, num_workers=0)

    testset = Pancreas(data_root, split_name, split='test')
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    return net, ema_net, optimizer, lab_loader, unlab_loader, test_loader


def get_two_model_and_dataloder(data_root, split_name, batch_size, lr, res18=False):
    print("Initialize networks, optimizer and datasets...")
    """Net & optimizer"""
    net1 = create_model(res18)
    net2 = create_model(res18)
    optimizer = optim.Adam([{'params': net1.parameters()},
                            {'params': net2.parameters()}],
                            lr=lr, betas=(0.5, 0.999))
    trainset_lab = Pancreas(data_root, split_name, split='train_lab')
    lab_loader = DataLoader(trainset_lab, batch_size=batch_size, shuffle=False, num_workers=0)

    trainset_unlab = Pancreas(data_root, split_name, split='train_unlab', no_crop=True)
    unlab_loader = DataLoader(trainset_unlab, batch_size=1, shuffle=False, num_workers=0)

    testset = Pancreas(data_root, split_name, split='test')
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    return net1, net2, optimizer, lab_loader, unlab_loader, test_loader


def save_net_opt(net, optimizer, path, epoch):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, str(path))

def save_net(net, path):
    state = {
        'net': net.state_dict(),
    }
    torch.save(state, str(path))

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_2_net_opt(net1, net2, optimizer, path, epoch):
    state = {
        'net1':net1.state_dict(),
        'net2':net2.state_dict(),
        'opt':optimizer.state_dict(),
        'epoch':epoch,
    }
    torch.save(state, str(path))

def load_2_net_opt(net1, net2, optimizer, path):
    state = torch.load(str(path))
    net1.load_state_dict(state['net1'])
    net2.load_state_dict(state['net2'])
    optimizer.load_state_dict(state['opt'])

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def get_mask(out, thres=0.5):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).float()
    masks = masks[:, 1, :, :].contiguous()
    return masks


@torch.no_grad()
def pred_unlabel(net, pred_loader, batch_size):
    unimg, unlab, unmask, labs = [], [], [], []
    plab_dice = 0
    for (step, data) in enumerate(pred_loader):
        img, lab = data
        img, lab = img.cuda(), lab.cuda()
        out = net(img)
        #p_out = predict_patch(net, img, dim_patch=3)

        plab0 = get_mask(out[0])
        plab1 = get_mask(out[1])
        plab2 = get_mask(out[2])

        # pplab0 = get_mask(p_out[0])
        # pplab1 = get_mask(p_out[1])
        # pplab2 = get_mask(p_out[2])

        # plab0 = 0.5 * plab0 + 0.5 * pplab0
        # plab1 = 0.5 * plab1 + 0.5 * pplab1
        # plab2 = 0.5 * plab2 + 0.5 * pplab2
        
        mask = (plab1 == plab2).long()
        plab = plab0
        unimg.append(img)
        unlab.append(plab)
        unmask.append(mask)
        labs.append(lab)

        plab_dice += statistic.dice_ratio(plab, lab)
    plab_dice /= len(pred_loader)
    new_loader = DataLoader(PancreasSTDataset(unimg, unlab, unmask, labs), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return new_loader, plab_dice


def config_log(save_path, tensorboard=False):
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


class PretrainMeasures(Measures):
    def __init__(self, writer, logger):
        keys = ['loss_ce', 'loss_dice', 'loss_con', 'loss_rad', 'loss_all', 'train_dice']
        super(PretrainMeasures, self).__init__(keys, writer, logger)

    def update(self, out, lab, *args):
        args = list(args)
        masks = get_mask(out[0])
        train_dice = statistic.dice_ratio(masks, lab)
        args.append(train_dice)

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


class CutmixFTMeasures(Measures):
    def __init__(self, writer, logger):
        keys = ['mix_loss_lab', 'mix_loss_unlab', 'loss_all']
        super(CutmixFTMeasures, self).__init__(keys, writer, logger)

    def update(self, *args):
        args = list(args)
        # masks = get_mask(out[0])
        # train_dice = statistic.dice_ratio(masks, lab)
        # args.append(train_dice)

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


class TMIMeasures(Measures):
    def __init__(self, writer, logger):
        keys = ['loss_lab', 'loss_unlab', 'loss_mse', 'loss_all']
        super(TMIMeasures, self).__init__(keys, writer, logger)

    def update(self, *args):
        args = list(args)
        # masks = get_mask(out[0])
        # train_dice = statistic.dice_ratio(masks, lab)
        # args.append(train_dice)

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


class DTCFTMeasures(Measures):
    def __init__(self, writer, logger):
        keys = ['l_seg', 'ul_seg', 'loss_all']
        super(DTCFTMeasures, self).__init__(keys, writer, logger)

    def update(self, *args):
        args = list(args)
        # masks = get_mask(out[0])
        # train_dice = statistic.dice_ratio(masks, lab)
        # args.append(train_dice)

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


class STMeasures(Measures):
    def __init__(self, writer, logger):
        keys = ['train_loss', 'sup_all_loss', 'sup_ce_loss', 'sup_rad_loss', 'sup_con_loss', 'sup_dice_loss',
                'certain_all_loss', 'certain_ce_loss', 'certain_rad_loss', 'certain_con_loss', 'uncertain_loss',
                'train_dice', 'unlab_dice', 'unlab_rad_dice', 'unlab_con_dice', 'lab_con_dice', 'lab_rad_dice']
        super(STMeasures, self).__init__(keys, writer, logger)

    @torch.no_grad()
    def update(self, out1, out2, lab1, lab2, *args):
        mask1 = get_mask(out1[0])
        mask2 = get_mask(out2[0])
        dices = [statistic.dice_ratio(mask1, lab1), statistic.dice_ratio(mask2, lab2), statistic.dice_ratio(get_mask(out1[2]), lab1),
                 statistic.dice_ratio(get_mask(out2[2]), lab2), statistic.dice_ratio(get_mask(out1[1]), lab1),
                 statistic.dice_ratio(get_mask(out2[1]), lab2)]
        args = list(args)
        args.extend(dices)
        dict_variables = dict(zip(self.keys, args))

        for k, v in dict_variables.items():
            self.measures[k].update(v)

    def log(self, epoch):
        log_keys = ['train_loss', 'sup_all_loss', 'certain_all_loss', 'uncertain_loss',
                    'train_dice', 'unlab_dice', 'lab_rad_dice', 'lab_con_dice', 'unlab_rad_dice', 'unlab_con_dice']
        log_string = 'Epoch : {}'
        params = []
        for k in log_keys:
            log_string += ', ' + k + ': {:.4f}'
            params.append(self.measures[k].val)
        self.logger.info(log_string.format(epoch, *params))

    def write_tensorboard(self, epoch):
        for k, measure in self.measures.items():
            if 'sup' in k or 'train_loss' in k:
                k = 'supervised_loss/' + k
            elif 'certain' in k:
                k = 'upsupervised_loss/' + k
            else:
                k = 'dice/' + k
            self.writer.add_scalar(k, measure.avg, epoch)
        self.writer.flush()




