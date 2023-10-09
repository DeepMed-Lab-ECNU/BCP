import os
import sys
import shutil
import time
import random
import torch
import logging
from pathlib import Path

import numpy as np
import statistic
from torch import multiprocessing
from torch.nn import functional as F
import nibabel as nib
from tensorboardX import SummaryWriter
from skimage.measure import label

def mkdir(path, level=2, create_self=True):
    """ Make directory for this path,
    level is how many parent folders should be created.
    create_self is whether create path(if it is a file, it should not be created)

    e.g. : mkdir('/home/parent1/parent2/folder', level=3, create_self=False),
    it will first create parent1, then parent2, then folder.

    :param path: string
    :param level: int
    :param create_self: True or False
    :return:
    """
    p = Path(path)
    if create_self:
        paths = [p]
    else:
        paths = []
    level -= 1
    while level != 0:
        p = p.parent
        paths.append(p)
        level -= 1

    for p in paths[::-1]:
        p.mkdir(exist_ok=True)
        

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
        
        
def get_mask(out, thres=0.5):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).float()
    masks = masks[:, 1, :, :].contiguous()
    return masks


def save_net_opt(net, optimizer, path, epoch):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, str(path))
    
    
def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])
    

def save_net(net, path):
    state = {
        'net': net.state_dict(),
    }
    torch.save(state, str(path))
    

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    
    
def generate_mask(img, patch_size):
    batch_l = img.shape[0]
    #batch_unlab = unimg.shape[0]
    loss_mask = torch.ones(batch_l, 96, 96, 96).cuda()
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


@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)