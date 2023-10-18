from asyncore import write
from audioop import avg
from cgi import test
import imp
from multiprocessing import reduction
from turtle import pd
from unittest import loader, result

from yaml import load
import torch
import os
import pdb
import torch.nn as nn

from tqdm import tqdm as tqdm_load
from pancreas_utils import *
from test_util import *
from losses import DiceLoss, softmax_mse_loss, mix_loss
from dataloaders import get_ema_model_and_dataloader

"""Global Variables"""
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
seed_test = 2020
seed_reproducer(seed = seed_test)

data_root, split_name = '/home/ubuntu/byh/code/CoraNet-master/preprocess/data', 'pancreas'
result_dir = 'result/cutmix/'
mkdir(result_dir)
batch_size, lr = 2, 1e-3
pretraining_epochs, self_training_epochs = 60, 200
pretrain_save_step, st_save_step, pred_step = 20, 20, 5
alpha, consistency, consistency_rampup = 0.99, 0.1, 40
label_percent = 20
u_weight = 1.5
connect_mode = 2
try_second = 1
sec_t = 0.5
self_train_name = 'self_train'

sub_batch = int(batch_size/2)
consistency_criterion = softmax_mse_loss
CE = nn.CrossEntropyLoss()
CE_r = nn.CrossEntropyLoss(reduction='none')
DICE = DiceLoss(nclass=2)
patch_size = 64

logger = None
overall_log = 'cutmix_log.txt'

def pretrain(net1, optimizer, lab_loader_a, labe_loader_b, test_loader):
    """pretrain image- & patch-aware network"""

    """Create Path"""
    save_path = Path(result_dir) / 'pretrain'
    save_path.mkdir(exist_ok=True)

    """Create logger and measures"""
    global logger
    logger, writer = cutmix_config_log(save_path, tensorboard=True)
    logger.info("cutmix Pretrain, patch_size: {}, save path: {}".format(patch_size, str(save_path)))

    max_dice = 0
    measures = CutPreMeasures(writer, logger)
    for epoch in tqdm_load(range(1, pretraining_epochs + 1), ncols=70):
        measures.reset()
        """Testing"""
        if epoch % pretrain_save_step == 0:
            avg_metric1 = test_calculate_metric(net1, test_loader.dataset)
            logger.info('average metric is : {}'.format(avg_metric1))
            val_dice = avg_metric1[0]

            if val_dice > max_dice:
                save_net_opt(net1, optimizer, save_path / f'best_ema{label_percent}_pre.pth', epoch)
                max_dice = val_dice
            
            writer.add_scalar('test_dice', val_dice, epoch)
            logger.info('Evaluation: val_dice: %.4f, val_maxdice: %.4f '%(val_dice, max_dice))
            save_net_opt(net1, optimizer, save_path / ('%d.pth' % epoch), epoch)
        
        """Training"""
        net1.train()
        for step, ((img_a, lab_a), (img_b, lab_b)) in enumerate(zip(lab_loader_a, lab_loader_b)):
            img_a, img_b, lab_a, lab_b  = img_a.cuda(), img_b.cuda(), lab_a.cuda(), lab_b.cuda()
            img_mask, loss_mask = generate_mask(img_a, patch_size)   

            img = img_a * img_mask + img_b * (1 - img_mask)
            lab = lab_a * img_mask + lab_b * (1 - img_mask)

            out = net1(img)[0]
            ce_loss1 = F.cross_entropy(out, lab)
            dice_loss1 = DICE(out, lab)
            loss = (ce_loss1 + dice_loss1) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            measures.update(out, lab, ce_loss1, dice_loss1, loss)
            measures.log(epoch, epoch * len(lab_loader_a) + step)
        writer.flush()
    return max_dice

def ema_cutmix(net, ema_net, optimizer, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader):
    """Create Path"""
    save_path = Path(result_dir) / self_train_name
    save_path.mkdir(exist_ok=True)

    """Create logger and measures"""
    global logger 
    logger, writer = config_log(save_path, tensorboard=True)
    logger.info("EMA_training, save_path: {}".format(str(save_path)))
    measures = CutmixFTMeasures(writer, logger)

    """Load Model"""
    pretrained_path = Path(result_dir) / 'pretrain'
    load_net_opt(net, optimizer, pretrained_path / f'best_ema{label_percent}_pre.pth')
    load_net_opt(ema_net, optimizer, pretrained_path / f'best_ema{label_percent}_pre.pth')
    logger.info('Loaded from {}'.format(pretrained_path))

    max_dice = 0
    max_list = None
    for epoch in tqdm_load(range(1, self_training_epochs+1)):
        measures.reset()
        logger.info('')

        """Testing"""
        if epoch % st_save_step == 0:
            avg_metric = test_calculate_metric(net, test_loader.dataset)
            logger.info('average metric is : {}'.format(avg_metric))
            val_dice = avg_metric[0]
            writer.add_scalar('val_dice', val_dice, epoch)

            """Save Model"""
            if val_dice > max_dice:
                save_net(net, str(save_path / f'best_ema_{label_percent}_self.pth'))
                max_dice = val_dice
                max_list = avg_metric

            logger.info('Evaluation: val_dice: %.4f, val_maxdice: %.4f' % (val_dice, max_dice))

        """Training"""
        net.train()
        ema_net.train()
        for step, ((img_a, lab_a), (img_b, lab_b), (unimg_a, unlab_a), (unimg_b, unlab_b)) in enumerate(zip(lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b)):
            img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b = to_cuda([img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b])
            """Generate Pseudo Label"""
            with torch.no_grad():
                unimg_a_out = ema_net(unimg_a)[0]
                unimg_b_out = ema_net(unimg_b)[0]
                uimg_a_plab = get_cut_mask(unimg_a_out, nms=True, connect_mode=connect_mode)
                uimg_b_plab = get_cut_mask(unimg_b_out, nms=True, connect_mode=connect_mode)
                img_mask, loss_mask = generate_mask(img_a, patch_size)     
            
            # """Mix input"""
            net3_input_l = unimg_a * img_mask + img_b * (1 - img_mask)
            net3_input_unlab = img_a * img_mask + unimg_b * (1 - img_mask)

            """Supervised Loss"""
            mix_output_l = net(net3_input_l)[0]
            loss_1 = mix_loss(mix_output_l, uimg_a_plab.long(), lab_b, loss_mask, unlab=True)

            """Unsupervised Loss"""
            mix_output_2 = net(net3_input_unlab)[0]
            loss_2 = mix_loss(mix_output_2, lab_a, uimg_b_plab.long(), loss_mask)

            loss = loss_1 + loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(net, ema_net, alpha)

            measures.update(loss_1, loss_2, loss)  
            measures.log(epoch, epoch*len(lab_loader_a) + step)

        if epoch ==  self_training_epochs:
            save_net(net, str(save_path / f'best_ema_{label_percent}_self_latest.pth'))
        writer.flush()
    return max_dice, max_list

def test_model(net, test_loader):
    load_path = Path(result_dir) / self_train_name
    load_net(net, load_path / 'best_2.pth')
    print('Successful Loaded')
    avg_metric, m_list = test_calculate_metric(net, test_loader.dataset, s_xy=16, s_z=4)
    test_dice = avg_metric[0]
    return avg_metric, m_list


if __name__ == '__main__':
    try:
        net, ema_net, optimizer, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader = get_ema_model_and_dataloader(data_root, split_name, batch_size, lr, labelp=label_percent)
        pretrain(net, optimizer, lab_loader_a, lab_loader_b, test_loader)
        ema_cutmix(net, ema_net, optimizer, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader)
        avg_metric, m_list = test_model(net, test_loader)
    except Exception as e:
        logger.exception("BUG FOUNDED ! ! !")


