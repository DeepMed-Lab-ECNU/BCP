from tkinter import N
import h5py
import os
import argparse
from scipy.fftpack import ss_diff
import torch
import pdb
import seaborn as sns
import pandas as pd
import random
import shutil
import cv2

from networks.net_factory import net_factory
from dataloaders.dataset import *
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/byh_data/SSNet_data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='BCP', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--stage_name', type=str, default='self_train', help='self or pre')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--saved_path', type=str, default='visible/previous_models/ACDC/')
FLAGS = parser.parse_args()

model_names = ['BCP']
# -- Patient number of trained--
# 3_labeled: 099, 038, 050
# 7_labeled: 099, 038, 050, 100, 058, 021, 049
BCP_model_path = FLAGS.saved_path + f'labeled_{FLAGS.labelnum}/BCP.pth'

p_num = 500
bw_ad = 0.5
line_wid = 5

def get_ACDC_masks(output):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)     
    return probs

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def plot_kde(BCP_feature,BCP_pred, labels, specific_c, f_dim, pic_num):
    total_pixel, total_fdim = BCP_feature.shape[0], BCP_feature.shape[1]
    labeled_pixel = int(total_pixel / 2) + 1
    save_path = f"KDE/ACDC/{f_dim}/labeled_{FLAGS.labelnum}/class_{specific_c}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # # Chose specific class pixels:
    # BCP_pred=[1, -1, 1, -1, 1, -1, 1, -1, 1, -1] 
    l_pred, u_pred = np.where(BCP_pred[:labeled_pixel,:]==specific_c), np.where(BCP_pred[labeled_pixel:,:]==specific_c)
    l_lab, u_lab = np.where(labels[:labeled_pixel,:]==specific_c), np.where(labels[labeled_pixel:,:]==specific_c)
    correct_cor_l, correct_cor_u = np.intersect1d(l_pred[0], l_lab[0]), np.intersect1d(u_pred[0], u_lab[0]) + labeled_pixel
    # l_lab, u_lab = np.where(labels[:labeled_pixel,]==specific_c), np.where(labels[labeled_pixel:,]==specific_c)
    # l_len, u_len = len(l_lab[0]), len(u_lab[0])
    pixel_num = min(len(correct_cor_l), len(correct_cor_u), p_num)
    #l_lab, u_lab = l_lab[0], u_lab[0] + labeled_pixel
    print(f"Total {pixel_num} pixels for class {specific_c}")
    BCP_feature_l, BCP_feature_u = np.mean(BCP_feature[correct_cor_l[:pixel_num],], axis=1), np.mean(BCP_feature[correct_cor_u[:pixel_num],], axis=1)

    method_name_list = ["BCP"]
    feature_list = [BCP_feature_l, BCP_feature_u]

    plt.figure()
    fig = plt.figure(figsize=(29, 4))
    sns.set_context("notebook", font_scale=2)
    for i in range(0, 1):
        plt.subplot(1, 1, i+1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.3, hspace=None)
        sns.kdeplot(feature_list[0], bw_adjust=bw_ad, color='g', linewidth=line_wid)
        sns.kdeplot(feature_list[1], bw_adjust=bw_ad, color='b', linewidth=line_wid)
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.ylabel(" ")
        plt.title(method_name_list[i])

    plt.savefig(f"KDE/ACDC/{f_dim}/labeled_{FLAGS.labelnum}/class_{specific_c}/kde_test_mean{pic_num}_{FLAGS.labelnum}_{specific_c}.png")
    print(f"Save to: KDE/ACDC/{f_dim}/labeled_{FLAGS.labelnum}/class_{specific_c}/kde_test_mean{pic_num}_{FLAGS.labelnum}_{specific_c}.png")
    plt.clf()
    

def Inference(FLAGS):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    BCP_Net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes, mode="test")

    BCP_Net.load_state_dict(torch.load(BCP_model_path))
    print("init models' weight successfully")
    BCP_Net.eval()
    
    def worker_init_fn(worker_id):
        random.seed(1337 + worker_id)

    db_train = BaseDataSets(base_dir=FLAGS.root_path,
                            split='train',
                            num=None,
                            transform=transforms.Compose([RandomGenerator(FLAGS.patch_size)]))
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(FLAGS.root_path, FLAGS.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idx = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idx, unlabeled_idxs, FLAGS.batch_size, FLAGS.batch_size-FLAGS.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    picture_number = 0
    for epoch_num in range(3):
        for _, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            label_batch = label_batch.detach().cpu().numpy()

            # What is "BCP_feature" below?
            # In our experiments, BCP_feature is x8_up of the Decoder,
            # here are simple pseudo code for example:

            #   class Decoder:
            #
            #       ...
            #
            #       def forward(self, x):
            #       x8_up = x8_up + x1
            #       x9 = self.block_nine(x8_up)
            #       x9 = F.dropout3d(x9, p=0.5, training=True)
            #       if self.has_dropout:
            #           x9 = self.dropout(x9)
            #           ...
            #           out_seg = self.out_conv(x9)
            #           return out_seg, x8_up

            #   class VNet:
            #
            #      ...
            #
            #       def forward(self, x):
            #           f = self.encoder(x)
            #           out_seg, BCP_feature = self.decoder(f)
            #           return out_seg, BCP_feature

            # Obviously there are many different choices to get the mdoel's feature.

            pred, BCP_feature = BCP_Net(volume_batch)

            B_pred = get_ACDC_masks(pred)

            f_dim, x_, y_ = BCP_feature.shape[1], BCP_feature.shape[2], BCP_feature.shape[3]
            
            BCP_feature = BCP_feature.permute(0, 2, 3, 1).contiguous()
            BCP_feature = BCP_feature.view(-1, f_dim) # 1000, 16

            resized_label = np.zeros((FLAGS.batch_size, x_, y_))
            for i in range(FLAGS.batch_size):
                resized_label[i,] = cv2.resize(label_batch[i,].squeeze(), (x_, y_))
            #resized_label = cv2.resize(label_batch, (x_, y_))
            label_batch = torch.from_numpy(resized_label).cuda()
            label = label_batch.view(-1, 1) # a (3, 1) b[a, :]
            BCP_pred = B_pred.view(-1, 1)

            BCP_feature = BCP_feature.detach().cpu().numpy()
            BCP_pred = BCP_pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            # try:
            #for spi_c in range(1, 4):
            spi_c = 2
            plot_kde(BCP_feature, BCP_pred, label, spi_c, f_dim, picture_number)
            picture_number += 1
            # except Exception as e:
            #     print(e)

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    Inference(FLAGS)
