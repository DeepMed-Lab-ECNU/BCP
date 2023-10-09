import numpy as np
import torch
import h5py

from torch import import_ir_module, nn as nn, optim as optim
from torch.utils.data import DataLoader
from Vnet import VNet
from torch.utils.data import Dataset
from torchvision.transforms import Compose


def create_Vnet(ema=False):
    net = VNet()
    net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def _get_transform(self, x):
        if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - x.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - x.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - x.shape[2]) // 2 + 1, 0)
            x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = x.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        def do_transform(image):
            if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]:
                try:
                    image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                except Exception as e:
                    print(e)
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return image

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]
    

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def _get_transform(self, label):
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 1, 0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = label.shape
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        def do_transform(x):
            if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
                x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            x = x[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return x

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]
    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample[0]
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample = [image] + [*sample[1:]]
        return [torch.from_numpy(s.astype(np.float32)) for s in sample]
    

def get_dataset_path(dataset='pancreas', labelp='10percent'):
    files = ['train_lab.txt', 'train_unlab.txt', 'test.txt']
    return ['/'.join(['/home/ubuntu/byh/code/CoraNet-master/data_lists', dataset, labelp, f]) for f in files]



class Pancreas(Dataset):
    """ Pancreas Dataset """
    def __init__(self, base_dir, name, split, no_crop=False, labelp=10, reverse=False, TTA=False):
        self._base_dir = base_dir
        self.split = split
        self.reverse=reverse
        self.labelp = '10percent'
        if labelp == 20:
            self.labelp = '20percent'

        tr_transform = Compose([
            # RandomRotFlip(),
            RandomCrop((96, 96, 96)),
            # RandomNoise(),
            ToTensor()
        ])
        if no_crop:
            test_transform = Compose([
                # CenterCrop((160, 160, 128)),
                CenterCrop((96, 96, 96)),
                ToTensor()
            ])
        else:
            test_transform = Compose([
                CenterCrop((96, 96, 96)),
                ToTensor()
            ])

        data_list_paths = get_dataset_path(name, self.labelp)

        if split == 'train_lab':
            data_path = data_list_paths[0]
            self.transform = tr_transform
        elif split == 'train_unlab':
            data_path = data_list_paths[1]
            self.transform = test_transform  # tr_transform
        else:
            data_path = data_list_paths[2]
            self.transform = test_transform

        with open(data_path, 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [self._base_dir + "/{}".format(item.strip()) for item in self.image_list]
        print("Split : {}, total {} samples".format(split, len(self.image_list)))

    def __len__(self):
        if self.split == 'train_lab' and self.labelp == '20percent':
            return len(self.image_list) * 5
        elif self.split == 'train_lab' and self.labelp == '10percent':
            return len(self.image_list) * 10
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx % len(self.image_list)]
        if self.reverse:
            image_path = self.image_list[len(self.image_list) - idx % len(self.image_list) - 1]
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)
        samples = image, label
        if self.transform:
            tr_samples = self.transform(samples)
        image_, label_ = tr_samples
        return image_.float(), label_.long()


def get_ema_model_and_dataloader(data_root, split_name, batch_size, lr, labelp=10):
    print("Initialize ema cutmix: network, optimizer and datasets...")
    """Net & optimizer"""
    net = create_Vnet()
    ema_net = create_Vnet(ema=True).cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    trainset_lab_a = Pancreas(data_root, split_name, split='train_lab', labelp=labelp)
    lab_loader_a = DataLoader(trainset_lab_a, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_lab_b = Pancreas(data_root, split_name, split='train_lab', labelp=labelp, reverse=True)
    lab_loader_b = DataLoader(trainset_lab_b, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    
    trainset_unlab_a = Pancreas(data_root, split_name, split='train_unlab', labelp=labelp)
    unlab_loader_a = DataLoader(trainset_unlab_a, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    trainset_unlab_b = Pancreas(data_root, split_name, split='train_unlab', labelp=labelp, reverse=True)
    unlab_loader_b = DataLoader(trainset_unlab_b, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    
    testset = Pancreas(data_root, split_name, split='test')
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    return net, ema_net, optimizer, lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, test_loader