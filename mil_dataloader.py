# -*- coding: utf-8 -*-
"""
@author: ZHANG Min, Wuhan University
@email: 007zhangmin@whu.edu.cn
"""

import numpy as np
import torch
import torch.utils.data as data_utils
import os
import util


class CDBags(data_utils.Dataset):
    def __init__(self, data_dir='', seed=1, train=True):
        self.train = train
        self.data_dir = data_dir
        self.random = np.random.RandomState(seed)
        self.bags = self._load_bags()
        if self.train:
            self.random.shuffle(self.bags)

    def _load_bags(self):
        if self.train:
            txt_path = os.path.join(self.data_dir, 'Train.txt')
        else:
            txt_path = os.path.join(self.data_dir, 'Test.txt')

        imgs = np.loadtxt(txt_path, dtype=str)
        file_list = list(imgs)
        return file_list

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        file_name = self.bags[index]
        # Examples of file name : "N_xxx_" ,i.e.,
        # T1 and T2 file names are N_xxx_T1.tif, N_xxx_T2.tif
        # "N" denotes "Negative bag" and "P" denotes "Positive bag"

        bag_path = os.path.join(self.data_dir, file_name)
        if file_name[0] == 'N':
            # Negative bag
            label = torch.LongTensor([0])
        else:
            # Positive bag
            label = torch.LongTensor([1])

        t1_path = bag_path + 'T1.tif'
        t2_path = bag_path + 'T2.tif'
        t1 = util.read_image(t1_path)
        t2 = util.read_image(t2_path)
        t2 = util.hist_match(t2, t1)

        data1 = t1.transpose((2, 0, 1))
        data2 = t2.transpose((2, 0, 1))

        return data1, data2, label, bag_path


if __name__ == "__main__":
    data_dir = r'D:\5.Download\Dataset\Landslide\ls_cdminet'

    train_loader = data_utils.DataLoader(
        CDBags(data_dir=data_dir,
               seed=1, train=True),
        batch_size=1)

    test_loader = data_utils.DataLoader(
        CDBags(data_dir=data_dir,
               seed=1,
               train=False),
        batch_size=1)

    train_bags = 0
    for batch_idx, (t1, t2, label, path) in enumerate(train_loader):
        train_bags += label[0].numpy()[0]
    print('Number of positive bags in training set: {}/{}\n'.format(train_bags, len(train_loader)))

    test_bags = 0
    for batch_idx, (t1, t2, label, path) in enumerate(test_loader):
        test_bags += label[0].numpy()[0]
    print('Number of positive bags in test set: {}/{}\n'.format(test_bags, len(test_loader)))
