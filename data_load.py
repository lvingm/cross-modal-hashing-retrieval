import torch
import torch.utils.data as torchdata
import numpy as np
from scipy.io import loadmat

def ind2vec(ind, N=None):
    # transform label to vectors
    # ind = np.asarray(ind)
    # ind = np.expand_dims(ind, 1)
    if N is None:
        N = ind.max() + 1
    return np.arange(N) == np.repeat(ind, N, axis=1)

class MatDataset(torchdata.Dataset):
    def __init__(self, data_file, mode):
        if mode == 'train':
            imgs = loadmat(data_file + 'train_img.mat')['train_img']
            texts = loadmat(data_file + 'train_txt.mat')['train_txt']
            labels = loadmat(data_file + 'train_img_lab.mat')['train_img_lab']
        elif mode == 'test':
            imgs = loadmat(data_file + 'test_img.mat')['test_img']
            texts = loadmat(data_file + 'test_txt.mat')['test_txt']
            labels = loadmat(data_file + 'test_img_lab.mat')['test_img_lab']
        else:
            imgs = loadmat(data_file + 'db_img.mat')['db_img']
            texts = loadmat(data_file + 'db_txt.mat')['db_txt']
            labels = loadmat(data_file + 'db_img_lab.mat')['db_img_lab']
        # labels = ind2vec(labels).astype('float32')
        self.imgs = imgs
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.imgs[index].astype('float32')
        text = self.texts[index].astype('float32')
        label = self.labels[index].astype('float32')    
        img = torch.from_numpy(img)
        text = torch.from_numpy(text)
        label = torch.from_numpy(label)
        return img, text, label

    def __len__(self):
        assert self.texts.shape[0] == self.labels.shape[0] == self.imgs.shape[0]
        return self.texts.shape[0]


class MatDataset1(torchdata.Dataset):
    def __init__(self, data_file, mode):
        if mode == 'train':
            pre = 'tr'
        elif mode == 'test':
            pre = 'te'
        else:
            pre = 'db'
        imgs = loadmat(data_file)[pre + '_i']
        conv3 = loadmat(data_file)[pre + '_conv3']
        conv4 = loadmat(data_file)[pre + '_conv4']
        texts = loadmat(data_file)[pre + '_t']
        labels = loadmat(data_file)[pre + '_l']
        self.imgs = imgs
        self.texts = texts
        self.labels = labels
        self.conv3 = conv3
        self.conv4 = conv4

    def __getitem__(self, index):
        img = self.imgs[index].astype('float32')
        text = self.texts[index].astype('float32')
        label = self.labels[index].astype('float32')
        conv3 = self.conv3[index].astype('float32')
        conv4 = self.conv4[index].astype('float32')
        img = torch.from_numpy(img)
        text = torch.from_numpy(text)
        label = torch.from_numpy(label)
        conv3 = torch.from_numpy(conv3)
        conv4 = torch.from_numpy(conv4)
        return img, conv3, conv4, text, label

    def __len__(self):
        assert self.texts.shape[0] == self.labels.shape[0] == self.imgs.shape[0]
        return self.texts.shape[0]


class MatDataset2(torchdata.Dataset):
    def __init__(self, data_file, mode):
        if mode == 'train':   
            pre = 'tr'
        elif mode == 'test':
            pre = 'te'
        else:
            pre = 'db'
        imgs = loadmat(data_file)[pre + '_i']
        texts = loadmat(data_file)[pre + '_t']
        labels = loadmat(data_file)[pre + '_l']
        self.imgs = imgs
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.imgs[index].astype('float32')
        text = self.texts[index].astype('float32')
        label = self.labels[index].astype('float32')
        img = torch.from_numpy(img)
        text = torch.from_numpy(text)
        label = torch.from_numpy(label)
        return img, text, label

    def __len__(self):
        assert self.texts.shape[0] == self.labels.shape[0] == self.imgs.shape[0]
        return self.texts.shape[0]







