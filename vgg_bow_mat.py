import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as torchdata
import torch.backends.cudnn as cudnn
import os
from scipy.io import savemat, loadmat
import re
import torch
from PIL import Image
from scipy.io import loadmat, savemat
import torchvision.transforms as transforms
import torchvision.models as models
import random

from data_load import *


def GAP(x):  # global average pooling
    N = x.size(-1) * x.size(-2)
    gap = x.sum(-1).sum(-1) / N
    return gap


class ImgModule(nn.Module):
    def __init__(self):
        super(ImgModule, self).__init__()
        self.model = models.vgg19(pretrained=True)
        # self.model = torch.load('/home/disk1/wangshy/vgg19.pth')
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-2])
        self.conv3 = self.model.features[:19]
        self.conv4 = self.model.features[:28]
        # self.conv5 = self.model.features
        # print(self.model)

    def forward(self, x):
        # print('conv3:', self.conv3(x).size())
        # print('conv4:', self.conv4(x).size())
        # print('conv5:', self.conv5(x).size())

        # conv3 = GAP(self.conv3(x))
        # conv4 = GAP(self.conv4(x))

        conv3 = self.conv3(x)   # (256, 28, 28)
        conv4 = self.conv4(x)   # (512, 14, 14)
        # conv5 = self.conv5(x)   # (512, 14, 14)

        features = self.model(x)
        # return conv3, conv4, conv5, features
        return conv3, conv4, features

# Wikipedia
class WikiDataset(torchdata.Dataset):
    def __init__(self, dir):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # [0,255] --> [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        f = open(dir)
        I = []
        L = []
        for line in f.readlines():
            cur = line.strip().split('\t')  # 3168
            img_name = cur[1]
            I.append(img_name)
            lbl = cur[2]
            label = torch.tensor([0] * 10)
            label[int(lbl) - 1] = 1
            L.append(label)
        self.I = I
        self.L = L

    def __getitem__(self, i):
        img = self.I[i]
        img_dir = '/home/disk1/wangshy/data/wikipedia_dataset/images/' + img + '.jpg'
        image_data = Image.open(img_dir, 'r').convert('RGB')
        image = self.transform(image_data)
        label = self.L[i]
        return image, label

    def __len__(self):
        return len(self.I)


# iapr_tc12
class IaprDataset(torchdata.Dataset):
    # load image, text and label
    def __init__(self, txt_file):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # [0,255] --> [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        imgs_path = []
        labels = []
        txt = []
        IL_f = open(txt_file, 'r')
        for line in IL_f:
            if line == '':
                break
            line = line.strip().split()
            imgs_path.append(line[0])
            txt.append(line[1:2913])
            labels.append(line[2913:])
        self.I = imgs_path
        self.T = txt
        self.L = labels

    def __getitem__(self, index):
        # load img
        img_path = '/home/disk1/wangshy/data/iapr-tc12/JPEGImages/' + self.I[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # load label
        label = torch.from_numpy(np.array(self.L[index], dtype='float32'))
        # load text
        text = torch.from_numpy(np.array(self.T[index], dtype='float32'))
        return img, text, label

    def __len__(self):
        return len(self.I)


# mir-flickr 25k BOW
class MirDataset(torchdata.Dataset):
    def __init__(self, IL_file, IT_file):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # [0,255] --> [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        imgs = []
        labels = []
        text = []
        IL_f = open(IL_file, 'r')
        IT_f = open(IT_file, 'r')
        for line in IL_f:
            if line == '':
                break
            line = line.strip().split()
            imgs.append(line[0])
            labels.append(line[1:])
        for line in IT_f:
            if line == '':
                break
            line = line.strip().split()  # bag of word
            text.append(line[1:])

        self.I = imgs
        self.L = labels
        self.T = text

    def __getitem__(self, index):
        img_path = '/home/disk1/wangshy/data/mir-25k/JPEGImages/' + self.I[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(np.array(self.L[index], dtype='float32'))
        text = torch.from_numpy(np.array(self.T[index], dtype='float32'))

        return img, text, label

    def __len__(self):
        return len(self.I)


# NUS_WIDE
class NusDataset(torchdata.Dataset):
    def __init__(self, IL_file):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # [0,255] --> [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        imgs = []
        labels = []
        txt = []
        IL_f = open(IL_file, 'r')
        for line in IL_f:
            if line == '':
                break
            line = line.strip().split()
            imgs.append(line[0])
            labels.append(line[1001:])
            txt.append(line[1:1001])
        self.I = imgs
        self.L = labels
        self.T = txt

    def __getitem__(self, index):
        img_path = '/home/disk1/wangshy/data/nuswide/JPEGImages/' + self.I[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:    
            img = self.transform(img)
        label = torch.from_numpy(np.array(self.L[index], dtype='float32'))
        txt = torch.from_numpy(np.array(self.T[index], dtype='float32'))
        return img, txt, label

    def __len__(self):
        return len(self.I)



if __name__ == '__main__':
    batch_size = 8   
    gpu = 'cuda:3'
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    # ## mir-25k
    # def load_data(IL_file, IT_file, batch_size, shuffle):
    #     data_ = MirDataset(IL_file, IT_file)
    #     loader_ = torchdata.DataLoader(
    #         dataset=data_,
    #         batch_size=batch_size,
    #         shuffle=shuffle
    #     )
    #     return loader_
    # tr_ILF = '/home/disk1/wangshy/data/mir-25k/docs/train_images.txt'
    # tr_ITF = '/home/disk1/wangshy/data/mir-25k/docs/train_bows.txt'
    # te_ILF = '/home/disk1/wangshy/data/mir-25k/docs/test_images.txt'
    # te_ITF = '/home/disk1/wangshy/data/mir-25k/docs/test_bows.txt'
    # db_ILF = '/home/disk1/wangshy/data/mir-25k/docs/retrieval_images.txt'
    # db_ITF = '/home/disk1/wangshy/data/mir-25k/docs/retrieval_bows.txt'
    # mat_file = '/home/disk1/wangshy/data/mir-25k/ITL.mat'
    # tr_loader = load_data(tr_ILF, tr_ITF, batch_size=batch_size, shuffle=False)
    # te_loader = load_data(te_ILF, te_ITF, batch_size=batch_size, shuffle=False)
    # db_loader = load_data(db_ILF, db_ITF, batch_size=batch_size, shuffle=False)

    # Nuswide
    # def load_data(IL_file, batch_size, shuffle):
    #     data_ = NusDataset(IL_file)
    #     loader_ = torchdata.DataLoader(
    #         dataset=data_,
    #         batch_size=batch_size,
    #         shuffle=shuffle
    #     )
    #     return loader_
    # tr_ILF = '/home/disk1/wangshy/data/nuswide/new_train'
    # te_ILF = '/home/disk1/wangshy/data/nuswide/new_test'
    # db_ILF = '/home/disk1/wangshy/data/nuswide/new_retrieval'
    # mat_file = '/home/disk1/wangshy/data/nuswide/ITL.mat'
    #
    # tr_loader = load_data(tr_ILF, batch_size=batch_size, shuffle=False)
    # te_loader = load_data(te_ILF, batch_size=batch_size, shuffle=False)
    # db_loader = load_data(db_ILF, batch_size=batch_size, shuffle=False)
    #
    # model = ImgModule()
    # # torch.nn.DataParallel(model, device_ids=[1, 2, 3])
    # model = model.to(device)
    # model.eval()
    #
    # # train set
    # tr_conv3 = []
    # tr_conv4 = []
    # tr_i = []
    # tr_t = []
    # tr_l = []
    # for i, (imgs, text, labels) in enumerate(tr_loader):
    #     conv3, conv4, embed = model.forward(imgs.to(device))
    #     print(i, conv3.size(), conv4.size(), embed.size(), text.size(), labels.size())
    #     tr_i.append(embed.data.cpu())
    #     tr_conv3.append(conv3.data.cpu())
    #     tr_conv4.append(conv4.data.cpu())
    #     tr_l.append(labels.data.cpu())
    #     tr_t.append(text.data.cpu())
    # tr_i = np.concatenate(tr_i)
    # tr_conv3 = np.concatenate(tr_conv3)
    # tr_conv4 = np.concatenate(tr_conv4)
    # tr_l = np.concatenate(tr_l)
    # tr_t = np.concatenate(tr_t)
    # print('tr:',tr_i.shape, tr_conv3.shape, tr_conv4.shape, tr_t.shape, tr_l.shape)
    #
    # # test set
    # te_conv3 = []
    # te_conv4 = []
    # te_i = []
    # te_t = []
    # te_l = []
    # for i, (imgs, text, labels) in enumerate(te_loader):
    #     conv3, conv4, embed = model.forward(imgs.to(device))
    #     te_i.append(embed.data.cpu())
    #     te_conv3.append(conv3.data.cpu())
    #     te_conv4.append(conv4.data.cpu())
    #     te_l.append(labels.data.cpu())
    #     te_t.append(text.data.cpu())
    # te_i = np.concatenate(te_i)
    # te_conv3 = np.concatenate(te_conv3)
    # te_conv4 = np.concatenate(te_conv4)
    # te_l = np.concatenate(te_l)
    # te_t = np.concatenate(te_t)
    # print('te:',te_i.shape, te_conv3.shape, te_conv4.shape, te_t.shape, te_l.shape)
    #
    # # retrieval set
    # db_conv3 = []
    # db_conv4 = []
    # db_i = []
    # db_t = []
    # db_l = []
    # for i, (imgs, text, labels) in enumerate(db_loader):
    #     conv3, conv4, embed = model.forward(imgs.to(device))
    #     db_i.append(embed.data.cpu())
    #     db_conv3.append(conv3.data.cpu())
    #     db_conv4.append(conv4.data.cpu())
    #     db_l.append(labels.data.cpu())
    #     db_t.append(text.data.cpu())
    # db_i = np.concatenate(db_i)
    # db_conv3 = np.concatenate(db_conv3)
    # db_conv4 = np.concatenate(db_conv4)
    # db_l = np.concatenate(db_l)
    # db_t = np.concatenate(db_t)
    # print('db:',db_i.shape, db_conv3.shape, db_conv4.shape, db_t.shape, db_l.shape)
    #
    # savemat(mat_file, {'tr_conv3':tr_conv3, 'tr_conv4':tr_conv4, 'tr_i':tr_i, 'tr_t':tr_t, 'tr_l':tr_l,
    #                    'te_conv3':te_conv3, 'te_conv4':te_conv4, 'te_i':te_i, 'te_t':te_t, 'te_l':te_l,
    #                    'db_conv3':db_conv3, 'db_conv4':db_conv4, 'db_i':db_i, 'db_t':db_t, 'db_l':db_l})


    ## wikipedia
    def load_data(IL_file, batch_size, shuffle):
        data_ = WikiDataset(IL_file)
        loader_ = torchdata.DataLoader(
            dataset=data_,
            batch_size=batch_size,
            shuffle=shuffle
        )
        return loader_
    tr_ILF = '/home/disk1/wangshy/data/wikipedia_dataset/trainset_txt_img_cat.list'
    te_ILF = '/home/disk1/wangshy/data/wikipedia_dataset/testset_txt_img_cat.list'
    tr_TF = '/home/disk1/wangshy/data/wikipedia_dataset/train_txt.mat'
    te_TF = '/home/disk1/wangshy/data/wikipedia_dataset/test_txt.mat'
    mat_file = '/home/disk1/wangshy/data/wikipedia_dataset/NL2ITL.mat'

    tr_loader = load_data(tr_ILF, batch_size=batch_size, shuffle=False)
    te_loader = load_data(te_ILF, batch_size=batch_size, shuffle=False)

    model = ImgModule()
    # torch.nn.DataParallel(model, device_ids=[1, 2, 3])
    model = model.to(device)
    model.eval()

    # train set
    tr_conv3 = []
    tr_conv4 = []
    tr_conv5 = []
    tr_i = []
    tr_l = []
    for i, (imgs, labels) in enumerate(tr_loader):
        # conv3, conv4, conv5, embed = model.forward(imgs.to(device))
        conv3, conv4, embed = model.forward(imgs.to(device))
        print(i, conv3.size(), conv4.size(), embed.size(), labels.size())
        tr_i.append(embed.data.cpu())
        tr_conv3.append(conv3.data.cpu())
        tr_conv4.append(conv4.data.cpu())
        # tr_conv5.append(conv5.data.cpu())
        tr_l.append(labels.data.cpu())
    tr_i = np.concatenate(tr_i)
    tr_conv3 = np.concatenate(tr_conv3)
    tr_conv4 = np.concatenate(tr_conv4)
    # tr_conv5 = np.concatenate(tr_conv5)
    tr_l = np.concatenate(tr_l)
    tr_t = loadmat('/home/disk1/wangshy/data/wikipedia_dataset/w2v.mat')['tr_W']
    print('tr:',tr_i.shape, tr_conv3.shape, tr_conv4.shape, tr_t.shape, tr_l.shape)

    # test set
    te_conv3 = []
    te_conv4 = []
    te_conv5 = []
    te_i = []
    te_l = []
    for i, (imgs, labels) in enumerate(te_loader):
        # conv3, conv4, conv5, embed = model.forward(imgs.to(device))
        conv3, conv4, embed = model.forward(imgs.to(device))
        te_i.append(embed.data.cpu())
        te_conv3.append(conv3.data.cpu())
        te_conv4.append(conv4.data.cpu())
        # te_conv5.append(conv5.data.cpu())
        te_l.append(labels.data.cpu())
    te_i = np.concatenate(te_i)
    te_conv3 = np.concatenate(te_conv3)
    te_conv4 = np.concatenate(te_conv4)
    # te_conv5 = np.concatenate(te_conv5)
    te_l = np.concatenate(te_l)
    idxs = sorted(random.sample([i for i in range(693)], 462))
    te_t = loadmat('/home/disk1/wangshy/data/wikipedia_dataset/w2v.mat')['te_W']
    te_i_ = te_i[idxs]
    te_t_ = te_t[idxs]
    te_l_ = te_l[idxs]
    te_conv3_ = te_conv3[idxs]
    te_conv4_ = te_conv4[idxs]
    # te_conv5_ = te_conv5[idxs]
    print('te:',te_i_.shape, te_conv3_.shape, te_conv4_.shape, te_t_.shape, te_l_.shape)

    # retrieval set
    mask = np.zeros(te_t.shape[0], dtype=int)
    mask[idxs] = 1
    resid_idxs = np.where(mask == 0)[0]
    db_i = np.concatenate((tr_i, te_i[resid_idxs]), axis=0)   
    db_t = np.concatenate((tr_t, te_t[resid_idxs]), axis=0)
    db_conv3 = np.concatenate((tr_conv3, te_conv3[resid_idxs]), axis=0)
    db_conv4 = np.concatenate((tr_conv4, te_conv4[resid_idxs]), axis=0)       
    # db_conv5 = np.concatenate((tr_conv5, te_conv5[resid_idxs]), axis=0)
    db_l = np.concatenate((tr_l, te_l[resid_idxs]), axis=0)
    print('db:',db_i.shape, db_conv3.shape, db_conv4.shape, db_t.shape, db_l.shape)

    # savemat(mat_file, {'tr_conv3':tr_conv3, 'tr_conv4':tr_conv4, 'tr_conv5':tr_conv5, 'tr_i':tr_i, 'tr_t':tr_t, 'tr_l':tr_l,
    #                    'te_conv3':te_conv3_, 'te_conv4':te_conv4_, 'te_conv5':te_conv5_, 'te_i':te_i_, 'te_t':te_t_, 'te_l':te_l_,
    #                    'db_conv3': db_conv3, 'db_conv4': db_conv4, 'db_conv5': db_conv5, 'db_i': db_i, 'db_t': db_t, 'db_l': db_l})
    savemat(mat_file,
            {'tr_conv3': tr_conv3, 'tr_conv4': tr_conv4, 'tr_i': tr_i, 'tr_t': tr_t, 'tr_l': tr_l,
             'te_conv3': te_conv3_, 'te_conv4': te_conv4_, 'te_i': te_i_, 'te_t': te_t_,
             'te_l': te_l_,
             'db_conv3': db_conv3, 'db_conv4': db_conv4, 'db_i': db_i, 'db_t': db_t,
             'db_l': db_l})     