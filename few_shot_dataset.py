import numpy as np
import random
from scipy.io import loadmat, savemat
import os
import argparse

parser = argparse.ArgumentParser(description='build few-shot dataset')
parser.add_argument('--dataset', type=str, default='nuswide', help='the name of dataset')
parser.add_argument('--categories', type=int, default=21, help='the number of categories contained in the dataset')
parser.add_argument('--unseen_num', type=int, default=4, help='the number of unseen class, 20%')
parser.add_argument('--k', type=int, default=1, help='k-shot')
parser.add_argument('--more', type=int, default=1, help='whether sample few-shot data more times than seen data, 1 means not, otherwise, means the number of times')
parser.add_argument('--seed', type=int, default=7)
args = parser.parse_args()
print(args)
random.seed(args.seed)

categories = args.categories
unseen_num = args.unseen_num # the number of unseen class, 20%
k_shot = args.k
sample_times = args.more # the times of sampling unseen data  #******
# labels_idx = [1, 4, 7, 8]
# labels_idx = [0, 3]  # wiki
labels_idx = random.sample([i for i in range(categories)], unseen_num)
# labels_idx = [10, 4, 12, 20, 1]  # mir
print('unseen labels:', labels_idx)
unseen_labels = np.zeros((unseen_num, categories))
for i in range(unseen_labels.shape[0]):
    unseen_labels[i, labels_idx[i]] = 1

#******* split into new dataset for fow shot*********
data_path = f'./data/{args.dataset}/'
if args.more != 1:
    root = f'./data/{args.dataset}/few_shot/{k_shot}shot_more/'    # every unseen samples 10 times more
else:
    root = f'./data/{args.dataset}/few_shot/{k_shot}shot/'   # ******
if not os.path.exists(root):
    os.makedirs(root)

# # ************train set**************     
# train_labels = loadmat(data_path + '/train_img_lab.mat')['train_img_lab']
# train_imgs = loadmat(data_path + '/train_img.mat')['train_img']
# train_txt = loadmat(data_path + '/train_txt.mat')['train_txt']
train_imgs = loadmat(data_path + 'ITL.mat')['tr_i']
train_conv3 = loadmat(data_path + 'ITL.mat')['tr_conv3']
train_conv4 = loadmat(data_path + 'ITL.mat')['tr_conv4']
# train_conv5 = loadmat(data_path + 'NL2ITL.mat')['tr_conv5']
train_txt = loadmat(data_path + 'ITL.mat')['tr_t']
train_labels = loadmat(data_path + 'ITL.mat')['tr_l']
# split original train set into seen and unseen, random sample k-shot， k samples per class
train_unseen = np.where((train_labels * unseen_labels[0]).sum(1) > 0)[0]  # unseen index
samples = np.array(random.sample(list(train_unseen), k_shot))
train_k = samples.repeat(sample_times)  # sample unseen index
for i in range(1, unseen_num):
    unseen_idxs = np.where((train_labels * unseen_labels[i]).sum(1) > 0)[0]
    samples = np.array(random.sample(list(unseen_idxs), k_shot))
    train_k = np.concatenate((train_k, samples.repeat(sample_times)))
    train_unseen = np.concatenate((train_unseen, unseen_idxs))
train_unseen = np.unique(train_unseen)

print('len train unseen:', len(train_unseen))
# train seen
train_size = train_labels.shape[0]
train_set = np.zeros(train_size) + 1
train_set[train_unseen] = 0
train_seen = np.where(train_set == 1)[0]
# new
train_index = np.concatenate((train_seen, train_k))
tr_l = train_labels[train_index]
train_imgs_ = train_imgs[train_index]
train_txt_ = train_txt[train_index]
train_conv3_ = train_conv3[train_index]
train_conv4_ = train_conv4[train_index]
# train_conv5_ = train_conv5[train_index]
train_size_ = train_size - len(train_unseen) + k_shot * unseen_num * sample_times
print('train_size:', train_size_, 'i:', train_imgs_.shape,
      'conv3:', train_conv3_.shape,  'conv4:', train_conv4_.shape,
      't:', train_txt_.shape, 'l:', tr_l.shape)
assert tr_l.shape[0] == train_txt_.shape[0] == train_imgs_.shape[0] == train_size_
savemat(root + 'ITL.mat', {'tr_conv3':train_conv3_, 'tr_conv4':train_conv4_,
                              'tr_i':train_imgs_, 'tr_t':train_txt_, 'tr_l':tr_l})
# savemat(root + '/train_img_lab.mat', {'train_img_lab':tr_l})
# savemat(root + '/train_img.mat', {'train_img':train_imgs_})
# savemat(root + '/train_txt.mat', {'train_txt':train_txt_})

# # # ************test set**************
# test_labels = loadmat(data_path + '/test_img_lab.mat')['test_img_lab']
# test_imgs = loadmat(data_path + '/test_img.mat')['test_img']
# test_txt = loadmat(data_path + '/test_txt.mat')['test_txt']
# test_imgs = loadmat(data_path + 'ITL.mat')['te_i']
# test_conv3 = loadmat(data_path + 'ITL.mat')['te_conv3']
# test_conv4 = loadmat(data_path + 'ITL.mat')['te_conv4']
# test_txt = loadmat(data_path + 'ITL.mat')['te_t']
# test_labels = loadmat(data_path + 'ITL.mat')['te_l']
# test_labels = np.argmax(test_labels, axis=1)
# # split original test set into seen and unseen
# test_unseen = np.where(test_labels[:] == unseen_labels[0])[0]
# for i in range(1, unseen_num):
#     test_unseen = np.concatenate((test_unseen, np.where(test_labels[:] == unseen_labels[i])[0]))
# test_size = test_labels.shape[0]
# test_set = np.zeros(test_size) + 1
# test_set[test_unseen] = 0
# test_seen = np.where(test_set == 1)[0]
# # new
# test_index = np.concatenate((test_seen, test_unseen), axis=0)
# test_labels_ = test_labels[test_index]
# test_imgs_ = test_imgs[test_index]
# test_txt_ = test_txt[test_index]
# # transform labels to one-hot
# te_l = np.zeros((test_labels_.shape[0], categories))
# for i in range(test_labels_.shape[0]):
#     te_l[i,test_labels_[i]] = 1.
# assert te_l.shape[0] == test_txt_.shape[0] == test_imgs_.shape[0] == test_size == test_conv3_.shape[0] == test_conv4_.shape[0]
# # save
