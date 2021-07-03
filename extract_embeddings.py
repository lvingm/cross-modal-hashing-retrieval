import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertModel, BertConfig
import os
import re
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from scipy.io import savemat

from data_load import *


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
os.environ['CUDA_LAUNCH_BLOCKING'] = "2"

stopwords = []
stop = open('./data/wikipedia_dataset/stop_list.txt', 'r')
for word in stop.readlines():
    stopwords.append(word.strip())
# stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
def clean_str(string):
    ## process sentences
    string = re.sub(r"[0-9,!?'`£*^./:;@#$%&+]", " ", string)
    string = re.sub(r"[()[]", "", string)
    sent = string.strip().lower().split()
    # filter stop words
    filter_sent = [w for w in sent if w not in stopwords and w.isalpha()]
    return filter_sent


def GAP(x):  # global average pooling
    N = x.size(-1) * x.size(-2)
    gap = x.sum(-1).sum(-1) / N
    return gap      

class TxtModule_BERT(nn.Module):
    def __init__(self, device):
        super(TxtModule_BERT, self).__init__()
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.config = BertConfig.from_pretrained('./bert-base-uncased-config.json')
        self.transformer = BertModel.from_pretrained('./bert-base-uncased-pytorch_model.bin', config=self.config)
        self.device = device

    def forward(self, tokens_idxs, segment_idxs):
        # forward
        transformer_output = self.transformer(input_ids=tokens_idxs, token_type_ids=segment_idxs)[0]
        output = transformer_output[:,0,:]# .squeeze(1)   # 768-D
        return output


class ImgModule(nn.Module):
    def __init__(self):
        super(ImgModule, self).__init__()
        self.model = models.vgg19(pretrained=True)
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-2])
        self.conv3 = self.model.features[:19]
        self.conv4 = self.model.features[:28]
        # print(self.model)
    def forward(self, x):
        conv3 = GAP(self.conv3(x))
        conv4 = GAP(self.conv4(x))
        features = self.model(x)
        return conv3, conv4, features

class WikiImg(torchdata.Dataset):
    def __init__(self, dir):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # [0,255] --> [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        f = open(dir)
        I = []
        for line in f.readlines():
            cur = line.strip().split('\t')  # 3168
            img_name = cur[1]
            I.append(img_name)
        self.I = I

    def __getitem__(self, i):
        img = self.I[i]
        img_dir = './data/wikipedia_dataset/images/' + img + '.jpg'
        try:
            image_data = Image.open(img_dir, 'r').convert('RGB')
        except:
            print(img_dir)
        image = self.transform(image_data)
        return image

    def __len__(self):
        return len(self.I)

    
class DataLoading_wiki(torchdata.Dataset):
    def __init__(self, dir):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # [0,255] --> [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        f = open(dir)
        I = []
        T = []
        L = []
        for line in f.readlines():
            cur = line.strip().split('\t')  # 3168
            txt_name = cur[0]
            lbl = cur[2]
            label = torch.tensor([0] * 10)
            label[int(lbl) - 1] = 1
            img_name = cur[1]
            I.append(img_name)
            T.append(txt_name)
            L.append(label)
        self.I = I
        self.T = T
        self.L = L

    def __getitem__(self, i):
        img = self.I[i]
        img_dir = './data/wikipedia_dataset/images/' + img + '.jpg'
        try:
            image_data = Image.open(img_dir, 'r').convert('RGB')
        except:
            print(img_dir)
        image = self.transform(image_data)
        txt_dir = self.T[i]
        f = open('./data/wikipedia_dataset/process_texts2/' + txt_dir + '.xml', 'r')
        text = f.read().strip()
        text = '[CLS] ' + text + ' [SEP]'
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        padding = [0] * (456 - len(indexed_tokens))
        indexed_tokens += padding
        tokens_tensor = torch.tensor([indexed_tokens])[0]
        segments_tensors = torch.tensor([0] * (456))
        f.close()
        label = self.L[i]
        return image, tokens_tensor, segments_tensors, label

    def __len__(self):
        return len(self.L)


class XmediaDataset(torchdata.Dataset):
    def __init__(self, txt_file):
        txt_paths = []
        f = open(txt_file, 'r')
        for line in f:
            if line == '':
                break
            line = line.encode('utf-8').decode('utf-8-sig').strip().split()
            txt_paths.append(line[0])
        # print(txt_path[1])
        self.T = txt_paths

    def __getitem__(self, index):
        # load text
        txt_path = './data/XMediaNet_Feature/text/' + self.T[index]
        txt = ''
        for line in open(txt_path, 'r'):
            if line == '':
                break
            txt += line
        txt = clean_str(txt)
        text = ''
        for i in range(len(txt)):
            text += txt[i] + ' '
        text = '[CLS] ' + text + ' [SEP]'
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        if len(indexed_tokens) > 480:
            print('max then 480:', len(indexed_tokens))
        padding = [0] * (480 - len(indexed_tokens))
        indexed_tokens += padding
        tokens_tensor = torch.tensor([indexed_tokens])[0]
        segments_tensors = torch.tensor([0] * (480))

        return tokens_tensor, segments_tensors

    def __len__(self):
        return len(self.T)


def load_data(txt_file, batch_size, shuffle):
    data_ = WikiImg(txt_file)
    loader_ = torchdata.DataLoader(
        dataset=data_,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return loader_

# xmedia net dataset
# hyper parameters
batch_size = 5          
gpu = 'cuda:2'
device = torch.device(gpu if torch.cuda.is_available() else "cpu")

# load data
f = './data/wikipedia_dataset/trainset_txt_img_cat.list'
txt_loader = load_data(f, batch_size=batch_size, shuffle=False)

model = ImgModule()
# torch.nn.DataParallel(model, device_ids=[1, 2, 3])
model = model.to(device)
model.eval()

# embeds = []
# for i, imgs in enumerate(txt_loader):
#     conv3, conv4, embed = model.forward(imgs.to(device))
#     print(i, conv3.size(), conv4.size(), embed.size())
#     embeds.append(embed.data.cpu())
# embeds = torch.cat(embeds)
# print(embeds.size())
# embeds = embeds.cpu().numpy()

# wikipedia dataset
batch_size = 4
gpu = 'cuda:0'
device = torch.device(gpu if torch.cuda.is_available() else "cpu")
root = './data/wikipedia_dataset/'
# # cudnn.benchmark = True

# load data
trf = './data/wikipedia_dataset/trainset_txt_img_cat.list'
tef = './data/wikipedia_dataset/testset_txt_img_cat.list'
tr_loader = load_data(trf, batch_size=batch_size, shuffle=False)
te_loader = load_data(tef, batch_size=batch_size, shuffle=False)

Imodel = ImgModule()
Imodel = Imodel.to(device)
Tmodel = TxtModule_BERT(device)
Tmodel = Tmodel.to(device)
Imodel.eval()
Tmodel.eval()

# train set
tr_i = []
tr_t = []
tr_l = []
for i, (img, token_idxs, segments_idxs, label) in enumerate(tr_loader):
    Iembed = Imodel(img.to(device))
    Tembed = Tmodel.forward(token_idxs.to(device), segments_idxs.to(device))
    print(i, Iembed.size(), Tembed.size(), label.shape)
    tr_i.append(Iembed.data.cpu())
    tr_t.append(Tembed.data.cpu())
    tr_l.append(label)
tr_i = torch.cat(tr_i)
tr_t = torch.cat(tr_t)
tr_l = torch.cat(tr_l)
tr_i = tr_i.cpu().numpy()
tr_t = tr_t.cpu().numpy()
tr_l = tr_l.cpu().numpy()
# tr_l = tr_l.reshape(-1, 1)
# tr_l = ind2vec(tr_l)
print('tr_i:', tr_i.shape, 'tr_t:', tr_t.shape, 'tr_l:', tr_l.shape)
savemat(root + 'vgg_bert/' + 'train_img.mat', {'train_img':tr_i})
savemat(root + 'vgg_bert/' + 'train_txt.mat', {'train_txt':tr_t})
savemat(root + 'vgg_bert/' + 'train_img_lab.mat', {'train_img_lab':tr_l})

# test set
te_i = []
te_t = []
te_l = []
for i, (img, token_idxs, segments_idxs, label) in enumerate(te_loader):
    Iembed = Imodel(img.to(device))
    Tembed = Tmodel.forward(token_idxs.to(device), segments_idxs.to(device))
    print(i, Iembed.size(), Tembed.size(), label.shape)
    te_i.append(Iembed.data.cpu())
    te_t.append(Tembed.data.cpu())
    te_l.append(label)
te_i = torch.cat(te_i)
te_t = torch.cat(te_t)
te_l = torch.cat(te_l)
te_i = te_i.cpu().numpy()
te_t = te_t.cpu().numpy()
te_l = te_l.cpu().numpy()
# te_l = te_l.reshape(-1, 1)
# te_l = ind2vec(te_l)
print('te_i:', te_i.shape, 'te_t:', te_t.shape, 'te_l:', te_l.shape)
savemat(root + 'vgg_bert/' + 'test_img.mat', {'test_img':te_i})
savemat(root + 'vgg_bert/' + 'test_txt.mat', {'test_txt':te_t})
savemat(root + 'vgg_bert/' + 'test_img_lab.mat', {'test_img_lab':te_l})




