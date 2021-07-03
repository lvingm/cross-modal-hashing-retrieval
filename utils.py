import torch
import numpy as np
from numpy.linalg import norm

def text2binary(dataloader, model, device):
    binary_code = []
    labels = []
    with torch.no_grad():
        for data, label in dataloader:
            labels.append(label)
            output = model(data.to(device))
            binary_code.append(output.data.cpu())
    return torch.sign(torch.cat(binary_code)), torch.cat(labels)

def img2binary(dataloader, model, device):
    binary_code = []
    labels = []
    with torch.no_grad():
        for data, label in dataloader:
            labels.append(label)
            output = model(data.to(device))
            binary_code.append(output.data.cpu())
    return torch.sign(torch.cat(binary_code)), torch.cat(labels)


def IT2binary(dataloader, I_model, T_model, device):
    IB = []
    TB = []
    labels = []
    with torch.no_grad():
        for image, text, label in dataloader:
            labels.append(label)
            Ioutput = I_model(image.to(device))
            text = text.to(device)
            Toutput = T_model(text)
            IB.append(Ioutput.data.cpu())
            TB.append(Toutput.data.cpu())
    return torch.sign(torch.cat(IB)), torch.sign(torch.cat(TB)), torch.cat(labels)

def IT2binary1(dataloader, I_model, T_model, device):
    IB = []
    TB = []
    labels = []
    with torch.no_grad():
        for image,  _, _, text, label in dataloader:
            labels.append(label)
            image = image.unsqueeze(1)
            text = text.unsqueeze(1)        
            Ioutput = I_model(image.to(device))
            if type(text) != tuple:
                text = text.to(device)
            Toutput = T_model(text)
            IB.append(Ioutput.data.cpu())
            TB.append(Toutput.data.cpu())
    return torch.sign(torch.cat(IB)), torch.sign(torch.cat(TB)), torch.cat(labels)


def IT2binary2(dataloader, I_model, T_model, device):
    IB = []
    TB = []
    labels = []
    with torch.no_grad():
        for batch_idx, (imgs, conv3, conv4, text, label) in enumerate(dataloader):
            imgs = imgs.to(device)
            text = text.to(device)
            conv3 = conv3.to(device)
            conv4 = conv4.to(device)
            # print('imgs', imgs.size(), 'text', text.size(), 'labels', labels.size())
            # encoder
            labels.append(label)
            Ioutput = I_model(imgs, conv3, conv4)   ###
            # Ioutput = I_model(imgs)
            Toutput= T_model(text.to(device))     
            IB.append(Ioutput.data.cpu())
            TB.append(Toutput.data.cpu())
    return torch.sign(torch.cat(IB)), torch.sign(torch.cat(TB)), torch.cat(labels)
    # return torch.round(torch.cat(IB)), torch.round(torch.cat(TB)), torch.cat(labels)

def IT2binary3(dataloader, I_model, T_model, device):
    IB = []
    TB = []
    labels = []
    with torch.no_grad():
        for batch_idx, (imgs, conv3, conv4, conv5, text, label) in enumerate(dataloader):
            imgs = imgs.to(device)
            text = text.to(device)
            conv3 = conv3.to(device)
            conv4 = conv4.to(device)
            conv5 = conv5.to(device)
            # print('imgs', imgs.size(), 'text', text.size(), 'labels', labels.size())
            # encoder
            labels.append(label.data.cpu())
            Ioutput = I_model(imgs, conv3, conv4, conv5)   ###
            # Ioutput = I_model(imgs)
            Toutput= T_model(text)
            IB.append(Ioutput.data.cpu())     
            TB.append(Toutput.data.cpu())
    return torch.sign(torch.cat(IB)), torch.sign(torch.cat(TB)), torch.cat(labels)


def TSTbinary0(dataloader, Imemory, Tmemory, Iembed_model, Tembed_model, Ilstm, Tlstm, device, args):
    IB = []
    TB = []     
    labels = []
    with torch.no_grad():
        for image, text, label in dataloader:
            labels.append(label)
            Iembed, Tembed = Iembed_model(image.to(device)), Tembed_model(text.to(device))
            Ihash = Ilstm(Iembed.to(device), Imemory.to(device))
            Thash = Tlstm(Tembed.to(device), Tmemory.to(device))
            IB.append(Ihash.data.cpu())
            TB.append(Thash.data.cpu())
    return torch.sign(torch.cat(IB)), torch.sign(torch.cat(TB)), torch.cat(labels)


def TSTbinary1(dataloader, Imemory, Tmemory, Iembed_model, Tembed_model, Ilstm, Tlstm, device, args):
    IB = []
    TB = []
    labels = []
    with torch.no_grad():
        for batch_idx, (imgs, conv3, conv4, text, label) in enumerate(dataloader):
            imgs = imgs.to(device)
            text = text.to(device)
            conv3 = conv3.to(device)
            conv4 = conv4.to(device)
            labels.append(label)
            Iembed, Tembed = Iembed_model(imgs, conv3, conv4), Tembed_model(text)
            Iconcate = torch.Tensor(Iembed.size(0), args.cls, Iembed.size(1) * 2).to(device)
            Tconcate = torch.Tensor(Iembed.size(0), args.cls, Iembed.size(1) * 2).to(device)
            for i in range(Iembed.size(0)):
                for j in range(args.cls):
                    Iconcate[i, j] = torch.cat((Imemory[j], Iembed[i].data.cpu()))
                    Tconcate[i, j] = torch.cat((Tmemory[j], Tembed[i].data.cpu()))
            # bid-lstm and hashing
            Ihash = Ilstm(Iconcate)
            Thash = Tlstm(Tconcate)
            IB.append(Ihash.data.cpu())
            TB.append(Thash.data.cpu())
    return torch.sign(torch.cat(IB)), torch.sign(torch.cat(TB)), torch.cat(labels)

def TSTbinary2(dataloader, Imemory, Tmemory, Iembed_model, Tembed_model, Ilstm, Tlstm, device, args):
    IB = []
    TB = []
    labels = []
    with torch.no_grad():
        for image, text, label in dataloader:
            labels.append(label)
            Iembed, Tembed = Iembed_model(image.to(device)), Tembed_model(text.to(device))
            Iconcate = torch.Tensor(Iembed.size(0), args.cls + 1, Iembed.size(1))
            Tconcate = torch.Tensor(Iembed.size(0), args.cls + 1, Tembed.size(1))
            for i in range(Iembed.size(0)):
                Iconcate[i] = torch.cat((Iembed[i].data.cpu().unsqueeze(0), Imemory), dim=0)
                Tconcate[i] = torch.cat((Tembed[i].data.cpu().unsqueeze(0), Tmemory), dim=0)
            _, Ihash = Ilstm(Iconcate.to(device), device)
            _, Thash = Tlstm(Tconcate.to(device), device)
            IB.append(Ihash.data.cpu())
            TB.append(Thash.data.cpu())
    return torch.sign(torch.cat(IB)), torch.sign(torch.cat(TB)), torch.cat(labels)

def TSTbinary(dataloader, memory, Iembed_model, Tembed_model, Ilstm, Tlstm, device, args):
    IB = []
    TB = []
    labels = []
    with torch.no_grad():
        for image, text, label in dataloader:
            labels.append(label)
            Iembed, Tembed = Iembed_model(image.to(device)), Tembed_model(text.to(device))
            Ihash = Ilstm(Iembed.to(device), memory.to(device))
            Thash = Tlstm(Tembed.to(device), memory.to(device))
            IB.append(Ihash.data.cpu())
            TB.append(Thash.data.cpu())
    return torch.sign(torch.cat(IB)), torch.sign(torch.cat(TB)), torch.cat(labels)

def TSTbinary3(dataloader, Imemory, Tmemory, Iembed_model, Tembed_model, Ilstm, Tlstm, device, args):
    IB = []
    TB = []
    labels = []
    with torch.no_grad():
        for batch_idx, (imgs, conv3, conv4, text, label) in enumerate(dataloader):
            imgs = imgs.to(device)
            text = text.to(device)
            conv3 = conv3.to(device)
            conv4 = conv4.to(device)
            labels.append(label)
            Iembed, Tembed = Iembed_model(imgs, conv3, conv4), Tembed_model(text)
            Iconcate = torch.Tensor(Iembed.size(0), args.cls + 1, Iembed.size(1))
            Tconcate = torch.Tensor(Iembed.size(0), args.cls + 1, Tembed.size(1))
            for i in range(Iembed.size(0)):
                Iconcate[i] = torch.cat((Iembed[i].data.cpu().unsqueeze(0), Imemory), dim=0)
                Tconcate[i] = torch.cat((Tembed[i].data.cpu().unsqueeze(0), Tmemory), dim=0)
            _, Ihash = Ilstm(Iconcate.to(device), device)
            _, Thash = Tlstm(Tconcate.to(device), device)
            IB.append(Ihash.data.cpu())
            TB.append(Thash.data.cpu())
    return torch.sign(torch.cat(IB)), torch.sign(torch.cat(TB)), torch.cat(labels)

def TSTbinary4(dataloader, memory, Iembed_model, Tembed_model, Ilstm, Tlstm, device, args):
    IB = []
    TB = []
    labels = []
    with torch.no_grad():
        for batch_idx, (imgs, conv3, conv4, text, label) in enumerate(dataloader):
            imgs = imgs.to(device)
            text = text.to(device)
            conv3 = conv3.to(device)
            conv4 = conv4.to(device)
            labels.append(label)
            Iembed, Tembed = Iembed_model(imgs, conv3, conv4), Tembed_model(text)
            Ihash = Ilstm(memory.to(device), Iembed.to(device), device)
            Thash = Tlstm(memory.to(device), Tembed.to(device), device)
            IB.append(Ihash.data.cpu())
            TB.append(Thash.data.cpu())
        return torch.sign(torch.cat(IB)), torch.sign(torch.cat(TB)), torch.cat(labels)


def TSTbinary5(dataloader, Imemory, Tmemory, Iembed_model, Tembed_model, Ilstm, Tlstm, device, args):
    IB = []
    TB = []
    labels = []
    with torch.no_grad():
        for batch_idx, (imgs, conv3, conv4, text, label) in enumerate(dataloader):
            imgs = imgs.to(device)
            text = text.to(device)
            conv3 = conv3.to(device)
            conv4 = conv4.to(device)
            labels.append(label)
            Iembed, Tembed = Iembed_model(imgs, conv3, conv4), Tembed_model(text)
            Ihash = Ilstm(Imemory.to(device), Iembed.to(device), device)
            Thash = Tlstm(Tmemory.to(device), Tembed.to(device), device)   
            IB.append(Ihash.data.cpu())
            TB.append(Thash.data.cpu())
    return torch.sign(torch.cat(IB)), torch.sign(torch.cat(TB)), torch.cat(labels)



def compute_AP(query_binarys, query_labels, db_binarys, db_labels, device):
    AP = []
    query_binarys = query_binarys.to(device)
    query_labels = query_labels.to(device)
    db_binarys = db_binarys.to(device)
    db_labels = db_labels.to(device)
    return_num = db_binarys.size(0)
    query_num = query_binarys.size(0)
    Ns = torch.arange(1, return_num + 1)
    Ns = Ns.type(torch.FloatTensor)
    for i in range(query_num):
        query_binary, query_label = query_binarys[i], query_labels[i]
        # query_label[query_label == 0] = -1
        _, sort_idx = torch.sum((query_binary != db_binarys).long(), dim=1).sort()
        correct = (torch.sum(db_labels[sort_idx] * query_label, dim=1) > 0).type(torch.FloatTensor)
        P = (torch.cumsum(correct, dim=0) / Ns).type(torch.FloatTensor)
        AP.append(torch.sum(P * correct) / torch.sum(correct))
    return torch.Tensor(AP)


def compute_NDCG(query_binarys, query_labels, db_binarys, db_labels):
    query_binarys = query_binarys.cpu().numpy()
    query_labels = query_labels.cpu().numpy()
    db_binarys = db_binarys.cpu().numpy()
    db_labels = db_labels.cpu().numpy()
    query_num = query_binarys.shape[0]
    top_k = db_binarys.shape[0]

    query_labels[query_labels==0.] = -1.
    db_labels[db_labels==0.] = -1.

    mean_ndcg = 0.
    for i in range(query_num):
        query_binary, query_label = query_binarys[i], query_labels[i]
        rel_idx = np.argsort((query_binary != db_binarys).sum(1))

        cos_sim = np.zeros(db_labels.shape[0])
        for j in range(db_labels.shape[0]):
            cos_sim[j] = np.matmul(query_label, np.transpose(db_labels[j])) / norm(query_label) / norm(db_labels[j])
        ideal_idx = np.argsort(-cos_sim)

        dcg = 0.
        ideal_dcg = 0.
        for j in range(top_k):
            if j == 0:
                dcg += cos_sim[rel_idx[j]]
                ideal_dcg += cos_sim[ideal_idx[j]]
            else:
                dcg += cos_sim[rel_idx[j]] / np.log(j + 1)
                ideal_dcg += cos_sim[ideal_idx[j]] / np.log(j + 1)
        if ideal_dcg == 0.:
            mean_ndcg += 0.      
        else:
            mean_ndcg += dcg / ideal_dcg
    return mean_ndcg / query_num


def save_binary(binary_codes, labels, file):
    binary_codes = binary_codes.cpu().numpy().astype('int')
    labels = labels.cpu().numpy().astype('int')
    if labels.shape[1] != 1:
        labels_ = np.argwhere(labels == 1)[:, 1]
    else:
        labels_ = labels.squeeze(1)
    f = open(file, 'w+')
    lines = []
    for i in range(binary_codes.shape[0]):
        string = ''
        for element in binary_codes[i]:  
            if element == 1:
                string += '1 '
            else:
                string += '0 '
        string += '  ' + str(labels_[i]) + '\n'
        lines.append(string)
    f.writelines(lines)
    f.close()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']