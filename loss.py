import torch
import torch.nn.functional as F
import numpy as np
import itertools


def triplet_loss(hash_codes, labels, margin, device):
    tloss = torch.tensor(margin + 0., requires_grad=True).to(device)
    labels_ = labels.cpu().data.numpy()
    triplets = []
    for label in labels_:
        label_mask = np.matmul(labels_, np.transpose(label)) > 0  # multi-labels
        label_indices = np.where(label_mask)[0]
        if len(label_indices) < 2:
            continue
        negative_indices = np.where(np.logical_not(label_mask))[0]
        if len(negative_indices) < 1:
            continue
        anchor_positives = list(itertools.combinations(label_indices, 2))
        temp = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                for neg_ind in negative_indices]
        triplets += temp
    if triplets:
        triplets = np.array(triplets)
        # print('triplet', triplets.shape)
        # intra triplet loss
        ap = (hash_codes[triplets[:, 0]] - hash_codes[triplets[:, 1]]).pow(2).sum(1)
        an = (hash_codes[triplets[:, 0]] - hash_codes[triplets[:, 2]]).pow(2).sum(1)
        tloss = F.relu(margin + ap - an).mean()

    return tloss


def inter_triplet_loss(Ihash, Thash, labels, margin, device):
    IT_tloss = TI_tloss = torch.tensor(margin + 0., requires_grad=True).to(device)

    labels_ = labels.cpu().data.numpy()
    triplets = []
    for label in labels_:
        label_mask = np.matmul(labels_, np.transpose(label)) > 0  # multi-labels
        label_indices = np.where(label_mask)[0]
        if len(label_indices) < 2:
            continue
        negative_indices = np.where(np.logical_not(label_mask))[0]
        if len(negative_indices) < 1:
            continue
        anchor_positives = list(itertools.combinations(label_indices, 2))
        temp = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                for neg_ind in negative_indices]
        triplets += temp

    if triplets:
        triplets = np.array(triplets)
        # inter triplet loss: I -> T, T -> I
        IT_ap = (Ihash[triplets[:, 0]] - Thash[triplets[:, 1]]).pow(2).sum(1)
        IT_an = (Ihash[triplets[:, 0]] - Thash[triplets[:, 2]]).pow(2).sum(1)
        TI_ap = (Thash[triplets[:, 0]] - Ihash[triplets[:, 1]]).pow(2).sum(1)
        TI_an = (Thash[triplets[:, 0]] - Ihash[triplets[:, 2]]).pow(2).sum(1)
        IT_tloss = F.relu(margin + IT_ap - IT_an).mean()
        TI_tloss = F.relu(margin + TI_ap - TI_an).mean()

    return IT_tloss, TI_tloss


def intra_inter_tl(Ihash, Thash, labels, margin, device):
    ## calculate intra and inter triplet loss
    II_tloss = TT_tloss = IT_tloss = TI_tloss = torch.tensor(margin + 0., requires_grad=True).to(device)

    labels_ = labels.cpu().data.numpy()
    triplets = []
    for label in labels_:
        label_mask = np.matmul(labels_, np.transpose(label)) > 0  # multi-labels
        label_indices = np.where(label_mask)[0]
        if len(label_indices) < 2:
            continue
        negative_indices = np.where(np.logical_not(label_mask))[0]
        if len(negative_indices) < 1:
            continue
        anchor_positives = list(itertools.combinations(label_indices, 2))
        temp = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                for neg_ind in negative_indices]
        triplets += temp

    if triplets:
        triplets = np.array(triplets)  
        # intra triplet loss: I -> I, T -> T
        II_ap = (Ihash[triplets[:, 0]] - Ihash[triplets[:, 1]]).pow(2).sum(1)
        II_an = (Ihash[triplets[:, 0]] - Ihash[triplets[:, 2]]).pow(2).sum(1)
        TT_ap = (Thash[triplets[:, 0]] - Thash[triplets[:, 1]]).pow(2).sum(1)
        TT_an = (Thash[triplets[:, 0]] - Thash[triplets[:, 2]]).pow(2).sum(1)
        II_tloss = F.relu(margin + II_ap - II_an).mean()
        TT_tloss = F.relu(margin + TT_ap - TT_an).mean()

        # inter triplet loss: I -> T, T -> I
        IT_ap = (Ihash[triplets[:, 0]] - Thash[triplets[:, 1]]).pow(2).sum(1)
        IT_an = (Ihash[triplets[:, 0]] - Thash[triplets[:, 2]]).pow(2).sum(1)
        TI_ap = (Thash[triplets[:, 0]] - Ihash[triplets[:, 1]]).pow(2).sum(1)
        TI_an = (Thash[triplets[:, 0]] - Ihash[triplets[:, 2]]).pow(2).sum(1)
        IT_tloss = F.relu(margin + IT_ap - IT_an).mean()
        TI_tloss = F.relu(margin + TI_ap - TI_an).mean()
    return II_tloss, TT_tloss, IT_tloss, TI_tloss


