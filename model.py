import torch.nn as nn
import math
import numpy as np
import torchvision.models as models
import torch
from Non_local import NONLocalBlock1D, NONLocalBlock2D
import torch.nn.functional as F
from torch.autograd import Variable
from gensim.models.keyedvectors import KeyedVectors


class IPyramid(nn.Module):
    def __init__(self, conv3_dim, conv4_dim, embeds_dim, hash_length, nl_version):
        super(IPyramid, self).__init__()
        self.fc1 = nn.Linear(conv3_dim, embeds_dim)
        self.fc2 = nn.Linear(conv4_dim, embeds_dim)
        if nl_version != 'None':
            self.non_local = NONLocalBlock2D(in_channels=embeds_dim)  # non-local block with dot product
        self.fcs = nn.Sequential(
            nn.Linear(embeds_dim, 2048),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.hash_layer = nn.Sequential(
            nn.Linear(512, hash_length),
            nn.Tanh()
        )
        self.nl_version = nl_version
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)

    def forward(self, embed, conv3, conv4):
        if self.nl_version == 'None':
            features = embed
        else:
            x0 = self.fc1(conv3)
            # print('x0:', x0.unsqueeze(1).size())
            x1 = self.fc2(conv4)
            # print('x1:', x1.unsqueeze(1).size())
            x = torch.cat([x0.unsqueeze(1), x1.unsqueeze(1), embed.unsqueeze(1)], dim=1)
            # print('Ix:', x.size())
            x = x.permute(0, 2, 1)
            x = x.unsqueeze(2)  # [b, 4096, 1, 3]
            # print('x:', x.size())
            if self.nl_version == 'dot':
                nl = self.non_local.forward_dot(x)
            elif self.nl_version == 'gaussian':
                nl = self.non_local.forward_gaussian(x)
            elif self.nl_version == 'embed_gau':
                nl = self.non_local.forward_embed_gau(x)
            else:
                nl = self.non_local.forward_concate(x)
            features = nl.squeeze(2).mean(2)
        output = self.fcs(features)
        hash_codes = self.hash_layer(output)
        return hash_codes


class TPyramid(nn.Module):
    def __init__(self, input_dim, hash_length, nl_version):
        super(TPyramid, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 4096), nn.ReLU())
        self.fc11 = nn.Linear(4096, 512)
        self.fc2 = nn.Sequential(nn.Linear(4096, 2048), nn.ReLU())
        # self.fc11 = nn.Linear(input_dim, 512)
        # self.fc2 = nn.Sequential(nn.Linear(input_dim, 2048), nn.ReLU())
        self.fc21 = nn.Linear(2048, 512)
        self.fc3 = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        if nl_version != 'None':
            self.non_local = NONLocalBlock2D(in_channels=512)  # non-local block with dot product

        self.hash_layer = nn.Sequential(
            nn.Linear(512, hash_length),
            nn.Tanh()
        )
        self.nl_version = nl_version
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)

    def forward(self, bow):
        x1 = self.fc1(bow)
        x2 = self.fc2(x1)
        # x2 = self.fc2(bow)
        x3 = self.fc3(x2)
        x4 = self.fc4(x3)
        if self.nl_version == 'None':
            features = x4
        else:
            x11 = self.fc11(x1)
            # x11 = self.fc11(bow)
            x21 = self.fc21(x2)
            x = torch.cat([x11.unsqueeze(1), x21.unsqueeze(1), x4.unsqueeze(1)], dim=1)
            x = x.permute(0, 2, 1)
            x = x.unsqueeze(2)
            # print('x:', x.size())
            if self.nl_version == 'dot':
                nl = self.non_local.forward_dot(x)
            elif self.nl_version == 'gaussian':
                nl = self.non_local.forward_gaussian(x)
            elif self.nl_version == 'embed_gau':
                nl = self.non_local.forward_embed_gau(x)
            else:
                nl = self.non_local.forward_concate(x)
            features = nl.squeeze(2).mean(2)
        hash_codes = self.hash_layer(features)
        return hash_codes


class IExtractor(nn.Module):
    def __init__(self, conv3_dim, conv4_dim, embeds_dim, nl_version):
        super(IExtractor, self).__init__()
        self.fc1 = nn.Linear(conv3_dim, embeds_dim)
        self.fc2 = nn.Linear(conv4_dim, embeds_dim)
        if nl_version != 'None':
            self.non_local = NONLocalBlock2D(in_channels=embeds_dim)  # non-local block with dot product
        self.fcs = nn.Sequential(
            nn.Linear(embeds_dim, 2048),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.nl_version = nl_version
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)

    def forward(self, embed, conv3, conv4):
        if self.nl_version == 'None':
            features = embed
        else:
            x0 = self.fc1(conv3)
            # print('x0:', x0.unsqueeze(1).size())
            x1 = self.fc2(conv4)
            # print('x1:', x1.unsqueeze(1).size())
            x = torch.cat([x0.unsqueeze(1), x1.unsqueeze(1), embed.unsqueeze(1)], dim=1)
            # print('Ix:', x.size())
            x = x.permute(0, 2, 1)
            x = x.unsqueeze(2)  # [b, 4096, 1, 3]
            # print('x:', x.size())
            if self.nl_version == 'dot':
                nl = self.non_local.forward_dot(x)
            elif self.nl_version == 'gaussian':
                nl = self.non_local.forward_gaussian(x)
            elif self.nl_version == 'embed_gau':
                nl = self.non_local.forward_embed_gau(x)
            else:
                nl = self.non_local.forward_concate(x)
            features = nl.squeeze(2).mean(2)
        output = self.fcs(features)
        return output


class TExtractor(nn.Module):
    def __init__(self, input_dim, nl_version):
        super(TExtractor, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 4096), nn.ReLU())
        self.fc11 = nn.Linear(4096, 512)
        self.fc2 = nn.Sequential(nn.Linear(4096, 2048), nn.ReLU())
        # self.fc11 = nn.Linear(input_dim, 512)
        # self.fc2 = nn.Sequential(nn.Linear(input_dim, 2048), nn.ReLU())
        self.fc21 = nn.Linear(2048, 512)
        self.fc3 = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        if nl_version != 'None':
            self.non_local = NONLocalBlock2D(in_channels=512)  # non-local block with dot product

        self.nl_version = nl_version
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)

    def forward(self, bow):
        x1 = self.fc1(bow)
        x2 = self.fc2(x1)
        # x2 = self.fc2(bow)
        x3 = self.fc3(x2)
        x4 = self.fc4(x3)
        if self.nl_version == 'None':
            features = x4
        else:
            x11 = self.fc11(x1)
            # x11 = self.fc11(bow)
            x21 = self.fc21(x2)
            x = torch.cat([x11.unsqueeze(1), x21.unsqueeze(1), x4.unsqueeze(1)], dim=1)
            x = x.permute(0, 2, 1)
            x = x.unsqueeze(2)
            # print('x:', x.size())
            if self.nl_version == 'dot':
                nl = self.non_local.forward_dot(x)
            elif self.nl_version == 'gaussian':
                nl = self.non_local.forward_gaussian(x)
            elif self.nl_version == 'embed_gau':
                nl = self.non_local.forward_embed_gau(x)
            else:
                nl = self.non_local.forward_concate(x)
            features = nl.squeeze(2).mean(2)
        return features


class IFCS(nn.Module):  # non-local + hash
    def __init__(self, input_dim, hash_length):
        super(IFCS, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.hash_layer = nn.Sequential(
            nn.Linear(512, hash_length),
            nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)

    def forward(self, x):
        output = self.fcs(x)
        hash_codes = self.hash_layer(output)
        return hash_codes


class TFCS(nn.Module):  # non-local + hash
    def __init__(self, input_dim, hidden_size, hash_length):
        super(TFCS, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, bidirectional=True, batch_first=True)
        self.fcs = nn.Sequential(
            nn.Linear(hidden_size * 2, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.hash_layer = nn.Sequential(
            nn.Linear(512, hash_length),
            nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)

    def forward(self, x):
        embeds, _ = self.lstm(x)  # (b, n, 2h)
        features = embeds[:, -1, :]
        out = self.fcs(features)
        hash_codes = self.hash_layer(out)
        return hash_codes


# bid-lstm + hash_layer
class BidLSTM(nn.Module):
    def __init__(self, input_dim, hash_length, device, hidden_size=512):
        super(BidLSTM, self).__init__()
        self.device = device
        self.bidLstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=1, batch_first=True,
                               bidirectional=True)

        self.hash_layer = nn.Sequential(
            nn.Linear(512, hash_length),
            nn.Tanh()
            # nn.Sigmoid()
        )

    def forward(self, x):
        output, (h_n, c_n) = self.bidLstm(x)  # h_n: [num_layers*num_direction, batch_size, hidden_size]
        new_features = (h_n[0] + h_n[1]) * 0.5
        output = self.hash_layer(new_features)
        return output


class BidLSTM0(nn.Module):
    def __init__(self, input_dim, hash_length, device, hidden_size=1024):
        super(BidLSTM0, self).__init__()
        self.device = device
        self.bidLstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=1, batch_first=True,
                               bidirectional=True)

        self.hash_layer = nn.Sequential(
            nn.Linear(1024, hash_length),
            nn.Tanh()
            # nn.Sigmoid()
        )

    def forward(self, x):
        _, (h_n, c_n) = self.bidLstm(x)  # h_n: [num_layers*num_direction, batch_size, hidden_size]
        new_features = (h_n[0] + h_n[1]) * 0.5
        output = self.hash_layer(new_features)
        return output


class BidLSTM1(nn.Module):
    def __init__(self, input_dim, hash_length):
        super(BidLSTM1, self).__init__()
        self.bidLstm = nn.LSTM(input_size=input_dim, hidden_size=hash_length, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        _, (h_n, c_n) = self.bidLstm(x)  # h_n: [num_layers*num_direction, batch_size, hidden_size]
        output = (h_n[0] + h_n[1]) * 0.5
        output = self.tanh(output)
        return output


class Feature_Extractor(nn.Module):
    def __init__(self, input_dim, output_dim=1024, nl_version='None'):
        super(Feature_Extractor, self).__init__()
        if nl_version != 'None':
            self.non_local = NONLocalBlock1D(in_channels=1)  # non-local block with dot product
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, output_dim),
            nn.ReLU()
        )
        self.nl_version = nl_version  # Non-local version

    def forward(self, x):
        x = x.unsqueeze(1)
        if self.nl_version == 'None':
            nl = x  # without non-local
        elif self.nl_version == 'dot':
            nl = self.non_local.forward_dot(x)
        elif self.nl_version == 'gaussian':
            nl = self.non_local.forward_gaussian(x)
        elif self.nl_version == 'embed_gau':
            nl = self.non_local.forward_embed_gau(x)
        else:
            nl = self.non_local.forward_concate(x)
        nl = nl.squeeze(1)
        features = self.fc(nl)
        return features


class TFeature_Extractor(nn.Module):
    def __init__(self, input_dim, nl_version='None'):
        super(TFeature_Extractor, self).__init__()
        self.fc0 = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU()
        )
        if nl_version != 'None':
            self.non_local = NONLocalBlock1D(in_channels=1)  # non-local block with dot product
        self.fc = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU()
        )
        self._init_weights()
        self.nl_version = nl_version  # Non-local version

    def _init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.fc0(x)
        x = x.unsqueeze(1)
        if self.nl_version == 'None':
            nl = x  # without non-local
        elif self.nl_version == 'dot':
            nl = self.non_local.forward_dot(x)
        elif self.nl_version == 'gaussian':
            nl = self.non_local.forward_gaussian(x)
        elif self.nl_version == 'embed_gau':
            nl = self.non_local.forward_embed_gau(x)
        else:
            nl = self.non_local.forward_concate(x)
        nl = nl.squeeze(1)
        output = self.fc(nl)
        return output


class FC_layer(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(FC_layer, self).__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        nn.init.xavier_uniform_(self.dense.weight)

    def forward(self, x):
        output = self.dense(x)
        output = self.tanh(output)
        return output


class cos_distance(nn.Module):
    def __init__(self):
        super(cos_distance, self).__init__()

    def forward(self, a, b):
        eps = 1e-10
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        cos_sim = torch.mm(a_norm, b_norm.transpose(0, 1))
        return cos_sim


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


def squash(tensor):
    norm = (tensor * tensor).sum(-1)
    scale = norm / (1 + norm)
    return scale.unsqueeze(-1) * tensor / torch.sqrt(norm).unsqueeze(-1)


# generate class vectors
class InductionNetwork(nn.Module):
    def __init__(self, input_dim):
        super(InductionNetwork, self).__init__()
        self.input_dim = input_dim
        self.l_1 = nn.Linear(self.input_dim, self.input_dim, bias=False)

    def forward(self, encoder_output, iter_routing=3):
        C, K, H = encoder_output.shape
        b = torch.zeros(C, K)
        for _ in range(iter_routing):
            d = F.softmax(b, dim=-1)
            encoder_output_hat = self.l_1(encoder_output)
            c_hat = torch.sum(encoder_output_hat * d.unsqueeze(-1), dim=1)
            c = squash(c_hat)

            b = b + torch.bmm(encoder_output_hat, c.unsqueeze(-1)).squeeze()

        return c


# calcualte the relation between class vectors and query vectors
class Relation(nn.Module):
    def __init__(self, input_dim, hash_length, output_size=1):
        super(Relation, self).__init__()
        self.output_size = output_size
        self.M = nn.init.xavier_normal_(torch.FloatTensor(input_dim, input_dim, output_size))
        self.M.requires_grad = True

        self.score_layer = nn.Linear(output_size, 1)
        self.hash_layer = nn.Sequential(
            nn.Linear(input_dim, hash_length),
            nn.Tanh()
        )

    def forward(self, class_vector, query_vector, device):
        mid_pro = []
        for i in range(self.output_size):
            v = self.M[:, :, i]
            inter = torch.mm(query_vector, torch.mm(class_vector, v.to(device)).transpose(0, 1))
            mid_pro.append(inter)
        tensor_bi_product = torch.stack(mid_pro, dim=0)  # (output_size, batch_size, class_num)
        activate = F.relu(tensor_bi_product)
        reshape = activate.permute(1, 2, 0)
        other = self.score_layer(reshape).squeeze()  # (batch_size, class_num)
        relation_scores = F.softmax(other, dim=1)
        # weight sum
        query_features = torch.mm(relation_scores, class_vector) + query_vector  # residual
        # hash
        hash_codes = self.hash_layer(query_features)
        return hash_codes


def GAP(x):  # global average pooling
    N = x.size(-1) * x.size(-2)
    gap = x.sum(-1).sum(-1) / N
    return gap


class ImgNonLocal(nn.Module):
    def __init__(self, conv3_dim, conv4_dim, embeds_dim, hash_length, nl_version):
        super(ImgNonLocal, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(conv3_dim, embeds_dim), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(conv4_dim, embeds_dim), nn.ReLU())

        if nl_version != 'None':
            self.nl_conv3 = NONLocalBlock2D(in_channels=conv3_dim)  # non-local block with dot product
            self.nl_conv4 = NONLocalBlock2D(in_channels=conv4_dim)  # non-local block with dot product
        self.fcs = nn.Sequential(
            nn.Linear(embeds_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            # nn.Linear(embeds_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.hash_layer = nn.Sequential(
            nn.Linear(512, hash_length),
            nn.Tanh()
        )
        self.max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        self.nl_version = nl_version
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)

    def forward(self, embed, conv3, conv4):
        if self.nl_version == 'None':
            features = embed
        else:
            if self.nl_version == 'dot':
                nl_conv3 = GAP(self.nl_conv3.forward_dot(conv3))
                nl_conv4 = GAP(self.nl_conv4.forward_dot(conv4))
            elif self.nl_version == 'gaussian':
                nl_conv3 = GAP(self.nl_conv3.forward_gaussian(conv3))
                nl_conv4 = GAP(self.nl_conv4.forward_gaussian(conv4))
            elif self.nl_version == 'embed_gau':
                nl_conv3 = GAP(self.nl_conv3.forward_embed_gau(conv3))
                nl_conv4 = GAP(self.nl_conv4.forward_embed_gau(conv4))
            else:
                nl_conv3 = GAP(self.nl_conv3.forward_concate(conv3))
                nl_conv4 = GAP(self.nl_conv4.forward_concate(conv4))
            x0 = self.fc1(nl_conv3)
            x1 = self.fc2(nl_conv4)
            nl = torch.cat([x0.unsqueeze(1), x1.unsqueeze(1), embed.unsqueeze(1)], dim=1)
            features = nl.mean(1)
        output = self.fcs(features)
        hash_codes = self.hash_layer(output)
        return hash_codes


class INLextractor(nn.Module):
    def __init__(self, conv3_dim, conv4_dim, embeds_dim, nl_version):
        super(INLextractor, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(conv3_dim, embeds_dim), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(conv4_dim, embeds_dim), nn.ReLU())

        if nl_version != 'None':
            self.nl_conv3 = NONLocalBlock2D(in_channels=conv3_dim)  # non-local block with dot product
            self.nl_conv4 = NONLocalBlock2D(in_channels=conv4_dim)  # non-local block with dot product
        self.fcs = nn.Sequential(
            nn.Linear(embeds_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        self.nl_version = nl_version
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)

    def forward(self, embed, conv3, conv4):
        if self.nl_version == 'None':
            features = embed
        else:
            if self.nl_version == 'dot':
                nl_conv3 = GAP(self.nl_conv3.forward_dot(conv3))
                nl_conv4 = GAP(self.nl_conv4.forward_dot(conv4))
            elif self.nl_version == 'gaussian':
                nl_conv3 = GAP(self.nl_conv3.forward_gaussian(conv3))
                nl_conv4 = GAP(self.nl_conv4.forward_gaussian(conv4))
            elif self.nl_version == 'embed_gau':
                nl_conv3 = GAP(self.nl_conv3.forward_embed_gau(conv3))
                nl_conv4 = GAP(self.nl_conv4.forward_embed_gau(conv4))
            else:
                nl_conv3 = GAP(self.nl_conv3.forward_concate(conv3))
                nl_conv4 = GAP(self.nl_conv4.forward_concate(conv4))
            x0 = self.fc1(nl_conv3)
            x1 = self.fc2(nl_conv4)
            nl = torch.cat([x0.unsqueeze(1), x1.unsqueeze(1), embed.unsqueeze(1)], dim=1)
            features = nl.mean(1)
        output = self.fcs(features)
        return output


class TxtAttention(nn.Module):
    def __init__(self, input_dim, hidden_size, hash_length, nl_version):
        super(TxtAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, bidirectional=True, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.fcs = nn.Sequential(
            nn.Linear(hidden_size * 2, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        self.hash_layer = nn.Sequential(
            nn.Linear(512, hash_length),
            nn.Tanh()
        )
        self._init_weights()
        self.nl_version = nl_version

    def _init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)

    def forward(self, w2v):
        embeds, _ = self.lstm(w2v)  # (b, n, 2h)
        if self.nl_version == 'None':
            features = embeds[:, -1, :]
        else:
            attention_w = F.softmax(self.attention(embeds), dim=1)
            features = (embeds * attention_w).sum(1)  # (b, 2h)
        out = self.fcs(features)
        hash_codes = self.hash_layer(out)
        return hash_codes


class TATextractor(nn.Module):
    def __init__(self, input_dim, hidden_size, nl_version):
        super(TATextractor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, bidirectional=True, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.fcs = nn.Sequential(
            nn.Linear(hidden_size * 2, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        self._init_weights()
        self.nl_version = nl_version

    def _init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)

    def forward(self, w2v):
        embeds, _ = self.lstm(w2v)  # (b, n, 2h)
        if self.nl_version == 'None':
            features = embeds[:, -1, :]
        else:
            attention_w = F.softmax(self.attention(embeds), dim=1)
            features = (embeds * attention_w).sum(1)  # (b, 2h)
        out = self.fcs(features)
        return out


