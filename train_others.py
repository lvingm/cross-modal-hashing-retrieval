import argparse
import torch.utils.data as data
import os
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from data_load import *
from model import *
from utils import *
from loss import *
import torch
from scipy.io import loadmat,savemat


def load_data(data_file, mode, batch_size, shuffle):
    data_ = MatDataset1(data_file, mode)
    loader_ = data.DataLoader(
        dataset=data_,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return loader_


def load_state(model, model_path):
    model_dict = model.state_dict()
    # print(model)
    pretrained_dict = torch.load(model_path, map_location="cpu").state_dict()

    # print('model:', model_dict.keys())
    # print('pre dicts:', pretrained_dict.keys())

    key = list(pretrained_dict.keys())
    # print(key)
    # 1. filter out unnecessary keys
    # 1.1 multi-GPU ->CPU
    if (str(key).startswith("module.")):
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if
                           k[7:] in model_dict and v.size() == model_dict[k[7:]].size()}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.size() == model_dict[k].size()}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # print('model update:', model.state_dict())


def support_memory2(args, device, Iembed_model, Tembed_model):
    ## 把均值做为各类的特征，图像和文本不同memory
    data_root = f'./data/{args.dataset}/few_shot/{args.k}shot/ITL.mat'
    train_loader = load_data(data_root, mode='train', batch_size=args.batch_size, shuffle=False)
    Iembeds = []
    Tembeds = []
    labels = []
    Iembed_model.eval()
    Tembed_model.eval()
    for batch_idx, (imgs, conv3, conv4, text, label) in enumerate(train_loader):
        imgs = imgs.to(device)
        text = text.to(device)
        conv3 = conv3.to(device)
        conv4 = conv4.to(device)
        # encoder
        Iembeds.append(Iembed_model(imgs, conv3, conv4).data.cpu())
        Tembeds.append(Tembed_model(text).data.cpu())
        labels.append(torch.argmax(label, dim=1))

    labels = torch.cat(labels)
    Iembeds = torch.cat(Iembeds)
    Tembeds = torch.cat(Tembeds)

    # mean embedding for per class
    I_mean = torch.zeros((args.cls, Tembeds.size(1)))
    T_mean = torch.zeros((args.cls, Tembeds.size(1)))
    cnt = torch.zeros(args.cls)
    for i in range(Iembeds.size(0)):
        I_mean[labels[i]] += Iembeds[labels[i]]
        T_mean[labels[i]] += Tembeds[labels[i]]
        cnt[labels[i]] += 1
    for i in range(args.cls):
        I_mean[i] /= cnt[i]
        T_mean[i] /= cnt[i]

    return I_mean, T_mean


def support_memory0(args, device, Iembed_model, Tembed_model):
    # 随机从各类中选出一个样本，将其embedding做为各类的embedding
    tr_i = loadmat(f'./data/{args.dataset}/few_shot/{args.k}shot_more/ITL.mat')['tr_i']
    tr_t = loadmat(f'./data/{args.dataset}/few_shot/{args.k}shot_more/ITL.mat')['tr_t']
    tr_l = loadmat(f'./data/{args.dataset}/few_shot/{args.k}shot_more/ITL.mat')['tr_l']
    tr_conv3 = loadmat(f'./data/{args.dataset}/few_shot/{args.k}shot_more/ITL.mat')['tr_conv3']
    tr_conv4 = loadmat(f'./data/{args.dataset}/few_shot/{args.k}shot_more/ITL.mat')['tr_conv4']
    labels = np.argmax(tr_l, axis=1)

    idxs = np.zeros(args.cls, dtype=int)
    for i in range(args.cls):
        idxs[i] = np.random.choice(np.where(labels == i)[0], 1)[0]

    imgs = torch.from_numpy(tr_i[idxs]).to(device)
    texts = torch.from_numpy(tr_t[idxs]).to(device)
    conv3  = torch.from_numpy(tr_conv3[idxs]).to(device)
    conv4  = torch.from_numpy(tr_conv4[idxs]).to(device)
    Iembed_model.eval()
    Tembed_model.eval()
    Imemory = Iembed_model(imgs, conv3, conv4)
    Tmemory = Tembed_model(texts)
    return Imemory, Tmemory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train image and text network with triplet loss')

    parser.add_argument('--dataset', type=str, default='wikipedia_dataset', help='the name of dataset')
    parser.add_argument('--cls', type=int, default=10, help='the number of categories')

    parser.add_argument('--load', type=int, default=0, help='if load trained model before')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--hash_length', type=int, default=16, help='length of hashing binary')
    parser.add_argument('--margin', type=int, default=8, help='loss_type')
    parser.add_argument('--optim', type=str, default='adam', help='choose optimizer')
    parser.add_argument('--k', type=int, default=1, help='k-shot')
    parser.add_argument('--n', type=int, default=20, help='the number of sampling times of those few samples')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of scheduler')
    parser.add_argument('--nl_version', type=str, default='None', help='version of non-local network')
    parser.add_argument('--update', type=int, default=5, help='update support memory after per n epoch')

    parser.add_argument('--max_epoch', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--lr1', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr2', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--step_size', type=int, default=250, help='the step size to change lr')
    parser.add_argument('--output_size', type=int, default=8, help='the output size of relation network')

    parser.add_argument('--gpu', type=str, default='cuda:0', help='which GPU to use')
    parser.add_argument('--seed', type=int, default=100, help='random seed')

    args = parser.parse_args()     
    print(args)

    torch.manual_seed(args.seed)
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)
    torch.autograd.set_detect_anomaly(True)

    # tensorboard
    writer = SummaryWriter(f'2Fusion_{args.dataset}m{args.margin}K{args.k}n{args.n}')

    # load data
    tr_data_root = f'./data/{args.dataset}/few_shot/{args.k}shot_more/ITL.mat'
    data_root = f'./data/{args.dataset}/ITL.mat'
    train_loader = load_data(tr_data_root, mode='train', batch_size=args.batch_size, shuffle=True)
    test_loader = load_data(data_root, mode='test', batch_size=args.batch_size, shuffle=False)
    db_loader = load_data(data_root, mode='db', batch_size=args.batch_size, shuffle=False)
    print('train_loader:', len(train_loader), 'test_loader:', len(test_loader), 'db_loader:', len(db_loader))

    ## build models: pretrained model with triplet loss
    Iembed_model = IExtractor(256, 512, 4096, args.nl_version)
    Iembed_model = Iembed_model.to(device)
    if args.dataset == 'mir-25k':
        Tembed_model = TExtractor(1386, args.nl_version)
    elif args.dataset == 'nuswide':
        Tembed_model = TExtractor(input_dim=1000, nl_version=args.nl_version)
    else:
        Tembed_model = TExtractor(768, args.nl_version) # wikipedia
    Tembed_model = Tembed_model.to(device)
    Ilstm = Relation(input_dim=512, hash_length=args.hash_length, output_size=args.output_size)
    Tlstm = Relation(input_dim=512, hash_length=args.hash_length)
    Ilstm = Ilstm.to(device)
    Tlstm = Tlstm.to(device)

    load_state(Iembed_model,
               f"./logs/{args.dataset}/pyramid/IT_fewshot/I_model/{args.hash_length}bits/1m{args.margin}_k{args.k}.pth")
    load_state(Tembed_model,
               f"./logs/{args.dataset}/pyramid/IT_fewshot/T_model/{args.hash_length}bits/1m{args.margin}_k{args.k}.pth")

    Imemory, Tmemory = support_memory2(args, device, Iembed_model, Tembed_model)

    max_MAP = max_IT_MAP = max_TI_MAP = max_seen_IT = max_seen_TI = max_unseen_IT = max_unseen_TI = 0.0
    # # initialize parameters with models trained triplet loss separately I->I T->T
    if args.load:
        print('loading pretrained model....')
        load_state(Iembed_model,
                   f'./logs/{args.dataset}/FS_fusion/Iembed/{args.hash_length}bits/k{args.k}_m{args.margin}4lr.pth')
        load_state(Tembed_model,
                   f'./logs/{args.dataset}/FS_fusion/Tembed/{args.hash_length}bits/k{args.k}_m{args.margin}4lr.pth')
        load_state(Ilstm,
                   f'./logs/{args.dataset}/FS_fusion/Ilstm/{args.hash_length}bits/k{args.k}_m{args.margin}4lr.pth')
        load_state(Tlstm,
                   f'./logs/{args.dataset}/FS_fusion/Tlstm/{args.hash_length}bits/k{args.k}_m{args.margin}4lr.pth')
        Iembed_model.eval()
        Tembed_model.eval()
        Ilstm.eval()
        Tlstm.eval()
        # calculate original mAP
        tst_IB, tst_TB, tst_labels = TSTbinary5(test_loader, Imemory, Tmemory, Iembed_model, Tembed_model, Ilstm, Tlstm, device, args)
        db_IB, db_TB, db_labels = TSTbinary5(db_loader, Imemory, Tmemory, Iembed_model, Tembed_model, Ilstm, Tlstm, device, args)
        # calculate MAP
        OIT_MAP = compute_AP(tst_IB, tst_labels, db_TB, db_labels, device).mean()
        OTI_MAP = compute_AP(tst_TB, tst_labels, db_IB, db_labels, device).mean()
        max_IT_MAP, max_TI_MAP, max_MAP = OIT_MAP, OTI_MAP, (OIT_MAP + OTI_MAP) / 2.
        print(f'original IT_MAP:{OIT_MAP:0.4f} TI_MAP:{OTI_MAP:0.4f} ')

    embed_params = list(Iembed_model.parameters()) + list(Tembed_model.parameters())
    lstm_params = list(Ilstm.parameters()) + list(Tlstm.parameters()) #+ list(adaptive.parameters())
    embed_optimizer = torch.optim.Adam(embed_params, lr=args.lr1, betas=(0.9, 0.999), weight_decay=0.00001)
    lstm_optimizer = torch.optim.Adam(lstm_params, lr=args.lr2, betas=(0.9, 0.999), weight_decay=0.00001)
    embed_scheduler = torch.optim.lr_scheduler.StepLR(embed_optimizer, step_size=args.step_size, gamma=0.5,
                                                      last_epoch=-1)
    lstm_scheduler = torch.optim.lr_scheduler.StepLR(lstm_optimizer, step_size=args.step_size, gamma=0.5, last_epoch=-1)

    # path to save trained models
    ckpt_dir = f'./logs/{args.dataset}/'
    ckpt_root = ckpt_dir + f'/induction_relation_max/'
    Iembed_ckpt = ckpt_root + f'/Iembed/{args.hash_length}bits/'
    Tembed_ckpt = ckpt_root + f'/Tembed/{args.hash_length}bits/'
    Ilstm_ckpt = ckpt_root + f'/Ilstm/{args.hash_length}bits/'
    Tlstm_ckpt = ckpt_root + f'/Tlstm/{args.hash_length}bits/'
    adaptive_ckpt = ckpt_root + f'/adaptive/{args.hash_length}bits/'
    if not os.path.exists(Iembed_ckpt):
        os.makedirs(Iembed_ckpt)
        os.makedirs(Tembed_ckpt)
        os.makedirs(Ilstm_ckpt)
        os.makedirs(Tlstm_ckpt)
        os.makedirs(adaptive_ckpt)

    # train
    max_epoch = 0
    print('start to train the image model and text model......')
    for epoch in range(args.max_epoch):
        Iembed_model.train()
        Ilstm.train()
        Tembed_model.train()
        Tlstm.train()
        # adaptive.train()
        epoch_loss = epoch_inter_tloss = epoch_I_tloss = epoch_T_tloss = epoch_IT_loss = epoch_TI_loss = 0.
        for batch_idx, (imgs, conv3, conv4, text, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            text = text.to(device)
            conv3 = conv3.to(device)
            conv4 = conv4.to(device)
            # print('imgs', imgs.size(), 'text', text.size(), 'labels', labels.size())

            # encoder
            Iembed = Iembed_model(imgs, conv3, conv4)
            # Iembed = Iembed_model(imgs)
            Tembed = Tembed_model(text)

            # Imemory, Tmemory = support_memory0(args, device, Iembed_model, Tembed_model)
            Ihash = Ilstm(Imemory.to(device), Iembed.to(device), device)
            Thash = Tlstm(Tmemory.to(device), Tembed.to(device), device)

            # triplet loss
            I_tloss, T_tloss, IT_tloss, TI_tloss = intra_inter_tl(Ihash, Thash, labels, args.margin, device)
            loss = I_tloss + T_tloss + IT_tloss + TI_tloss

            embed_optimizer.zero_grad()
            lstm_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            embed_optimizer.step()
            lstm_optimizer.step()

            epoch_loss += loss.item()
            epoch_IT_loss += IT_tloss.item()
            epoch_TI_loss += TI_tloss.item()
            epoch_I_tloss += I_tloss.item()
            epoch_T_tloss += T_tloss.item()

        # update lr
        embed_scheduler.step()
        lstm_scheduler.step()

        epoch_loss /= len(train_loader)
        epoch_IT_loss /= len(train_loader)
        epoch_TI_loss /= len(train_loader)
        epoch_I_tloss /= len(train_loader)
        epoch_T_tloss /= len(train_loader)
        print(f'[{epoch}/{args.max_epoch}] '
              f'loss:{epoch_loss:0.4f} IT_tloss:{epoch_IT_loss:0.4f} TI_tloss:{epoch_TI_loss:0.4f} '
              f'I_tloss:{epoch_I_tloss:0.4f} T_tloss:{epoch_T_tloss:0.4f}')

        # # tensorboard
        writer.add_scalar('loss', epoch_loss, epoch)
        writer.add_scalar('IT_loss', epoch_IT_loss, epoch)
        writer.add_scalar('TI_loss', epoch_TI_loss, epoch)
        writer.add_scalar('I_tloss', epoch_I_tloss, epoch)
        writer.add_scalar('T_tloss', epoch_T_tloss, epoch)

        # test
        if epoch % 5 == 0:
            Iembed_model.eval()
            Tembed_model.eval()
            Ilstm.eval()
            Tlstm.eval()
            # adaptive.eval()
            # Imemory, Tmemory = support_memory2(args, device, Iembed_model, Tembed_model)
            tst_IB, tst_TB, tst_labels = TSTbinary5(test_loader, Imemory, Tmemory, Iembed_model, Tembed_model, Ilstm, Tlstm,
                                                    device, args)
            db_IB, db_TB, db_labels = TSTbinary5(db_loader, Imemory, Tmemory, Iembed_model, Tembed_model, Ilstm, Tlstm, device,
                                                 args)

            # calculate MAP
            IT_MAP = compute_AP(tst_IB, tst_labels, db_TB, db_labels, device).mean()
            TI_MAP = compute_AP(tst_TB, tst_labels, db_IB, db_labels, device).mean()
            if max_MAP < (IT_MAP + TI_MAP) / 2.:
                max_MAP = (IT_MAP + TI_MAP) / 2.
                max_IT_MAP = IT_MAP
                max_TI_MAP = TI_MAP
                max_epoch = epoch
                torch.save(Iembed_model, os.path.join(Iembed_ckpt,
                                                 f'k{args.k}_m{args.margin}.pth'))
                torch.save(Tembed_model, os.path.join(Tembed_ckpt,
                                                  f'k{args.k}_m{args.margin}.pth'))
                torch.save(Ilstm, os.path.join(Ilstm_ckpt,
                                                      f'k{args.k}_m{args.margin}.pth'))
                torch.save(Tlstm, os.path.join(Tlstm_ckpt,
                                                      f'k{args.k}_m{args.margin}.pth'))

            print(f'[{epoch}/{args.max_epoch}] '
                  f'IT_MAP:{IT_MAP:0.4f}  TI_MAP:{TI_MAP:0.4f}  '
                  f'max_IT_MAP:{max_IT_MAP:0.4f} max_TI_MAP:{max_TI_MAP:0.4f} '      
                  f'max_MAP:{max_MAP:0.4f} max_epoch:{max_epoch}')

        # # update support memory
        if epoch % args.update == 0 and epoch > 0:
            Imemory, Tmemory = support_memory2(args, device, Iembed_model, Tembed_model)

    print(f'finish trained, max_IT_MAP:{max_IT_MAP:0.4f} max_TI_MAP:{max_TI_MAP:0.4f} '
          f'max_seen_IT:{max_seen_IT:04f} max_seen_TI:{max_seen_TI:0.4f} '
          f'max_unseen_IT:{max_unseen_IT:0.4f} max_unseen_TI:{max_unseen_TI:0.4f} '
          f'epoch:{max_epoch}')
    print(args)






