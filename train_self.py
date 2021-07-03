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

    # print('model:', model)
    # print('pre dicts:', pretrained_dict)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train image and text network with triplet loss')

    parser.add_argument('--dataset', type=str, default='wikipedia_dataset', help='the name of dataset')
    parser.add_argument('--cls', type=int, default=10, help='the number of categories of dataset')

    parser.add_argument('--load', type=int, default=0, help='if load trained model before')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--hash_length', type=int, default=16, help='length of hashing binary')
    parser.add_argument('--margin', type=int, default=8, help='loss_type')
    parser.add_argument('--optim', type=str, default='adam', help='choose optimizer')
    parser.add_argument('--few_shot', type=int, default=1, help='whether train with few-shot')
    parser.add_argument('--k', type=int, default=1, help='k-shot')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of scheduler')
    parser.add_argument('--nl_version', type=str, default='None', help='version of non-local network')

    parser.add_argument('--max_epoch', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--step_size', type=int, default=250, help='the step size to change lr')

    parser.add_argument('--gpu', type=str, default='cuda:0', help='which GPU to use')
    parser.add_argument('--seed', type=int, default=100, help='random seed')

    args = parser.parse_args()
    print(args)        

    torch.manual_seed(args.seed)
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

    # tensorboard
    writer = SummaryWriter(f'{args.nl_version}_{args.dataset}_l{args.hash_length}m{args.margin}_k{args.k}')

    # load data
    if args.few_shot == 1:
        tr_data_root = f'/home/disk1/wangshy/data/{args.dataset}/few_shot/{args.k}shot/ITL.mat'
    else:
        tr_data_root = f'/home/disk1/wangshy/data/{args.dataset}/ITL.mat'
    data_root = f'/home/disk1/wangshy/data/{args.dataset}/ITL.mat'
    train_loader = load_data(tr_data_root, mode='train', batch_size=args.batch_size, shuffle=True)
    test_loader = load_data(data_root, mode='test', batch_size=args.batch_size, shuffle=False)
    db_loader = load_data(data_root, mode='db', batch_size=args.batch_size, shuffle=False)
    print('train_loader:', len(train_loader), 'test_loader:', len(test_loader))

    # path to save trained model
    ckpt_dir = f'/home/disk1/wangshy/FSCM/logs/{args.dataset}/pyramid/'
    if args.few_shot == 0 and args.nl_version != 'None':
        I_ckpt_dir = ckpt_dir + f'/IT_NonLocal_{args.nl_version}/I_model/{args.hash_length}bits/'
        T_ckpt_dir = ckpt_dir + f'/IT_NonLocal_{args.nl_version}/T_model/{args.hash_length}bits/'
    elif args.few_shot == 0 and args.nl_version == 'None':
        I_ckpt_dir = ckpt_dir + f'/IT_vggbert/I_model/{args.hash_length}bits/'
        T_ckpt_dir = ckpt_dir + f'/IT_vggbert/T_model/{args.hash_length}bits/'
    elif args.few_shot and args.nl_version != 'None':
        I_ckpt_dir = ckpt_dir + f'/fewshot_NL/I_model/{args.hash_length}bits/'
        T_ckpt_dir = ckpt_dir + f'/fewshot_NL/T_model/{args.hash_length}bits/'
    else:
        I_ckpt_dir = ckpt_dir + f'/IT_fewshot/I_model/{args.hash_length}bits/'
        T_ckpt_dir = ckpt_dir + f'/IT_fewshot/T_model/{args.hash_length}bits/'
    if not os.path.exists(I_ckpt_dir):
        os.makedirs(I_ckpt_dir)
        os.makedirs(T_ckpt_dir)

    ## build models: pretrained model with triplet loss
    I_model = IPyramid(256, 512, 4096, args.hash_length, args.nl_version)  ##
    # I_model = IPyramid(4096, args.hash_length, args.nl_version)
    I_model.to(device)
    if args.dataset == 'mir-25k':
        T_model = TPyramid(1386, args.hash_length, args.nl_version) # mir
    else:
        T_model = TPyramid(768, args.hash_length, args.nl_version) # mir
    T_model.to(device)

    max_MAP = max_IT_MAP = max_TI_MAP = max_seen_IT = max_seen_TI = max_unseen_IT = max_unseen_TI = 0.0
    # # initialize parameters with models trained triplet loss separately I->I T->T
    if args.load:
        print('loading pretrained model....')
        load_state(I_model, I_ckpt_dir + f'm{args.margin}_k{args.k}.pth')
        load_state(T_model, T_ckpt_dir + f'm{args.margin}_k{args.k}.pth')
        # calculate original mAP
        I_model.eval()
        T_model.eval()
        tst_IB, tst_TB, tst_labels = IT2binary2(test_loader, I_model, T_model, device)
        db_IB, db_TB, db_labels = IT2binary2(db_loader, I_model, T_model, device)
        IT_MAP = compute_AP(tst_IB, tst_labels, db_TB, db_labels, device).mean()
        TI_MAP = compute_AP(tst_TB, tst_labels, db_IB, db_labels, device).mean()
        max_IT_MAP, max_TI_MAP, max_MAP = IT_MAP, TI_MAP, (IT_MAP + TI_MAP) / 2.
        print(f'original IT_MAP:{IT_MAP:0.4f} TI_MAP:{TI_MAP:0.4f} mean:{max_MAP:0.4f}')

    # optimize
    Iparams = list(I_model.parameters())
    Tparams = list(T_model.parameters())
    params = Iparams + Tparams
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5, last_epoch=-1)

    # train
    max_epoch = 0     
    print('start to train the image model and text model......')
    for epoch in range(args.max_epoch):
        I_model.train()
        T_model.train()
        # scheduler.step()
        epoch_loss = epoch_inter_tloss = epoch_I_tloss = epoch_T_tloss = epoch_I_closs = epoch_T_closs = epoch_IT_loss = epoch_TI_loss = 0.
        for batch_idx, (imgs, conv3, conv4, text, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            text = text.to(device)
            conv3 = conv3.to(device)
            conv4 = conv4.to(device)
            # print('imgs', imgs.size(), 'text', text.size(), 'labels', labels.size())

            # encoder
            Ihash = I_model(imgs, conv3, conv4) ##
            Thash= T_model(text)

            # triplet loss
            I_tloss, T_tloss, IT_tloss, TI_tloss = intra_inter_tl(Ihash, Thash, labels, args.margin, device)
            loss = IT_tloss + TI_tloss + I_tloss + T_tloss

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_IT_loss += IT_tloss.item()
            epoch_TI_loss += TI_tloss.item()
            epoch_I_tloss += I_tloss.item()
            epoch_T_tloss += T_tloss.item()

        scheduler.step()
        epoch_loss /= len(train_loader)
        epoch_IT_loss /= len(train_loader)
        epoch_TI_loss /= len(train_loader)
        epoch_I_tloss /= len(train_loader)
        epoch_T_tloss /= len(train_loader)
        epoch_I_closs /= len(train_loader)
        epoch_T_closs /= len(train_loader)
        print(f'[{epoch}/{args.max_epoch}] '  
              f'loss:{epoch_loss:0.4f} IT_tloss:{epoch_IT_loss:0.4f} TI_tloss:{epoch_TI_loss:0.4f} '
              f'I_tloss:{epoch_I_tloss:0.4f} T_tloss:{epoch_T_tloss:0.4f} ')

        # tensorboard
        writer.add_scalar('loss', epoch_loss, epoch)
        writer.add_scalar('IT_loss', epoch_IT_loss, epoch)
        writer.add_scalar('TI_loss', epoch_TI_loss, epoch)
        writer.add_scalar('I_tloss', epoch_I_tloss, epoch)
        writer.add_scalar('T_tloss', epoch_T_tloss, epoch)

        # test
        # generate binary codes
        if epoch % 5 == 0:
            I_model.eval()
            T_model.eval()
            tst_IB, tst_TB, tst_labels = IT2binary2(test_loader, I_model, T_model, device)
            db_IB, db_TB, db_labels = IT2binary2(db_loader, I_model, T_model, device)
            if args.dataset == 'mir-25k':
                IT_MAP = compute_NDCG(tst_IB, tst_labels, db_TB, db_labels)
                TI_MAP = compute_NDCG(tst_TB, tst_labels, db_IB, db_labels)
            else:
                IT_MAP = compute_AP(tst_IB, tst_labels, db_TB, db_labels, device).mean()
                TI_MAP = compute_AP(tst_TB, tst_labels, db_IB, db_labels, device).mean()

            if max_MAP < (IT_MAP + TI_MAP) / 2.:
                max_MAP = (IT_MAP + TI_MAP) / 2.
                max_IT_MAP = IT_MAP
                max_TI_MAP = TI_MAP
                max_epoch = epoch
                torch.save(I_model, os.path.join(I_ckpt_dir,
                                                 f'1m{args.margin}_k{args.k}.pth'))
                torch.save(T_model, os.path.join(T_ckpt_dir,
                                                 f'1m{args.margin}_k{args.k}.pth'))
            print(f'[{epoch}/{args.max_epoch}] '                                      
                  f'IT_MAP:{IT_MAP:0.4f}  TI_MAP:{TI_MAP:0.4f}  '        
                  f'max_IT_MAP:{max_IT_MAP:0.4f} max_TI_MAP:{max_TI_MAP:0.4f} '                 
                  f'max_MAP:{max_MAP:0.4f} max_epoch:{max_epoch}')

    print(f'finish trained, max_IT_MAP:{max_IT_MAP:0.4f} max_TI_MAP:{max_TI_MAP:0.4f} '  
          f'epoch:{max_epoch}')
    print(args)







