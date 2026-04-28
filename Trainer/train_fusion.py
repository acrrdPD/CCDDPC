import sys

sys.path.append("..")

import pathlib
import warnings
import logging.config
import argparse, os

import numpy
import torch.backends.cudnn
import torch.utils.data
import torch.nn.functional

from tqdm import tqdm
from dataloader.fuse_data_vsm import FuseDataVSM
from models.BasicEFNet import basiceNet
from models.BasicEFNet import basicfNet
from loss.fusion_loss import FusionLoss_main

import setproctitle
setproctitle.setproctitle('dky_basicEFNet')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def hyper_args():
    """
    get hyper parameters from args
    """
    parser = argparse.ArgumentParser(description='RobF Net train process')

    # dataset
    parser.add_argument('--ir', default='../dataset/Train/512RS/ir', type=pathlib.Path)
    parser.add_argument('--vis', default='../dataset/Train/512RS/vis', type=pathlib.Path)
    parser.add_argument('--ir_map', default='../dataset/Train/512RS/ir_map', type=pathlib.Path)
    parser.add_argument('--vis_map', default='../dataset/Train/512RS/vis_map', type=pathlib.Path)
    # train loss weights
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=20.0, type=float)
    parser.add_argument('--theta', default=5.0, type=float)
    parser.add_argument('--dim', default=1, type=int, help='AFuse feather dim')
    parser.add_argument('--batchsize', default=8, type=int, help='mini-batch size')  # 32
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument('--nEpochs', default=1000, type=int, help='number of total epochs to run')
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")
    parser.add_argument("--step", type=int, default=1000, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument('--resume', default='', help='resume checkpoint')
    parser.add_argument('--interval', default=20, help='record interval')
    # checkpoint
    parser.add_argument("--load_model_extract", default=None, help="path to pretrained model (default: none)")
    parser.add_argument("--load_model_fuse", default=None, help="path to pretrained model (default: none)")
    parser.add_argument('--ckpt_e', default='../cache/Extract/DataEn-lr0.001-0.05l1-240712ml_c1', help='checkpoint cache folder')
    parser.add_argument('--ckpt_f', default='../cache/Fusion/DataEn-lr0.001-0.05l1-240712ml_c1', help='checkpoint cache folder')

    args = parser.parse_args()
    return args

# def main(args, visdom):
def main(args):

    cuda = args.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    log = logging.getLogger()

    epoch = args.nEpochs
    interval = args.interval

    print("===> Creating Save Path of Checkpoints")
    cache_e = pathlib.Path(args.ckpt_e)
    cache_f = pathlib.Path(args.ckpt_f)

    print("===> Loading datasets")
    data = FuseDataVSM(args.ir, args.vis, args.ir_map, args.vis_map)
    training_data_loader = torch.utils.data.DataLoader(data, args.batchsize, True, pin_memory=True)


    print("===> Building models")
    ENet = basiceNet().to(device)
    FuseNet = basicfNet().to(device)

    print("===> Defining Loss fuctions")
    criterion_fus = FusionLoss_main().to(device)

    print("===> Setting Optimizers")
    optimizer_fus = torch.optim.Adam(params=FuseNet.parameters(), lr=args.lr)


    # TODO: optionally copy weights from a checkpoint
    if args.load_model_extract is not None:
        print('Loading pre-trained ENet checkpoint %s' % args.load_model_extract)
        log.info(f'Loading pre-trained checkpoint {str(args.load_model_extract)}')
        state = torch.load(str(args.load_model_extract))
        ENet.load_state_dict(state['net'])
    else:
        print("=> no model found at '{}'".format(args.load_model_extract))

    if args.load_model_fuse is not None:
        print('Loading pre-trained FuseNet checkpoint %s' % args.load_model_fuse)
        log.info(f'Loading pre-trained checkpoint {str(args.load_model_fuse)}')
        state = torch.load(str(args.load_model_fuse))
        FuseNet.load_state_dict(state['net'])
    else:
        print("=> no model found at '{}'".format(args.load_model_fuse))

    print("===> Starting Training")
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        tqdm_loader = tqdm(training_data_loader, disable=True)
        train(args, tqdm_loader, optimizer_fus, ENet, FuseNet, criterion_fus, epoch)

        # TODO: save checkpoint
        save_checkpoint(ENet, epoch, cache_e) if epoch % interval == 0 else None
        save_checkpoint(FuseNet, epoch, cache_f) if epoch % interval == 0 else None

def train(args, tqdm_loader, optimizer_fus, ENet, FuseNet, criterion_fus, epoch):

    ENet.train()
    FuseNet.train()
    # TODO: update learning rate of the optimizer
    lr_F = adjust_learning_rate(args, optimizer_fus, epoch - 1)
    print("Epoch={}, lr_F={} ".format(epoch, lr_F))

    loss_basicefus = []
    for (ir, vis), _, (ir_map, vi_map) in tqdm_loader:

        ir, vis = ir.cuda(), vis.cuda()
        ir_map, vi_map = ir_map.cuda(), vi_map.cuda()
        ir = ir.repeat(1, 3, 1, 1)
        vis = vis.repeat(1, 3, 1, 1)
        ir_feat, vis_feat = ENet(ir, vis)
        fuse_feats, fuse_out, ir_c_mask, vis_c_mask = FuseNet(ir_feat, vis_feat)

        loss = criterion_fus(fuse_out, ir, vis, ir_map, vi_map, ir_c_mask, vis_c_mask)
        optimizer_fus.zero_grad()
        loss.backward()
        optimizer_fus.step()

        loss_basicefus.append(loss.item())
    loss_avg = numpy.mean(loss_basicefus)
    print('loss_avg', loss_avg)


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def save_checkpoint(net, epoch, cache):
    model_folder = cache
    model_out_path = str(model_folder / f'fus_{epoch:04d}.pth')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(net.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = hyper_args()

    main(args)