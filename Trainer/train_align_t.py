import sys

sys.path.append("..")

import warnings
import logging.config
import argparse, os
import pathlib

import numpy
import torch.backends.cudnn
import torch.utils.data
import torch.nn.functional
import kornia.losses

from tqdm import tqdm
from dataloader.fuse_data_vsm import FuseTestData_crop
from dataloader.reg_data import ImageTransform_vari
from dataloader.reg_data import Warper2d
from models.BasicEFNet import basiceNet
from models.BasicEFNet import basicfNet
from models.deformable_net import SuperNet_t_unet
from loss.fusion_loss import affineLoss_dky

from functions.affine_transform import AffineTransform
from functions.elastic_transform import ElasticTransform

import torch.nn as nn
from models.util import MaskedAutoencoderViT


import setproctitle
setproctitle.setproctitle('dky_alignNet_teacher')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def hyper_args():
    """
    get hyper parameters from args
    """
    parser = argparse.ArgumentParser(description='RobF Net train process')

    # dataset
    parser.add_argument('--ir', default='../dataset/Train/512RS/ir', type=pathlib.Path)  #RoadScene
    parser.add_argument('--vis', default='../dataset/Train/512RS/vi', type=pathlib.Path)
    parser.add_argument('--ir_map', default='../dataset/Train/512RS/ir_map', type=pathlib.Path)
    parser.add_argument('--vis_map', default='../dataset/Train/512RS/vi_map', type=pathlib.Path)

    # train loss weights
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=20.0, type=float)
    parser.add_argument('--theta', default=5.0, type=float)

    # implement details
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")
    parser.add_argument('--dim', default=1, type=int, help='AFuse feather dim')
    parser.add_argument('--batchsize', default=4, type=int, help='mini-batch size')  # 8
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate') #0.001
    parser.add_argument('--lr_ftf', default=0.001, type=float, help='learning rate') #0.001
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument('--nEpochs', default=2000, type=int, help='number of total epochs to run')
    parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument('--resume', default='', help='resume checkpoint')
    parser.add_argument('--interval', default=100, help='record interval')
    # checkpoint
    parser.add_argument('--ckpt_e', default='../cache/Extract/DataEn-lr0.001-0.05l1-240712ml_c1/fus_1000.pth', help='checkpoint cache folder')
    parser.add_argument('--ckpt_f', default='../cache/Fusion/DataEn-lr0.001-0.05l1-240712ml_c1/fus_1000.pth', help='checkpoint cache folder')
    parser.add_argument('--ckpt_affine', default='../cache/Affine/DataEn-lr0.001-0.05l1-240903rml_c1-varychange-fus-2field-feat-UNETcyd-224', help='checkpoint cache folder')

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

    print("===> Creating Save Path of Checkpoints-affine")
    cache_affine = pathlib.Path(args.ckpt_affine)


    print("===> Loading datasets")
    data = FuseTestData_crop(args.ir, args.vis)
    training_data_loader = torch.utils.data.DataLoader(data, args.batchsize, True, pin_memory=True)


    print("===> Building models")
    ENet = basiceNet().to(device)
    AffNet = SuperNet_t_unet().to(device)
    mae = MaskedAutoencoderViT().to(device)
    FuseNet = basicfNet().to(device)

    print("===> loading trained basice_ir model '{}'".format(args.ckpt_e))
    f_model_state_dict = torch.load(args.ckpt_e)
    ENet.load_state_dict(f_model_state_dict)

    print("===> loading trained basicf model '{}'".format(args.ckpt_f))
    f_model_state_dict = torch.load(args.ckpt_f)
    FuseNet.load_state_dict(f_model_state_dict)

    print("===> Defining Loss fuctions")
    criterion_affine = affineLoss_dky().to(device)
    print("===> Building deformation")
    affine = AffineTransform(translate=0.01)
    elastic = ElasticTransform(kernel_size=101, sigma=14)
    image_trans = ImageTransform_vari()
    warp = Warper2d()

    print("===> Setting Optimizers")
    optimizer_affine = torch.optim.Adam(params=AffNet.parameters(), lr=args.lr)

    # TODO: optionally copy weights from a checkpoint

    print("===> Starting Training")
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        tqdm_loader = tqdm(training_data_loader, disable=True)
        train(args, tqdm_loader, optimizer_affine, ENet, AffNet, FuseNet, mae, criterion_affine, epoch, elastic, affine, image_trans, warp)

        # TODO: save checkpoint
        save_checkpoint(AffNet, epoch, cache_affine) if epoch % interval == 0 else None

def train(args, tqdm_loader, optimizer_deco, ENet, AffNet, FuseNet, mae, criterion_affine, epoch, elastic, affine, image_trans, warp):

    ENet.eval()
    FuseNet.eval()  # OnlyReg
    AffNet.train()
    # TODO: update learning rate of the optimizer
    lr_R = 0.001
    print("Epoch={}, lr_R={}".format(epoch, lr_R))

    loss_deco = []
    for (ir, vis), _, (ir_warp, vis_warp), disp in tqdm_loader:

        ir, vis = ir.cuda(), vis.cuda()
        ir_warp, vis_warp = ir_warp.cuda(), vis_warp.cuda()
        disp = disp.cuda()

        with torch.no_grad():
            ir_feat, vis_feat = ENet(ir, vis)
            ir_warp_feat, vis_warp_feat = ENet(ir_warp, vis_warp)
            fuse_feats_reg, fuse_out_reg, ir_c_mask, vis_c_mask = FuseNet(ir_feat, vis_feat)
            fuse_feats_warp, fuse_out_warp, ir_c_mask, vis_c_mask = FuseNet(ir_warp_feat, vis_warp_feat)
            fuse_out_reg_gra = kornia.filters.laplacian(fuse_out_reg, 3)
            fuse_out_warp_gra = kornia.filters.laplacian(fuse_out_warp, 3)
            target_gra_w2r = fuse_out_reg_gra
            target_gra_r2w = fuse_out_warp_gra

        fus_reg_img_r2w, disp_pre_r2w, fus_reg_feat_r2w, fus_reg_img_w2r, disp_pre_w2r, fus_reg_feat_w2r = AffNet(fuse_feats_reg, fuse_feats_warp, fuse_out_warp, fuse_out_reg, fuse_out_warp_gra, fuse_out_reg_gra, target_gra_r2w, target_gra_w2r)

        total_loss, ncc, grad = criterion_affine(fus_reg_img_r2w, fus_reg_img_w2r, fuse_out_warp, fuse_out_reg, disp_pre_r2w, disp_pre_w2r,
                                             disp)


        optimizer_deco.zero_grad()
        nn.utils.clip_grad_norm_(AffNet.parameters(), 5) #TRY
        total_loss.backward()
        optimizer_deco.step()

        loss_deco.append(total_loss.item())  # singleReg

    loss_avg = numpy.mean(loss_deco)
    with open("./train_loss_fus529.txt", "a") as train_los:
        train_los.write('/n' + str(loss_avg))
    print('loss_avg', loss_avg)


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


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