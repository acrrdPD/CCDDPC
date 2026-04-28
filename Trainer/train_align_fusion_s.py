
import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import pathlib
import warnings
import logging.config
import argparse, os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy
import torch.backends.cudnn
import torch.utils.data
import torch.nn.functional
import kornia.losses

from tqdm import tqdm
from dataloader.fuse_data_vsm import FuseTestData_crop_fus
from dataloader.reg_data import ImageTransform_vari
from dataloader.reg_data import Warper2d
from models.BasicEFNet import basiceNet
from models.BasicEFNet import basicfNet
from models.deformable_net import SuperNet_s_unet, SINet
import clip

from loss.fusion_loss import affineLoss_dky_s_not


from functions.affine_transform import AffineTransform
from functions.elastic_transform import ElasticTransform

import torch.nn as nn


import setproctitle
setproctitle.setproctitle('dky_alignNet')

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
    parser.add_argument('--batchsize', default=8, type=int, help='mini-batch size')  # 8
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate') #0.001
    parser.add_argument('--lr_fta', default=0.0001, type=float, help='learning rate') #0.001
    parser.add_argument('--lr_ftf', default=0.0001, type=float, help='learning rate') #0.001
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument('--nEpochs', default=5000, type=int, help='number of total epochs to run')
    parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument('--resume', default='', help='resume checkpoint')
    parser.add_argument('--interval', default=100, help='record interval')
    # checkpoint
    parser.add_argument('--ckpt_e', default='../cache/Extract/DataEn-lr0.001-0.05l1-240712ml_c1/fus_1000.pth', help='checkpoint cache folder')
    parser.add_argument('--ckpt_f', default='../cache/Fusion/DataEn-lr0.001-0.05l1-240712ml_c1/fus_1000.pth', help='checkpoint cache folder')
    parser.add_argument('--ckpt_affine', default='../cache/Affine/DataEn-lr0.001-0.05l1-240903rml_c1-varychange-fus-2field-feat-UNETcyd-224/fus_0800.pth', help='checkpoint cache folder')

    parser.add_argument('--ckpt_f_s', default='../cache/Fusion_s/DataEn-lr0.001-0.05l1-240913rml_c1not', help='checkpoint cache folder')
    parser.add_argument('--ckpt_si', default='../cache/SI/DataEn-lr0.001-0.05l1-240914rml_c1-varychange-fus-2field-feat-UNETnots-224', help='checkpoint cache folder')
    parser.add_argument('--ckpt_affine_s', default='../cache/Affine_s/DataEn-lr0.001-0.05l1-240914rml_c1-varychange-fus-2field-feat-UNETnots-224', help='checkpoint cache folder')

    args = parser.parse_args()
    return args

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

    print("===> Creating Save Path of Checkpoints-si")
    cache_si = pathlib.Path(args.ckpt_si)
    print("===> Creating Save Path of Checkpoints-affine_s")
    cache_affine_s = pathlib.Path(args.ckpt_affine_s)

    print("===> Loading datasets")

    data = FuseTestData_crop_fus(args.ir, args.vis, args.ir_map, args.vis_map)
    training_data_loader = torch.utils.data.DataLoader(data, args.batchsize, True, pin_memory=True)


    print("===> Building models")
    ENet = basiceNet().to(device)
    siNet = SINet().to(device)
    AffNet_s = SuperNet_s_unet().to(device)
    FuseNet = basicfNet().to(device)
    FuseNet_s = basicfNet().to(device)

    print("===> loading trained basice model '{}'".format(args.ckpt_e))
    f_model_state_dict = torch.load(args.ckpt_e)
    ENet.load_state_dict(f_model_state_dict)

    print("===> loading trained affine_t model to student '{}'".format(args.ckpt_affine))
    f_model_state_dict = torch.load(args.ckpt_affine)
    AffNet_s.load_state_dict(f_model_state_dict)

    print("===> loading trained basicf model '{}'".format(args.ckpt_f))
    f_model_state_dict = torch.load(args.ckpt_f)
    FuseNet.load_state_dict(f_model_state_dict)

    print("===> loading trained basicf model to student '{}'".format(args.ckpt_f))
    f_model_state_dict = torch.load(args.ckpt_f)
    FuseNet_s.load_state_dict(f_model_state_dict)

    print("===> loading trained clip model")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    print("===> Defining Loss fuctions")
    criterion_affine_s = affineLoss_dky_s_not().to(device)

    print("===> Building deformation")
    affine = AffineTransform(translate=0.01)
    elastic = ElasticTransform(kernel_size=101, sigma=14)
    image_trans = ImageTransform_vari()
    warp = Warper2d()

    print("===> Setting Optimizers")
    optimizer_si = torch.optim.Adam(siNet.parameters(), lr=args.lr)
    scheduler_si = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_si, T_max=args.nEpochs, eta_min=args.lr * 1e-2)
    optimizer_affine = torch.optim.Adam(AffNet_s.parameters(), lr=args.lr_fta)
    scheduler_affine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_affine, T_max=args.nEpochs, eta_min=args.lr_fta * 1e-2)


    # TODO: optionally copy weights from a checkpoint


    loss_deco = []
    print("===> Starting Training")
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        tqdm_loader = tqdm(training_data_loader, disable=True)
        loss_avg = train(args, tqdm_loader, optimizer_si, scheduler_si, optimizer_affine, scheduler_affine, ENet, siNet, AffNet_s, FuseNet, FuseNet_s, clip_model, criterion_affine_s, epoch, elastic, affine, image_trans, warp)
        loss_deco.append(loss_avg)
        # TODO: save checkpoint
        save_checkpoint(siNet, epoch, cache_si) if epoch % interval == 0 else None
        save_checkpoint(AffNet_s, epoch, cache_affine_s) if epoch % interval == 0 else None

        if epoch % 100 == 0:
            plt.figure("train", (12, 12))
            plt.title("Average Loss")
            x = [(i + 1) for i in range(len(loss_deco))]
            y = loss_deco
            plt.xlabel("Iteration")
            plt.plot(x, y)
            plt.show()


def train(args, tqdm_loader, optimizer_si, scheduler_si, optimizer_affine, scheduler_affine, ENet, siNet, AffNet_s, FuseNet, FuseNet_s,
              clip_model, criterion_affine_s, epoch, elastic, affine, image_trans, warp):

    ENet.eval()
    FuseNet.eval()  # OnlyReg
    FuseNet_s.eval()  # OnlyReg
    siNet.train()
    AffNet_s.train()
    clip_model.eval()
    # TODO: update learning rate of the optimizer
    lr_SI = 0.001
    lr_AF = 0.0001
    print("Epoch={}, lr_SI={}, lr_AF={}".format(epoch, lr_SI, lr_AF))

    loss_deco = []
    for (ir, vis), _, (ir_warp, vis_warp), (ir_map, vi_map), disp in tqdm_loader:

        ir, vis = ir.cuda(), vis.cuda()
        ir_map, vi_map = ir_map.cuda(), vi_map.cuda()
        ir_warp, vis_warp = ir_warp.cuda(), vis_warp.cuda()
        disp = disp.cuda()

        ir_gra = kornia.filters.laplacian(ir, 3)
        ir_warp_gra = kornia.filters.laplacian(ir_warp, 3)
        vis_gra = kornia.filters.laplacian(vis, 3)
        vis_warp_gra = kornia.filters.laplacian(vis_warp, 3)
        target_gra_w2r = vis_gra
        target_gra_r2w = vis_warp_gra

        ir_warp_clip = ir_warp.repeat(1, 3, 1, 1)
        vis_clip = vis.repeat(1, 3, 1, 1)

        ir_clip = ir.repeat(1, 3, 1, 1)
        vis_warp_clip = vis_warp.repeat(1, 3, 1, 1)


        with torch.no_grad():
            ir_feat, vis_feat = ENet(ir, vis)
            ir_warp_feat, vis_warp_feat = ENet(ir_warp, vis_warp)
            fuse_feats_reg, fuse_out_reg, ir_c_mask, vis_c_mask = FuseNet(ir_feat, vis_feat)
            fuse_feats_warp, fuse_out_warp, ir_c_mask, vis_c_mask = FuseNet(ir_warp_feat, vis_warp_feat)

            clip_features_ir_warp = clip_model.encode_image(ir_warp_clip)
            clip_features_vis = clip_model.encode_image(vis_clip)

            clip_features_ir = clip_model.encode_image(ir_clip)
            clip_features_vis_warp = clip_model.encode_image(vis_warp_clip)


        ir_warp_feat_s = ir_warp_feat
        vis_feat_s = vis_feat

        ir_feat_s = ir_feat
        vis_warp_feat_s = vis_warp_feat


        clip_ir_warp_fus, clip_vis_fus = siNet(ir_warp_feat_s, vis_feat_s, clip_features_ir_warp, clip_features_vis)
        clip_ir_fus, clip_vis_warp_fus = siNet(ir_feat_s, vis_warp_feat_s, clip_features_ir, clip_features_vis_warp)
        ir_reg_img_r2w, dis_pre_s_r2w, ir_reg_feat_r2w, ir_reg_img_w2r, dis_pre_s_w2r, ir_reg_feat_w2r = AffNet_s(clip_ir_warp_fus, clip_vis_fus, clip_ir_fus, clip_vis_warp_fus, ir_feat, ir_warp_feat, ir, ir_warp, ir_warp_gra, vis_gra, ir_gra, vis_warp_gra, target_gra_r2w, target_gra_w2r)
        fuse_feats, fuse_out_reg_img, ir_c_mask, vis_c_mask = FuseNet_s(ir_reg_feat_w2r, vis_feat)

        total_loss, regis_loss, feat_loss = criterion_affine_s(clip_ir_warp_fus, clip_vis_fus, fuse_feats_warp,
                                                                     fuse_feats_reg, dis_pre_s_r2w, dis_pre_s_w2r, disp,
                                                                     fuse_out_reg_img, fuse_out_reg, ir, ir_warp, vis,
                                                                     ir_reg_img_r2w, ir_reg_img_w2r, ir_map, vi_map,
                                                                     ir_c_mask,
                                                                     vis_c_mask)

        optimizer_si.zero_grad()
        optimizer_affine.zero_grad()
        nn.utils.clip_grad_norm_(AffNet_s.parameters(), 5)  # TRY
        total_loss.backward()
        optimizer_si.step()
        optimizer_affine.step()

        loss_deco.append(total_loss.item())  # singleReg

    scheduler_si.step()
    scheduler_affine.step()
    loss_avg = numpy.mean(loss_deco)

    print('loss_avg', loss_avg)
    print('regis_loss', regis_loss.item(), 'feat_loss', feat_loss.item())

    return loss_avg

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
    # visdom = visdom.Visdom(port=8097, env='Reg')

    main(args)