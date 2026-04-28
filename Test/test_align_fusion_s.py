import sys

sys.path.append("..")

import argparse
import pathlib
import warnings
import statistics
import time

import os
import cv2
import numpy as np
import kornia
import torch.backends.cudnn
import torch.cuda
import torch.utils.data
import torchvision
from torch import Tensor
from tqdm import tqdm

from dataloader.extract_data_vsm import ExtractTestData, ExtractTestData_same
from dataloader.extract_data_vsm import ExtractTestData_SF
from dataloader.reg_data import ImageTransform_1
from dataloader.reg_data import Warper2d
# from dataloader.extract_data_vsm import ExtractTestData_SF0
# from dataloader.fuse_data_vsm import FuseTestData
from models.BasicEFNet import basiceNet, basiceNet_re, basiceNet_re_512, basiceNet_512
from models.BasicEFNet import basicfNet, basicfNet_s
# from models.DecoNet import decoNet
# from models.DecoNet import decoNet_dky
from models.restormer_arch import decoNet_restormer
# from models.AffineNet import Affine_Model_warp
# from models.deformable_net import DeformableNet
from models.deformable_net import SuperNet_t, SuperNet_re, SuperNet_s_unet, SINet

from functions.affine_transform import AffineTransform
from functions.elastic_transform import ElasticTransform

from models.util import MaskedAutoencoderViT
import torchvision.transforms as transforms

import clip
import torch.nn.functional as F

from thop import profile


def hyper_args():
    """
    get hyper parameters from args
    """
    parser = argparse.ArgumentParser(description='Fuse Net eval process')
    # dataset
    # parser.add_argument('--ir',      default='../dataset/raw/ctest/Road/ir_reg', type=pathlib.Path)
    # parser.add_argument('--vi',      default='../dataset/raw/ctest/Road/vi', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/ir2vis/RoadScene/ir', type=pathlib.Path) #RoadScene
    # parser.add_argument('--ir_reverse', default='../dataset/ir2vis/RoadScene/ir_reverse', type=pathlib.Path)
    # parser.add_argument('--vis_warp', default='../dataset/ir2vis/RoadScene/vis_warp', type=pathlib.Path)
    # parser.add_argument('--vis_warp_reverse', default='../dataset/ir2vis/RoadScene/vis_warp_reverse', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/TNO/ir', type=pathlib.Path) #TNO
    # parser.add_argument('--vis', default='../dataset/Test/TNO/vis', type=pathlib.Path)
    # parser.add_argument('--ir', default='/home/l/data2/dky/UnregisMask/dataset/Train/RoadScene_224/ir', type=pathlib.Path)  # RoadScene 1111111111111111111111111
    od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='/home/l/data2/dky/UnregisMask/dataset/Train/RoadScene_224/vis', type=pathlib.Path)
    # parser.add_argument('--ir', default='/home/l/data2/dky/UnregisMask/dataset/Train/RoadScene_224/ir',
    #                     type=pathlib.Path)  # RoadScene_same dataset
    # parser.add_argument('--ir_warp', default='/home/l/data2/dky/UnregisMask/dataset/Test/224RS64/ir_warp',
    #                     type=pathlib.Path)  # RoadScene_same dataset
    # parser.add_argument('--vis_warp', default='/home/l/data2/dky/UnregisMask/dataset/Test/224RS64/vis_warp',
    #                     type=pathlib.Path)  # RoadScene_same dataset
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='/home/l/data2/dky/UnregisMask/dataset/Test/224RS64/vis', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/224llvip/ir', type=pathlib.Path)  # LLVIP
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Test/224llvip/vi', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/224TNO-lr2/ir', type=pathlib.Path)  # TNO_same dataset
    # parser.add_argument('--ir_warp', default='../dataset/Test/224TNO-lr2/ir_warp', type=pathlib.Path)  # TNO_same dataset
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Test/224TNO-lr2/vis', type=pathlib.Path)
    # parser.add_argument('--vis_warp', default='../dataset/Test/224TNO-lr2/vis', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/224TNO0511/ir_warp', type=pathlib.Path)  # TNO_same dataset2
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Test/224TNO0511/vis', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/224TNO/ir', type=pathlib.Path)  # 224TNO
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Test/224TNO/vi', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/TNO/ir', type=pathlib.Path)  # TNO original size
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Test/TNO/vis', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/512TNO/ir_warp', type=pathlib.Path)  # 512TNO original size
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Test/512TNO/vi', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Train/RoadScene_ori/ir', type=pathlib.Path)  # RoadScene original size
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Train/RoadScene_ori/vis', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/512RS/ir_warp', type=pathlib.Path)  # RoadScene 512
    # parser.add_argument('--vis', default='../dataset/Test/512RS/vi', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/512CVC-Day/ir', type=pathlib.Path)  # real(day
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Test/512CVC-Day/vis', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/CVC-Night/ir', type=pathlib.Path)  # real(night
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Test/CVC-Night/vis', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/512RS_revise/ir_warp', type=pathlib.Path)  # real(night
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Test/512RS_revise/vis', type=pathlib.Path)
    parser.add_argument('--ir', default='../dataset/Test/512RS/ir', type=pathlib.Path)  # real(night
    od = 1 + 16 + 16 + 2
    parser.add_argument('--vis', default='../dataset/Test/512RS/vi', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/224M3FD/ir', type=pathlib.Path)  # M3FD
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Test/224M3FD/vi', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/224VOT/ir', type=pathlib.Path)  # 224VOT
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Test/224VOT/vi', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/224CVC-14/Day/ir', type=pathlib.Path)  # real
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Test/224CVC-14/Day/vi', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/512CVC-Day/ir', type=pathlib.Path)  # 512real-day
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Test/512CVC-Day/vis', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/224CVC/IR', type=pathlib.Path)  # real-road(day
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Test/224CVC/VIS', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/224CVC-14/Day/ir', type=pathlib.Path)  # real(day
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Test/224CVC-14/Day/vi', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/224CVC-14/Night/ir', type=pathlib.Path)  # real(night
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Test/224CVC-14/Night/vi', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/CVC-Night/ir', type=pathlib.Path)  # 512real(night
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Test/CVC-Night/vis', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/test/224CVC-Day/ir', type=pathlib.Path)  # real(day
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/test/224CVC-Day/vi', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/Test/224CVC-person/ir', type=pathlib.Path)  # real-person(day
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='../dataset/Test/224CVC-person/vi', type=pathlib.Path)
    # parser.add_argument('--ir', default='/home/l/data2/dky/UnregisMask/dataset/Test/224LXS/ir', type=pathlib.Path)  # real(lxs
    # od = 1 + 16 + 16 + 2
    # parser.add_argument('--vis', default='/home/l/data2/dky/UnregisMask/dataset/Test/224LXS/vi', type=pathlib.Path)
    # parser.add_argument('--vis_warp', default='../dataset/ir2vis/TNO/vis', type=pathlib.Path)
    # parser.add_argument('--vis_warp_reverse', default='../dataset/ir2vis/TNO/vis_reverse', type=pathlib.Path)
    # checkpoint
    parser.add_argument('--ckpt_e', default='../cache/Extract/DataEn-lr0.001-0.05l1-240712ml_c1/fus_1000.pth',
                        help='checkpoint cache folder')
    parser.add_argument('--ckpt_f', default='../cache/Fusion/DataEn-lr0.001-0.05l1-240712ml_c1/fus_1000.pth', help='checkpoint cache folder')
    # parser.add_argument('--ckpt_f_s', default='../cache/Fusion_s/DataEn-lr0.001-0.05l1-240618not/fus_0300.pth', help='checkpoint cache folder')
    parser.add_argument('--ckpt_f_s', default='../cache/Fusion/DataEn-lr0.001-0.05l1-240712ml_c1/fus_1000.pth', help='checkpoint cache folder')
    # parser.add_argument('--ckpt_f_s', default='../cache/Fusion_s/DataEn-lr0.001-0.05l1-240822rml_c1not/fus_1000.pth', help='checkpoint cache folder')
    # parser.add_argument('--ckpt_f_s', default='../cache/Fusion_s/DataEn-lr0.001-0.05l1-240727ml_c1not/fus_0500.pth', help='checkpoint cache folder')
    # parser.add_argument('--ckpt_si', default='../cache/SI/DataEn-lr0.001-0.05l1-240822rml_c1-varychange-fus-2field-feat-UNETnots-224/fus_1000.pth', help='checkpoint cache folder')
    parser.add_argument('--ckpt_si', default='../cache/SI/DataEn-lr0.001-0.05l1-240914rml_c1-varychange-fus-2field-feat-UNETnots-224/fus_1000.pth', help='checkpoint cache folder')
    # parser.add_argument('--ckpt_si', default='../cache/SI/DataEn-lr0.001-0.05l1-240902rml_c1-varychange-fus-2field-feat-UNETnots-224/fus_1000.pth', help='checkpoint cache folder')
    # parser.add_argument('--ckpt_affine_s', default='../cache/Affine_s/DataEn-lr0.001-0.05l1-240618-varychange-fus-2field-feat-UNETnots-224/fus_1000.pth', help='checkpoint cache folder')
    # parser.add_argument('--ckpt_affine_s', default='../cache/Affine_s/DataEn-lr0.001-0.05l1-240822rml_c1-varychange-fus-2field-feat-UNETnots-224/fus_1000.pth', help='checkpoint cache folder')
    parser.add_argument('--ckpt_affine_s', default='../cache/Affine_s/DataEn-lr0.001-0.05l1-240914rml_c1-varychange-fus-2field-feat-UNETnots-224/fus_1000.pth', help='checkpoint cache folder')
    # parser.add_argument('--ckpt_affine_s', default='../cache/Affine/DataEn-lr0.001-0.05l1-240903rml_c1-varychange-fus-2field-feat-UNETcyd-224/fus_0800.pth', help='checkpoint cache folder')
    # parser.add_argument('--ckpt_ftf', default='../cache/ftFusion/DataEn-lr0.001-0.05l1-240514-2field/fus_1000.pth', help='checkpoint cache folder')  # fine-tuning fusion
    # parser.add_argument('--dst', default='/home/l/data2/dky/UnregisMask/results_final/240418-RoadScene/SuperF', help='fuse image save folder', type=pathlib.Path)
    # parser.add_argument('--dst', default='../results_final/240421_0_image_lr_loss-TNO-1000/SuperF', help='fuse image save folder', type=pathlib.Path)
    # parser.add_argument('--dst', default='../results_final/240429_final_loss-TNO-1000/SuperF', help='fuse image save folder', type=pathlib.Path)
    # parser.add_argument('--dst', default='../results_final/240430-feat-a0.2t0.01-224-RoadScene-2000/SuperF', help='fuse image save folder', type=pathlib.Path)
    # parser.add_argument('--dst', default='../results_align_student/92time240822rml_c1not-varychange-fus-2field-feat-UNETnots-TNOsame512-1000/SuperF', help='fuse image save folder', type=pathlib.Path)
    # parser.add_argument('--dst', default='../results_align_student/240914rml_c1not-varychange-fus-2field-feat-UNETnots-CVCNightsame512-1000/SuperF', help='fuse image save folder', type=pathlib.Path)
    # parser.add_argument('--dst', default='../results_align_student/240914rml_c1not-varychange-fus-2field-feat-UNETnots-RS_revise512-1000/SuperF', help='fuse image save folder', type=pathlib.Path)
    parser.add_argument('--dst', default='../results_align_student/240914rml_c1not-varychange-fus-2field-feat-UNETnots-RS_revise256256speed-1000/SuperF', help='fuse image save folder', type=pathlib.Path)


    # parser.add_argument('--dim', default=64, type=int, help='AFuse feather dim')
    parser.add_argument('--dim', default=1, type=int, help='AFuse feather dim')
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")

    args = parser.parse_args()
    return args

def main(args):

    cuda = args.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    print("===> Loading datasets")
    # data = ExtractTestData_SF0(args.ir, args.vis)
    # data = ExtractTestData_SF(args.ir, args.vis)
    data = ExtractTestData(args.ir, args.vis)  # using
    # data = ExtractTestData_same(args.ir, args.vis, args.ir_warp, args.vis_warp)  # same dataset
    # data = ExtractTestData_norever(args.ir, args.vis_warp)
    test_data_loader = torch.utils.data.DataLoader(data, 1, True, pin_memory=True)

    print("===> Building model")
    ENet = basiceNet().to(device)
    # ENet_s = basiceNet_512().to(device)
    # AffNet_s = SuperNet_re().to(device)
    siNet = SINet().to(device)
    AffNet_s = SuperNet_s_unet().to(device)
    # DecoNet_vis = DecoNet().to(device)
    mae = MaskedAutoencoderViT().to(device)
    # ReconNet_vis = ReconNet().to(device)
    FuseNet = basicfNet().to(device)
    # FuseNet_s = basicfNet_s().to(device)
    FuseNet_s = basicfNet().to(device)

    print("===> loading trained basice model '{}'".format(args.ckpt_e))
    f_model_state_dict = torch.load(args.ckpt_e)
    ENet.load_state_dict(f_model_state_dict)

    # print("===> loading trained basice_vis_s model '{}'".format(args.ckpt_e_s))
    # f_model_state_dict = torch.load(args.ckpt_e_s)
    # ENet_s.load_state_dict(f_model_state_dict)

    print("===> loading trained si model '{}'".format(args.ckpt_si))
    f_model_state_dict = torch.load(args.ckpt_si)
    siNet.load_state_dict(f_model_state_dict)

    print("===> loading trained affine_s model '{}'".format(args.ckpt_affine_s))
    f_model_state_dict = torch.load(args.ckpt_affine_s)
    AffNet_s.load_state_dict(f_model_state_dict)

    print("===> loading trained basicf_s model '{}'".format(args.ckpt_f_s))  # original fusion
    f_model_state_dict = torch.load(args.ckpt_f_s)
    FuseNet_s.load_state_dict(f_model_state_dict)

    # print("===> loading trained basicf model '{}'".format(args.ckpt_f))  # original fusion
    # f_model_state_dict = torch.load(args.ckpt_f)
    # FuseNet.load_state_dict(f_model_state_dict)

    print("===> loading trained clip model")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    # print("===> loading trained fine-tuning fusion model '{}'".format(args.ckpt_ftf))  # fine-tuning fusion
    # f_model_state_dict = torch.load(args.ckpt_ftf)
    # basicf_net.load_state_dict(f_model_state_dict)

    print("===> Building deformation")
    # affine = AffineTransform(translate=0.01)
    # elastic = ElasticTransform(kernel_size=101, sigma=16)
    # elastic = ElasticTransform(kernel_size=101, sigma=14)
    image_trans = ImageTransform_1()
    warp = Warper2d()

    print("===> Starting Testing")
    # test(ENet, ENet_s, AffNet_s, mae, clip_model, elastic, affine, image_trans, warp, FuseNet_s, FuseNet, test_data_loader, args.dst, device)
    # test(ENet, ENet_s_ir, ENet_s_vis, AffNet_s, mae, clip_model, elastic, affine, image_trans, warp, FuseNet_s, FuseNet, test_data_loader, args.dst, device)
    # test(ENet, ENet_s_ir, ENet_s_vis, AffNet_s, clip_model, elastic, affine, image_trans, warp, FuseNet_s, FuseNet, test_data_loader, args.dst, device)
    test(ENet, siNet, AffNet_s, clip_model, image_trans, warp, FuseNet_s, FuseNet, test_data_loader, args.dst, device)

def freeze(net):
    for param in net.named_parameters():
        param[1].requires_grad = False
    net.eval()


# def test(ENet, ENet_s, AffNet, mae, clip_model, elastic, affine, image_trans, warp, FuseNet, FuseNet_t, test_data_loader, dst, device):
# def test(ENet, ENet_s_ir, ENet_s_vis, AffNet, clip_model, elastic, affine, image_trans, warp, FuseNet, FuseNet_t, test_data_loader, dst, device):
def test(ENet, siNet, AffNet, clip_model, image_trans, warp, FuseNet, FuseNet_t, test_data_loader, dst, device):

    # ENet_ir.eval()
    # ENet_vis.eval()
    # ENet_ir_s.eval()
    # ENet_vis_s.eval()
    # AffNet.eval()
    # FuseNet.eval()
    # clip_model.eval()
    freeze(ENet)
    # freeze(ENet_s)
    freeze(siNet)
    freeze(AffNet)
    freeze(FuseNet)
    freeze(clip_model)
    freeze(FuseNet_t)


    fus_time = []
    tqdm_loader = tqdm(test_data_loader, disable=True)
    # for (ir, vis), (ir_path, vi_path) in tqdm_loader:
    for (ir, vis), (ir_path, vis_path) in tqdm_loader:  # using
    # for (ir, vis), (ir_path, vis_path), (ir_warp, vis_warp), (ir_warp_path, vis_warp_path) in tqdm_loader:
    # for (ir, vis, ir_warp, vis_warp, disp), (ir_path, vis_path) in tqdm_loader:

        name, ext = os.path.splitext(os.path.basename(ir_path[0]))
        file_name = name + ext

        ir, vis = ir.cuda(), vis.cuda()
        ir_warp, vis = ir.cuda(), vis.cuda()

        resize_dky = transforms.Resize([1024, 1024])
        ir_warp = resize_dky(ir_warp)
        vis = resize_dky(vis)

        # ir_warp, vis_warp = ir_warp.cuda(), vis_warp.cuda()
        # disp = disp.cuda()
        #
        # if len(ir.shape) > 4:
        #     ir = ir.squeeze(1)
        #     vis = vis.squeeze(1)
        #     ir_warp = ir_warp.squeeze(1)
        #     vis_warp = vis_warp.squeeze(1)
        #     disp = disp.squeeze(1)

        # iw, disp = image_trans(ir)
        # disp = disp.permute(0, 3, 1, 2) * 224
        # ir_warp = warp(disp, ir)

        # iw, disp = image_trans(ir)
        # disp = disp.permute(0, 3, 1, 2) * 512
        # ir_warp = warp(disp, ir)
        # vis_warp = warp(disp, vis)

        # H, W = ir.shape[2], ir.shape[3]
        #
        # # resize = transforms.Resize([224, 224])
        # resize = transforms.Resize([512, 512])
        # resize_ori = transforms.Resize([H, W])
        #
        # ir_re = resize(ir)
        # vis_re = resize(vis)
        #
        # iw, disp = image_trans(ir_re)
        # # disp = disp.permute(0, 3, 1, 2) * 224
        # disp = disp.permute(0, 3, 1, 2) * 512
        # ir_warp = warp(disp, ir_re)
        # ir_warp_re = resize_ori(ir_warp)
        # # vis_warp = warp(disp, vis_re)

        ir_re = ir_warp
        # ir_re = ir
        vis_re = vis

        ir_warp_gra = kornia.filters.laplacian(ir_warp, 3)
        vis_gra = kornia.filters.laplacian(vis_re, 3)
        target_gra_w2r = vis_gra

        # ir_re = ir
        # vis_re = vis
        # resize_ori = transforms.Resize([H, W])
        #
        # iw, disp = image_trans(ir_re)
        # # disp = disp.permute(0, 3, 1, 2) * 224
        # disp = disp.permute(0, 3, 1, 2) * 512
        # ir_warp = warp(disp, ir_re)
        # ir_warp_re = resize_ori(ir_warp)
        # # vis_warp = warp(disp, vis_re)
        #
        # ir_warp_gra = kornia.filters.laplacian(ir_warp, 3)
        # vis_gra = kornia.filters.laplacian(vis_re, 3)

        # ir = ir.repeat(1, 3, 1, 1)
        # ir_warp = ir_warp.repeat(1, 3, 1, 1)
        # vis = vis.repeat(1, 3, 1, 1)

        # ir_affine, affine_disp = affine(ir)
        # ir_elastic, elastic_disp = elastic(ir_affine)
        # disp = affine_disp + elastic_disp
        # ir_warp = ir_elastic

        start = time.time()
        with torch.no_grad():
            # ir_feat, vis_feat = basice_net(ir, vis)  # for real scene
            # ir_feat_un, vis_feat2 = basice_net(ir_warp, vis)  # for synthesis scene
            # ir_feat, vis_feat_un = basice_net(ir, vis_warp)
            # # feat_aft_deco_ir, transfer_ir = DecoNet_ir(ir_feat_un, vis_feat)  # version1
            # # feat_aft_deco_vis, transfer_vis = DecoNet_vis(ir_feat_un, vis_feat)  # version1
            # # feat_aft_deco_ir = DecoNet_ir(ir_feat_un, vis_feat)  # version2
            # # feat_aft_deco_vis = DecoNet_vis(vis_feat, ir_feat_un)  # version2
            # # # feat_aft_deco_ir = DecoNet_ir(ir_feat, vis_feat)  # version_real
            # # # feat_aft_deco_vis = DecoNet_vis(vis_feat, ir_feat)  # version_real

            fake_fuse = ir_warp + vis_re
            # fake_fuse = ir + vis  #same&real dataset
            # ir_warp_feat_s, vis_feat_s = ENet_s(ir_warp, vis)
            ir_warp_clip = ir_warp.repeat(1, 3, 1, 1)
            vis_clip = vis_re.repeat(1, 3, 1, 1)
            # ir_warp_feat_s = ENet_s_ir(ir_warp)
            # vis_feat_s = ENet_s_vis(vis)
            ir_warp_feat, vis_feat = ENet(ir_warp, vis_re)
            ir_feat11, vis_feat11 = ENet(ir_re, vis_re)
            # ir_feat, vis_feat = ENet(ir, vis)  # only test for the effectiveness of student fusion net
            # fuse_feats, fuse_out_reg_img_t, m1, m2 = FuseNet_t(ir_feat, vis_feat)
            ir_warp_feat_s = ir_warp_feat
            vis_feat_s = vis_feat
            resize = transforms.Resize([224, 224])
            # resize = transforms.Resize([512, 512])
            ir_warp_resize = resize(ir_warp_clip)

            vis_resize = resize(vis_clip)
            # clip_features_ir_warp = clip_model.encode_image(ir_warp)  # 224size
            # clip_features_vis = clip_model.encode_image(vis)  # 224size
            clip_features_ir_warp = clip_model.encode_image(ir_warp_resize)
            clip_features_vis = clip_model.encode_image(vis_resize)

            # ir_warp_gra = kornia.filters.laplacian(ir_warp, 3)
            # vis_gra = kornia.filters.laplacian(vis, 3)

            clip_ir_warp_fus, clip_vis_fus = siNet(ir_warp_feat_s, vis_feat_s, clip_features_ir_warp, clip_features_vis)
            # ir_reg_img, disp_pre_s, ir_reg_feat, clip_ir_warp_fus, clip_vis_fus = AffNet(
            #     ir_warp_feat_s, vis_feat_s, ir_warp_feat, ir_warp, clip_features_ir_warp, clip_features_vis, ir_warp_gra, vis_gra)  # my superfusion with prompt 1111111111111
            # fuse_feats, fuse_out_reg_img, m1, m2 = FuseNet(ir_reg_feat, vis_feat)
            # ir_reg_img, disp_pre_s, ir_reg_feat, clip_ir_warp_fus, clip_vis_fus = AffNet(
            #     ir_warp_feat_s, vis_feat_s, ir_warp_feat, ir_warp, clip_features_ir_warp, clip_features_vis, ir_warp_gra, vis_gra)  # my superfusion with prompt 1111111111111

            # # 1. 先 L2 归一化（非常重要，CLIP 本身也是这么用的）
            # ir_warp_feat_nor = F.normalize(ir_warp_feat, dim=1)
            # vis_feat_nor = F.normalize(vis_feat, dim=1)

            # # 2. 余弦相似度
            # cos_sim = torch.sum(ir_warp_feat_nor * vis_feat_nor, dim=1)  # [B]
            #
            # # 3. 余弦距离
            # cos_dist = 1.0 - cos_sim  # [B]
            #
            # print('cos_dist', cos_dist.mean())
            #
            #
            # # 1. 先 L2 归一化（非常重要，CLIP 本身也是这么用的）
            # clip_ir_warp_fus_nor = F.normalize(clip_ir_warp_fus, dim=1)
            # clip_vis_fus_nor = F.normalize(clip_vis_fus, dim=1)
            #
            # # 2. 余弦相似度
            # cos_sim_clip = torch.sum(clip_ir_warp_fus_nor * clip_vis_fus_nor, dim=1)  # [B]
            #
            # # 3. 余弦距离
            # cos_dist_clip = 1.0 - cos_sim_clip  # [B]
            #
            # print('cos_dist_clip', cos_dist_clip.mean())

            ir_reg_img_r2w, dis_pre_s_r2w, ir_reg_feat_r2w, ir_reg_img_w2r, dis_pre_s_w2r, ir_reg_feat_w2r = AffNet(
                clip_ir_warp_fus, clip_vis_fus, clip_ir_warp_fus, clip_vis_fus, ir_warp_feat, ir_warp_feat, ir, ir_warp,
                ir_warp_gra, vis_gra, ir_warp_gra, vis_gra, target_gra_w2r, target_gra_w2r)

            fuse_feats, fuse_out_reg_img, m1, m2 = FuseNet(ir_reg_feat_w2r, vis_feat)
            # fuse_feats2, fuse_out_reg_img, m1, m2 = FuseNet(ir_warp_feat, vis_feat)
            # fuse_feats11, fuse_out_reg_img11, m111, m211 = FuseNet(ir_warp_feat, vis_feat11)
            # fuse_feats_t_1, fuse_out_reg_img_t_0, m1, m2 = FuseNet_t(ir_feat, vis_feat)
            # fuse_feats_t, fuse_out_reg_img_t_1, m1, m2 = FuseNet_t(ir_reg_feat, vis_feat)
            # fuse_feats, fuse_out_reg_img = FuseNet(ir_feat, vis_feat)

            # clip_ir_warp_fus = clip_ir_warp_fus.permute(0, 2, 3, 1)
            # clip_vis_fus = clip_vis_fus.permute(0, 2, 3, 1)
            # clip_ir_warp_fus_visual = torch.cat([clip_ir_warp_fus[:, :, :, i] for i in range(64)], dim=1)
            # clip_vis_fus_visual = torch.cat([clip_vis_fus[:, :, :, i] for i in range(64)], dim=1)
            clip_ir_warp_fus_mean = torch.mean(clip_ir_warp_fus, dim=1)
            clip_vis_fus_mean = torch.mean(clip_vis_fus, dim=1)

            ir_warp_feat_s_mean = torch.mean(ir_warp_feat_s, dim=1)
            vis_feat_s_mean = torch.mean(vis_feat_s, dim=1)

            # fuse_out_reg_img_re = resize_ori(fuse_out_reg_img)

            bv = torch.cat([ir_warp, ir_re, vis_re, ir_reg_img_w2r], dim=2)
            # bv_re = torch.cat([ir_warp_re, ir, vis, fuse_out_reg_img_re], dim=2)
            # bv = torch.cat([ir_warp, vis, ir_reg_img], dim=2)
            bv_clip_fuse = torch.cat([clip_ir_warp_fus_mean, clip_vis_fus_mean], dim=2)
            # bv = torch.cat([ir_bv, ir_warp_bv, ir_reg_img_bv, vis_bv])
            bv_bfclip = torch.cat([ir_warp_feat_s_mean, vis_feat_s_mean], dim=2)


            # feat11 = feat11.permute(0, 2, 3, 1)
            # feat12 = feat12.permute(0, 2, 3, 1)
            # feat11 = torch.cat([feat11[:, :, :, i] for i in range(32)], dim=1)
            # feat12 = torch.cat([feat12[:, :, :, i] for i in range(32)], dim=1)

        torch.cuda.synchronize() if str(device) == 'cuda' else None
        end = time.time()
        fus_time.append(end - start)

        # TODO: save fused images
        imsave(ir_warp, dst / 'ir_warp' / file_name)
        # imsave(vis_warp, dst / 'vis_warp' / file_name)
        imsave(vis_gra, dst / 'vis_gra' / file_name)
        # imsave(ir, dst / 'ir' / file_name)
        # imsave(vis, dst / 'vis' / file_name)
        imsave(vis_re, dst / 'vis' / file_name)
        imsave(bv, dst / 'bv' / file_name)
        # imsave(bv_re, dst / 'bv_re' / file_name)
        imsave(bv_clip_fuse, dst / 'bv_clip_fuse' / file_name)
        imsave(bv_bfclip, dst / 'bv_bfclip' / file_name)
        imsave(ir_reg_img_w2r, dst / 'ir_reg_img_w2r' / file_name)
        imsave(fuse_out_reg_img, dst / 'fuse_out_reg_img' / file_name)
        # imsave(fuse_out_reg_img11, dst / 'fuse_out_reg_img11' / file_name)
        imsave(fake_fuse, dst / 'fake' / file_name)
        # imsave(fus_reg_img_1_src_gra_w2r, dst / 'fus_reg_img_1_src_gra_w2r' / file_name)
        # imsave(fus_reg_img_gra_src_gra_w2r, dst / 'fus_reg_img_gra_src_gra_w2r' / file_name)
        # imsave(fuse_out_reg_img_t, dst / 'fuse_out_reg_img_t' / file_name)
        # imsave(fuse_out_reg_img_t_1, dst / 'fuse_out_reg_img_t_1' / file_name)
        # imsave(fuse_out_reg_img_t_0, dst / 'fuse_out_reg_img_t_0' / file_name)

        # imsave(feat11, dst / 'feat11' / file_name)
        # imsave(feat12, dst / 'feat12' / file_name)

        # fuse_mean = statistics.mean(fus_time[1:])
    #     print('fuse time (average): {:.4f}'.format(fuse_mean))
    #     print('fps (equivalence): {:.4f}'.format(1. / fuse_mean))
    # # -------------------------------------------------------------------------------------------------#
    #     print('fuse time (average): {:.4f}'.format(end - start))

        # inputir_w = torch.randn(1, 1, 512, 512).cuda()
        # inputvis = torch.randn(1, 1, 512, 512).cuda()
        #
        # inputsi1 = torch.randn(1, 64, 512, 512).cuda()
        # inputsi2 = torch.randn(1, 64, 512, 512).cuda()
        # inputsi3 = torch.randn(1, 512).cuda()
        # inputsi4 = torch.randn(1, 512).cuda()
        #
        # input1 = torch.randn(1, 64, 512, 512).cuda()
        # input2 = torch.randn(1, 64, 512, 512).cuda()
        # input3 = torch.randn(1, 64, 512, 512).cuda()
        # input4 = torch.randn(1, 64, 512, 512).cuda()
        # input5 = torch.randn(1, 64, 512, 512).cuda()
        # input6 = torch.randn(1, 64, 512, 512).cuda()
        # input7 = torch.randn(1, 1, 512, 512).cuda()
        # input8 = torch.randn(1, 1, 512, 512).cuda()
        # input9 = torch.randn(1, 1, 512, 512).cuda()
        # input10 = torch.randn(1, 1, 512, 512).cuda()
        # input11 = torch.randn(1, 1, 512, 512).cuda()
        # input12 = torch.randn(1, 1, 512, 512).cuda()
        # input13 = torch.randn(1, 1, 512, 512).cuda()
        # input14 = torch.randn(1, 1, 512, 512).cuda()
        # #
        # inputfus1 = torch.randn(1, 64, 512, 512).cuda()
        # inputfus2 = torch.randn(1, 64, 512, 512).cuda()

        # inputir_w = torch.randn(1, 1, 256, 256).cuda()
        # inputvis = torch.randn(1, 1, 256, 256).cuda()
        #
        # inputsi1 = torch.randn(1, 64, 256, 256).cuda()
        # inputsi2 = torch.randn(1, 64, 256, 256).cuda()
        # inputsi3 = torch.randn(1, 512).cuda()
        # inputsi4 = torch.randn(1, 512).cuda()
        #
        # input1 = torch.randn(1, 64, 256, 256).cuda()
        # input2 = torch.randn(1, 64, 256, 256).cuda()
        # input3 = torch.randn(1, 64, 256, 256).cuda()
        # input4 = torch.randn(1, 64, 256, 256).cuda()
        # input5 = torch.randn(1, 64, 256, 256).cuda()
        # input6 = torch.randn(1, 64, 256, 256).cuda()
        # input7 = torch.randn(1, 1, 256, 256).cuda()
        # input8 = torch.randn(1, 1, 256, 256).cuda()
        # input9 = torch.randn(1, 1, 256, 256).cuda()
        # input10 = torch.randn(1, 1, 256, 256).cuda()
        # input11 = torch.randn(1, 1, 256, 256).cuda()
        # input12 = torch.randn(1, 1, 256, 256).cuda()
        # input13 = torch.randn(1, 1, 256, 256).cuda()
        # input14 = torch.randn(1, 1, 256, 256).cuda()
        # #
        # inputfus1 = torch.randn(1, 64, 256, 256).cuda()
        # inputfus2 = torch.randn(1, 64, 256, 256).cuda()

        inputir_w = torch.randn(1, 1, 1024, 1024).cuda()
        inputvis = torch.randn(1, 1, 1024, 1024).cuda()

        inputsi1 = torch.randn(1, 64, 1024, 1024).cuda()
        inputsi2 = torch.randn(1, 64, 1024, 1024).cuda()
        inputsi3 = torch.randn(1, 512).cuda()
        inputsi4 = torch.randn(1, 512).cuda()

        input1 = torch.randn(1, 64, 1024, 1024).cuda()
        input2 = torch.randn(1, 64, 1024, 1024).cuda()
        input3 = torch.randn(1, 64, 1024, 1024).cuda()
        input4 = torch.randn(1, 64, 1024, 1024).cuda()
        input5 = torch.randn(1, 64, 1024, 1024).cuda()
        input6 = torch.randn(1, 64, 1024, 1024).cuda()
        input7 = torch.randn(1, 1, 1024, 1024).cuda()
        input8 = torch.randn(1, 1, 1024, 1024).cuda()
        input9 = torch.randn(1, 1, 1024, 1024).cuda()
        input10 = torch.randn(1, 1, 1024, 1024).cuda()
        input11 = torch.randn(1, 1, 1024, 1024).cuda()
        input12 = torch.randn(1, 1, 1024, 1024).cuda()
        input13 = torch.randn(1, 1, 1024, 1024).cuda()
        input14 = torch.randn(1, 1, 1024, 1024).cuda()
        #
        inputfus1 = torch.randn(1, 64, 1024, 1024).cuda()
        inputfus2 = torch.randn(1, 64, 1024, 1024).cuda()

        flops_enco, params_enco = profile(ENet, (inputir_w, inputvis))
        flops_si, params_si = profile(siNet, (inputsi1, inputsi2, inputsi3, inputsi4))
        flops_aff, params_aff = profile(AffNet, (input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13, input14))
        flops_fuse, params_fuse = profile(FuseNet, (inputfus1, inputfus2))
        print('flops_enco: %.4f G, params_enco:%.4f M' % (flops_enco / 1e9, params_enco / 1e6))
        print('flops_si: %.4f G, params_si:%.4f M' % (flops_si / 1e9, params_si / 1e6))
        print('flops_aff: %.4f G, params_aff:%.4f M' % (flops_aff / 1e9, params_aff / 1e6))
        print('flops_fuse: %.4f G, params_fuse:%.4f M' % (flops_fuse / 1e9, params_fuse / 1e6))
    #-------------------------------------------------------------------------------------------------#
        # print("Params(M_enco): %.4f" % (params_count(ENet) / (1000 ** 2)))
        # print("Params(M_si): %.4f" % (params_count(siNet) / (1000 ** 2)))
        # print("Params(M_aff): %.4f" % (params_count(AffNet) / (1000 ** 2)))
        # print("Params(M_fuse): %.4f" % (params_count(FuseNet) / (1000 ** 2)))

    # statistics time record
    fuse_mean = statistics.mean(fus_time[1:])
    print('fuse time (average): {:.4f}'.format(fuse_mean))
    print('fps (equivalence): {:.4f}'.format(1. / fuse_mean))

    pass
def imsave(im_s: [Tensor], dst: pathlib.Path, im_name: str = ''):
    """
    save images to path
    :param im_s: image(s)
    :param dst: if one image: path; if multiple images: folder path
    :param im_name: name of image
    """

    im_s = im_s if type(im_s) == list else [im_s]
    dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
    for im_ts, p in zip(im_s, dst):
        im_ts = im_ts.squeeze().cpu()

        # DKY
        im_ts = im_ts.numpy()
        im_ts = (im_ts - np.min(im_ts)) / (np.max(im_ts) - np.min(im_ts))
        im_ts = np.clip(im_ts * 255.0, 0., 255.)
        im_ts = torch.from_numpy(im_ts)

        p.parent.mkdir(parents=True, exist_ok=True)
        # im_cv = kornia.utils.tensor_to_image(im_ts) * 255.
        im_cv = kornia.utils.tensor_to_image(im_ts)
        cv2.imwrite(str(p), im_cv)


def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = hyper_args()
    main(args)