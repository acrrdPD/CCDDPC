import torch
import torch.nn as nn

from dataloader.reg_data import Warper2d

from models.unet import UNet_3, UNet_64
import kornia.losses
import torch.nn.functional as F

# shape = (256, 256)
shape = (224, 224)

class SuperNet_t_unet(nn.Module):
    def __init__(self):
        super(SuperNet_t_unet, self).__init__()
        # int_steps = 7   #
        self.warp = Warper2d()
        self.sig = nn.Sigmoid()
        self.unet_3 = UNet_3()
        self.unet_64 = UNet_64()

    def forward(self, feat_fus_reg, feat_fus_warp, img_fus_warp, img_fus_reg, fus_warp_gra, fus_reg_gra, target_gra_r2w, target_gra_w2r):

        cat_feat_fus_w2r = torch.cat((feat_fus_warp, feat_fus_reg), dim=1)
        cat_feat_fus_r2w = torch.cat((feat_fus_reg, feat_fus_warp), dim=1)
        deformation1_r2w = self.unet_64(cat_feat_fus_r2w)
        deformation1_w2r = self.unet_64(cat_feat_fus_w2r)

        fus_reg_img_1_src_r2w = self.warp(deformation1_r2w, img_fus_reg)  #sdy
        fus_reg_img_1_src_gra_r2w = kornia.filters.laplacian(fus_reg_img_1_src_r2w, 3)
        s_1_r2w = torch.cat((fus_reg_img_1_src_gra_r2w, target_gra_r2w), dim=1)
        deformation1_avg_c_r2w = F.avg_pool2d(s_1_r2w, kernel_size=deformation1_r2w.size()[2:])
        deformation1_max_c_r2w = F.max_pool2d(s_1_r2w, kernel_size=deformation1_r2w.size()[2:])
        deformation1_mask_c_r2w = deformation1_avg_c_r2w + deformation1_max_c_r2w
        deformation1_mask_c_r2w = self.sig(deformation1_mask_c_r2w)
        deformation1_avg_s_r2w = torch.mean(s_1_r2w, dim=1, keepdim=True)
        deformation1_max_s_r2w, _ = torch.max(s_1_r2w, dim=1, keepdim=True)
        deformation1_mask_s_r2w = deformation1_avg_s_r2w + deformation1_max_s_r2w
        deformation1_mask_s_r2w = self.sig(deformation1_mask_s_r2w)
        deformation1_mask_r2w = deformation1_mask_c_r2w * deformation1_mask_s_r2w

        deformation1_r2w = deformation1_r2w * deformation1_mask_r2w

        cat_feat_grad_r2w = torch.cat((fus_reg_gra, fus_warp_gra), dim=1)
        deformation1_gra_r2w = self.unet_3(cat_feat_grad_r2w)

        fus_reg_img_gra_src_r2w = self.warp(deformation1_gra_r2w, img_fus_reg)  #sdy
        fus_reg_img_gra_src_gra_r2w = kornia.filters.laplacian(fus_reg_img_gra_src_r2w, 3)
        s_gra_r2w = torch.cat((fus_reg_img_gra_src_gra_r2w, target_gra_r2w), dim=1)
        deformation1_gra_avg_c_r2w = F.avg_pool2d(s_gra_r2w, kernel_size=deformation1_r2w.size()[2:])
        deformation1_gra_max_c_r2w = F.max_pool2d(s_gra_r2w, kernel_size=deformation1_r2w.size()[2:])
        deformation1_gra_mask_c_r2w = deformation1_gra_avg_c_r2w + deformation1_gra_max_c_r2w
        deformation1_gra_mask_c_r2w = self.sig(deformation1_gra_mask_c_r2w)
        deformation1_gra_avg_s_r2w = torch.mean(s_gra_r2w, dim=1, keepdim=True)
        deformation1_gra_max_s_r2w, _ = torch.max(s_gra_r2w, dim=1, keepdim=True)
        deformation1_gra_mask_s_r2w = deformation1_gra_avg_s_r2w + deformation1_gra_max_s_r2w
        deformation1_gra_mask_s_r2w = self.sig(deformation1_gra_mask_s_r2w)
        deformation1_gra_mask_r2w = deformation1_gra_mask_c_r2w * deformation1_gra_mask_s_r2w

        deformation1_gra_r2w = deformation1_gra_r2w * deformation1_gra_mask_r2w
        deformation_r2w = deformation1_r2w + deformation1_gra_r2w

        fus_reg_feat_r2w = self.warp(deformation_r2w, img_fus_reg)  #sdy
        fus_reg_img_r2w = self.warp(deformation_r2w, img_fus_reg)  #sdy

        #another

        fus_reg_img_1_src_w2r = self.warp(deformation1_w2r, img_fus_warp)  # sdy
        fus_reg_img_1_src_gra_w2r = kornia.filters.laplacian(fus_reg_img_1_src_w2r, 3)
        s_1_w2r = torch.cat((fus_reg_img_1_src_gra_w2r, target_gra_w2r), dim=1)
        deformation1_avg_c_w2r = F.avg_pool2d(s_1_w2r, kernel_size=deformation1_w2r.size()[2:])
        deformation1_max_c_w2r = F.max_pool2d(s_1_w2r, kernel_size=deformation1_w2r.size()[2:])
        deformation1_mask_c_w2r = deformation1_avg_c_w2r + deformation1_max_c_w2r
        deformation1_mask_c_w2r = self.sig(deformation1_mask_c_w2r)
        deformation1_avg_s_w2r = torch.mean(s_1_w2r, dim=1, keepdim=True)
        deformation1_max_s_w2r, _ = torch.max(s_1_w2r, dim=1, keepdim=True)
        deformation1_mask_s_w2r = deformation1_avg_s_w2r + deformation1_max_s_w2r
        deformation1_mask_s_w2r = self.sig(deformation1_mask_s_w2r)
        deformation1_mask_w2r = deformation1_mask_c_w2r * deformation1_mask_s_w2r

        deformation1_w2r = deformation1_w2r * deformation1_mask_w2r

        cat_feat_grad_w2r = torch.cat((fus_warp_gra, fus_reg_gra), dim=1)
        deformation1_gra_w2r = self.unet_3(cat_feat_grad_w2r)

        fus_reg_img_gra_src_w2r = self.warp(deformation1_gra_w2r, img_fus_warp)  # sdy
        fus_reg_img_gra_src_gra_w2r = kornia.filters.laplacian(fus_reg_img_gra_src_w2r, 3)
        s_gra_w2r = torch.cat((fus_reg_img_gra_src_gra_w2r, target_gra_w2r), dim=1)
        deformation1_gra_avg_c_w2r = F.avg_pool2d(s_gra_w2r, kernel_size=deformation1_w2r.size()[2:])
        deformation1_gra_max_c_w2r = F.max_pool2d(s_gra_w2r, kernel_size=deformation1_w2r.size()[2:])
        deformation1_gra_mask_c_w2r = deformation1_gra_avg_c_w2r + deformation1_gra_max_c_w2r
        deformation1_gra_mask_c_w2r = self.sig(deformation1_gra_mask_c_w2r)
        deformation1_gra_avg_s_w2r = torch.mean(s_gra_w2r, dim=1, keepdim=True)
        deformation1_gra_max_s_w2r, _ = torch.max(s_gra_w2r, dim=1, keepdim=True)
        deformation1_gra_mask_s_w2r = deformation1_gra_avg_s_w2r + deformation1_gra_max_s_w2r
        deformation1_gra_mask_s_w2r = self.sig(deformation1_gra_mask_s_w2r)
        deformation1_gra_mask_w2r = deformation1_gra_mask_c_w2r * deformation1_gra_mask_s_w2r

        deformation1_gra_w2r = deformation1_gra_w2r * deformation1_gra_mask_w2r
        deformation_w2r = deformation1_w2r + deformation1_gra_w2r

        fus_reg_feat_w2r = self.warp(deformation_w2r, img_fus_warp)  #sdy
        fus_reg_img_w2r = self.warp(deformation_w2r, img_fus_warp)  #sdy


        # return fus_reg_img, deformation, down2, down4, down8, fus_reg_feat, gdm, gdm_inverse  #only for training
        return fus_reg_img_r2w, deformation_r2w, fus_reg_feat_r2w, fus_reg_img_w2r, deformation_w2r, fus_reg_feat_w2r #only for test
        # return ir_reg_img, deformation, ir_reg_feat, gdm, gdm_inverse  #only for test


class SuperNet_s_unet(nn.Module):
    def __init__(self):
        super(SuperNet_s_unet, self).__init__()
        # int_steps = 7   #
        self.warp = Warper2d()
        self.sig = nn.Sigmoid()
        self.unet_3 = UNet_3()
        self.unet_64 = UNet_64()

    def forward(self, clip_ir_warp_fus, clip_vis_fus, clip_ir_fus, clip_vis_warp_fus, ir_feat, ir_warp_feat, ir, ir_warp, ir_warp_gra, vis_gra, ir_gra, vis_warp_gra, target_gra_r2w, target_gra_w2r):



        cat_feat_w2r = torch.cat((clip_ir_warp_fus, clip_vis_fus), dim=1)
        deformation1_w2r = self.unet_64(cat_feat_w2r)

        fus_reg_img_1_src_w2r = self.warp(deformation1_w2r, ir_warp)  # sdy
        fus_reg_img_1_src_gra_w2r = kornia.filters.laplacian(fus_reg_img_1_src_w2r, 3)
        s_1_w2r = torch.cat((fus_reg_img_1_src_gra_w2r, target_gra_w2r), dim=1)
        deformation1_avg_c_w2r = F.avg_pool2d(s_1_w2r, kernel_size=deformation1_w2r.size()[2:])
        deformation1_max_c_w2r = F.max_pool2d(s_1_w2r, kernel_size=deformation1_w2r.size()[2:])
        deformation1_mask_c_w2r = deformation1_avg_c_w2r + deformation1_max_c_w2r
        deformation1_mask_c_w2r = self.sig(deformation1_mask_c_w2r)
        deformation1_avg_s_w2r = torch.mean(s_1_w2r, dim=1, keepdim=True)
        deformation1_max_s_w2r, _ = torch.max(s_1_w2r, dim=1, keepdim=True)
        deformation1_mask_s_w2r = deformation1_avg_s_w2r + deformation1_max_s_w2r
        deformation1_mask_s_w2r = self.sig(deformation1_mask_s_w2r)
        deformation1_mask_w2r = deformation1_mask_c_w2r * deformation1_mask_s_w2r

        deformation1_w2r = deformation1_w2r * deformation1_mask_w2r

        cat_feat_grad_w2r = torch.cat((ir_warp_gra, vis_gra), dim=1)
        deformation1_gra_w2r = self.unet_3(cat_feat_grad_w2r)
        # deformation1_gra, deformation2_gra = self.DM(ir_warp_gra, vis_gra)

        fus_reg_img_gra_src_w2r = self.warp(deformation1_gra_w2r, ir_warp)  # sdy
        fus_reg_img_gra_src_gra_w2r = kornia.filters.laplacian(fus_reg_img_gra_src_w2r, 3)
        s_gra_w2r = torch.cat((fus_reg_img_gra_src_gra_w2r, target_gra_w2r), dim=1)
        deformation1_gra_avg_c_w2r = F.avg_pool2d(s_gra_w2r, kernel_size=deformation1_w2r.size()[2:])
        deformation1_gra_max_c_w2r = F.max_pool2d(s_gra_w2r, kernel_size=deformation1_w2r.size()[2:])
        deformation1_gra_mask_c_w2r = deformation1_gra_avg_c_w2r + deformation1_gra_max_c_w2r
        deformation1_gra_mask_c_w2r = self.sig(deformation1_gra_mask_c_w2r)
        deformation1_gra_avg_s_w2r = torch.mean(s_gra_w2r, dim=1, keepdim=True)
        deformation1_gra_max_s_w2r, _ = torch.max(s_gra_w2r, dim=1, keepdim=True)
        # deformation1_gra_avg_s = self.CAP(s_gra)
        # deformation1_gra_max_s = self.CMP(s_gra)
        deformation1_gra_mask_s_w2r = deformation1_gra_avg_s_w2r + deformation1_gra_max_s_w2r
        deformation1_gra_mask_s_w2r = self.sig(deformation1_gra_mask_s_w2r)
        deformation1_gra_mask_w2r = deformation1_gra_mask_c_w2r * deformation1_gra_mask_s_w2r

        deformation1_gra_w2r = deformation1_gra_w2r * deformation1_gra_mask_w2r
        deformation_w2r = deformation1_w2r + deformation1_gra_w2r

        ir_reg_feat_w2r = self.warp(deformation_w2r, ir_warp_feat)  #sdy
        ir_reg_img_w2r = self.warp(deformation_w2r, ir_warp)  #sdy

        #another

        cat_feat_r2w = torch.cat((clip_ir_fus, clip_vis_warp_fus), dim=1)
        deformation1_r2w = self.unet_64(cat_feat_r2w)

        fus_reg_img_1_src_r2w = self.warp(deformation1_r2w, ir)  # sdy
        fus_reg_img_1_src_gra_r2w = kornia.filters.laplacian(fus_reg_img_1_src_r2w, 3)
        s_1_r2w = torch.cat((fus_reg_img_1_src_gra_r2w , target_gra_r2w), dim=1)
        deformation1_avg_c_r2w = F.avg_pool2d(s_1_r2w, kernel_size=deformation1_w2r.size()[2:])
        deformation1_max_c_r2w = F.max_pool2d(s_1_r2w, kernel_size=deformation1_w2r.size()[2:])
        deformation1_mask_c_r2w = deformation1_avg_c_r2w + deformation1_max_c_r2w
        deformation1_mask_c_r2w = self.sig(deformation1_mask_c_r2w)
        deformation1_avg_s_r2w = torch.mean(s_1_r2w, dim=1, keepdim=True)
        deformation1_max_s_r2w, _ = torch.max(s_1_r2w, dim=1, keepdim=True)
        # deformation1_avg_s = self.CAP(s_1)
        # deformation1_max_s = self.CMP(s_1)
        deformation1_mask_s_r2w = deformation1_avg_s_r2w + deformation1_max_s_r2w
        deformation1_mask_s_r2w = self.sig(deformation1_mask_s_r2w)
        deformation1_mask_r2w = deformation1_mask_c_r2w * deformation1_mask_s_r2w

        deformation1_r2w = deformation1_r2w * deformation1_mask_r2w

        # deformation1_gra, deformation2_gra, down2_gra, down4_gra, down8_gra = self.DM(fus_warp_gra, fus_reg_gra)
        cat_feat_grad_r2w = torch.cat((ir_gra, vis_warp_gra), dim=1)
        deformation1_gra_r2w = self.unet_3(cat_feat_grad_r2w)
        # deformation1_gra, deformation2_gra = self.DM(ir_warp_gra, vis_gra)

        fus_reg_img_gra_src_r2w = self.warp(deformation1_gra_r2w, ir)  # sdy
        fus_reg_img_gra_src_gra_r2w = kornia.filters.laplacian(fus_reg_img_gra_src_r2w, 3)
        s_gra_r2w = torch.cat((fus_reg_img_gra_src_gra_r2w, target_gra_r2w), dim=1)
        deformation1_gra_avg_c_r2w = F.avg_pool2d(s_gra_r2w, kernel_size=deformation1_r2w.size()[2:])
        deformation1_gra_max_c_r2w = F.max_pool2d(s_gra_r2w, kernel_size=deformation1_r2w.size()[2:])
        deformation1_gra_mask_c_r2w = deformation1_gra_avg_c_r2w + deformation1_gra_max_c_r2w
        deformation1_gra_mask_c_r2w = self.sig(deformation1_gra_mask_c_r2w)
        deformation1_gra_avg_s_r2w = torch.mean(s_gra_r2w, dim=1, keepdim=True)
        deformation1_gra_max_s_r2w, _ = torch.max(s_gra_r2w, dim=1, keepdim=True)
        # deformation1_gra_avg_s = self.CAP(s_gra)
        # deformation1_gra_max_s = self.CMP(s_gra)
        deformation1_gra_mask_s_r2w = deformation1_gra_avg_s_r2w + deformation1_gra_max_s_r2w
        deformation1_gra_mask_s_r2w = self.sig(deformation1_gra_mask_s_r2w)
        deformation1_gra_mask_r2w = deformation1_gra_mask_c_r2w * deformation1_gra_mask_s_r2w

        deformation1_gra_r2w = deformation1_gra_r2w * deformation1_gra_mask_r2w
        deformation_r2w = deformation1_r2w + deformation1_gra_r2w

        ir_reg_feat_r2w = self.warp(deformation_r2w, ir_feat)  # sdy
        ir_reg_img_r2w = self.warp(deformation_r2w, ir)  # sdy

        # return fus_reg_img, deformation, down2, down4, down8, fus_reg_feat, gdm, gdm_inverse  #only for training
        return ir_reg_img_r2w, deformation_r2w, ir_reg_feat_r2w, ir_reg_img_w2r, deformation_w2r, ir_reg_feat_w2r  #only for test
        # return ir_reg_img, deformation, ir_reg_feat, gdm, gdm_inverse  #only for test

class SINet(nn.Module):
    def __init__(self):
        super(SINet, self).__init__()
        # int_steps = 7   #
        lr = 0.001
        # encoders
        lr = 0.001
        # encoders

        self.fuse_res_ir = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fuse_res_vis = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.mlp_vis = nn.Linear(512, 64)
        self.mlp_ir = nn.Linear(512, 64)

    def forward(self, feat_warp_ir_for, feat_vis_for, clip_features_ir_warp, clip_features_vis):

        clip_features_vis = clip_features_vis.to(torch.float32)
        clip_features_vis = self.mlp_vis(clip_features_vis)
        clip_features_vis = clip_features_vis.unsqueeze(2)
        clip_features_vis = clip_features_vis.unsqueeze(3)
        clip_ir_warp = clip_features_vis + feat_warp_ir_for
        clip_ir_warp_fus = self.fuse_res_ir(clip_ir_warp)

        clip_features_ir_warp = clip_features_ir_warp.to(torch.float32)
        clip_features_ir_warp = self.mlp_ir(clip_features_ir_warp)
        clip_features_ir_warp = clip_features_ir_warp.unsqueeze(2)
        clip_features_ir_warp = clip_features_ir_warp.unsqueeze(3)
        clip_vis = clip_features_ir_warp + feat_vis_for
        clip_vis_fus = self.fuse_res_vis(clip_vis)

        return clip_ir_warp_fus, clip_vis_fus  #only for test
        # return ir_reg_img, deformation, ir_reg_feat, gdm, gdm_inverse  #only for test


