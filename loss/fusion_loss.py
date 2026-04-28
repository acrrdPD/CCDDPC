import kornia.losses
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.ms_ssim import MSSSIM
import torchvision


class FusionLoss_main(nn.Module):
    # def __init__(self, alpha=1, beta=1, theta=0.5):
    #     super(FusionLoss_main, self).__init__()
    def __init__(self):
        super(FusionLoss_main, self).__init__()

        self.ms_ssim = MSSSIM()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.grad_loss = JointGrad()

    def forward(self, im_fus, im_ir, im_vi, map_ir, map_vi, ir_c_mask, vis_c_mask):

        l1_loss = self.l1_loss(im_fus, (map_ir * im_ir + map_vi * im_vi))
        grad_loss = self.grad_loss(im_fus, im_ir, im_vi)

        loss_fus = 0.05 * l1_loss + grad_loss    #ours
        loss_cc = self.l1_loss(1 - vis_c_mask, ir_c_mask)
        loss = loss_fus + loss_cc

        return loss


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X) # torch.Size([1, 64, 256, 256])
        h_relu2 = self.slice2(h_relu1) # torch.Size([1, 128, 128, 128])
        h_relu3 = self.slice3(h_relu2) # torch.Size([1, 256, 64, 64])
        h_relu4 = self.slice4(h_relu3) # torch.Size([1, 512, 32, 32])
        h_relu5 = self.slice5(h_relu4) # torch.Size([1, 512, 16, 16])
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        if torch.cuda.is_available():
            self.vgg.cuda()
        self.vgg.eval()
        set_requires_grad(self.vgg, False)
        self.L1Loss = nn.L1Loss()
        self.criterion2 = nn.MSELoss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 , 1.0]

    def forward(self, x, y):
        contentloss = 0
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x_vgg = self.vgg(x)
        with torch.no_grad():
            y_vgg = self.vgg(y)

        contentloss += self.L1Loss(x_vgg[3], y_vgg[3].detach())

        return contentloss


class gradient_loss(nn.Module):
    def __init__(self):
        super(gradient_loss, self).__init__()

    def forward(self, s, penalty='l2'):
        dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
        dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0


class affineLoss_dky_s_not(nn.Module):
    def __init__(self):
        super(affineLoss_dky_s_not, self).__init__()
        self.gradient_loss = gradient_loss()
        # self.multi_loss = multi_loss()
        self.feat_loss = VGGLoss()
        self.l2_loss = nn.MSELoss()

        self.l1_loss = nn.L1Loss()
        self.grad_loss = JointGrad()

        self.kl = nn.KLDivLoss(reduction='batchmean')


    def forward(self, feat_warp_s, feat_reg_s, feat_warp, feat_reg, flow_s_r2w, flow_s_w2r, flow, fuse_s, fuse, ir, ir_warp, vis, ir_reg_img_r2w, ir_reg_img_w2r,
                    map_ir, map_vi, ir_c_mask, vis_c_mask):  # tgt: torch.Size([16, 1, 224, 224])

        feat_loss = self.l1_loss(feat_warp, feat_warp_s) + self.l1_loss(feat_reg, feat_reg_s)


        feat_loss_add = feat_loss

        # TODO: regis loss
        flow_loss = torch.nn.functional.l1_loss(flow, flow_s_r2w)
        grad = self.gradient_loss(flow_s_w2r)
        feat_1_rw2 = self.feat_loss(ir_reg_img_r2w, ir_warp)
        feat_1_w2r = self.feat_loss(ir_reg_img_w2r, ir)
        feat_1 = feat_1_rw2 + feat_1_w2r
        regis_loss = 0.1 * flow_loss + feat_1 + 0.1 * grad


        loss = regis_loss + feat_loss_add

        return loss, regis_loss, feat_loss_add


class affineLoss_dky(nn.Module):
    def __init__(self):
        super(affineLoss_dky, self).__init__()
        self.gradient_loss = gradient_loss()
        self.feat_loss = VGGLoss()

    def forward(self, y, y_f, tgt, src, flow_r2w, flow_w2r, flow_label): # tgt: torch.Size([16, 1, 224, 224])

        hyper_grad = 10
        hyper_feat = 1

        ncc_1 = torch.nn.functional.l1_loss(tgt, y) #1111111111111111

        ncc = ncc_1

        # TODO: feature loss
        feat_1 = self.feat_loss(y, tgt)
        feat_2 = self.feat_loss(y_f, src)
        feat = feat_1 + feat_2


         # TODO: gradient loss
        smooth_loss = self.gradient_loss(flow_w2r)
        f_loss = torch.nn.functional.l1_loss(flow_r2w, flow_label)

        filed_loss = 0.1 * smooth_loss + 0.1 * f_loss

        # TODO: total loss
        loss = hyper_feat * feat + filed_loss  #cyd
        return loss, ncc, filed_loss  #cyd


class JointGrad(nn.Module):
    def __init__(self):
        super(JointGrad, self).__init__()

        self.laplacian = kornia.filters.laplacian
        self.l1_loss = nn.L1Loss()

    def forward(self, im_fus, im_ir, im_vi):

        ir_grad = torch.abs(self.laplacian(im_ir, 3))
        vi_grad = torch.abs(self.laplacian(im_vi, 3))
        # fus_grad = torch.abs(self.laplacian(im_fus, 3))
        fus_grad = self.laplacian(im_fus, 3)

        JGrad = torch.where(ir_grad-vi_grad >= 0, self.laplacian(im_ir, 3), self.laplacian(im_vi, 3))
        loss_JGrad = self.l1_loss(JGrad, fus_grad)


        return loss_JGrad


class SingleGrad(nn.Module):
    def __init__(self):
        super(SingleGrad, self).__init__()

        self.laplacian = kornia.filters.laplacian
        self.l1_loss = nn.L1Loss()

    def forward(self, im_fus, im_ir):

        ir_grad = self.laplacian(im_ir, 3)
        fus_grad = self.laplacian(im_fus, 3)

        loss_SGrad = self.l1_loss(ir_grad, fus_grad)

        return loss_SGrad

    
    
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()

    def forward(self, p, q):

        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)

       # loss = F.kl_div(q.log(), p, reducion='batchmean')
        loss = F.kl_div(q.log(), p)

        return loss