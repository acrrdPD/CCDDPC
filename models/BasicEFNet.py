import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
    self.instance_norm = nn.InstanceNorm2d(num_features=nChannels)
  def forward(self, x):
    out = self.instance_norm(x)
    out = F.relu(self.conv(out))
    out = torch.cat((x, out), 1)
    return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate
    self.dense_layers = nn.Sequential(*modules)
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out


class CrossAttention_norm_q(nn.Module):
    def __init__(self, kernel_size=7):
        super(CrossAttention_norm_q, self).__init__()
        self.a1 = nn.Conv2d(64, 64, (1, 1))  # for feature(test
        self.b1 = nn.Conv2d(64, 64, (1, 1))
        self.c1 = nn.Conv2d(64, 64, (1, 1))
        self.out_conv1 = nn.Conv2d(64, 64, (1, 1))
        self.sm1 = nn.Softmax(dim=-1)

    def forward(self, q, v): # mutual representation
        Q = self.a1(q)  # s2c1
        K = self.b1(q)

        V = self.c1(v)
        b, c, h, w = Q.size()
        Q = Q.view(b, -1, w * h)  # C x HsWs
        b, c, h, w = K.size()
        K = K.view(b, -1, w * h).permute(0, 2, 1)  # HsWs x C

        S = torch.bmm(Q, K)  # HcWc x HsWs
        S = S / 64
        S = self.sm1(S)  # style attention map

        b, c, h, w = V.size()
        V = V.view(b, -1, w * h)  # C x HsWs
        O = torch.bmm(S, V)  # C x HcWc

        O = O.view(b, c, h, w)
        O = self.out_conv1(O)
        O += q
        return O


class basiceNet(nn.Module):
  def __init__(self, nfeats=64):
    super(basiceNet, self).__init__()

    # head
    self.conv1_1 = nn.Sequential(
      nn.Conv2d(1, nfeats, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.2)
    )
    self.conv2_1 = nn.Sequential(
      nn.Conv2d(1, nfeats, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.2)
    )

    # body-densenet
    self.nChannels = nfeats
    self.nDenselayer = 3
    self.growthRate = nfeats
    Irb_path = []
    Vib_path = []
    for i in range(1):
      Irb_path.append(RDB(self.nChannels, self.nDenselayer, self.growthRate))
      Vib_path.append(RDB(self.nChannels, self.nDenselayer, self.growthRate))
    self.Irb_path = nn.Sequential(*Irb_path)
    self.Vib_path = nn.Sequential(*Vib_path)
    self.CA = CrossAttention_norm_q()

  def forward(self, ir, vis):

    # head
    irb_feat = self.conv1_1(ir)
    visb_feat = self.conv2_1(vis)

    # body-densenet
    irb_dfeats = self.Irb_path(irb_feat)
    visb_dfeats = self.Vib_path(visb_feat)

    return irb_dfeats, visb_dfeats


class Channel_Max_Pooling(torch.nn.MaxPool1d):
    def __init__(self, channels, isize):
        super(Channel_Max_Pooling, self).__init__(channels)
        self.kernel_size = channels
        self.stride = isize

    def forward(self, input):
        n, c, w, h = input.size()
        hi = h
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = torch.nn.functional.max_pool1d(input, self.kernel_size, self.stride,
                                                self.padding, self.dilation, self.ceil_mode,
                                                self.return_indices)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        pooled = rearrange(pooled, 'b c (h w) -> b c h w', h=224, c=1)
        return pooled

class Channel_Avg_Pooling(torch.nn.MaxPool1d):
    def __init__(self, channels, isize):
        super(Channel_Avg_Pooling, self).__init__(channels)
        self.kernel_size = channels
        self.stride = isize

    def forward(self, input):
        n, c, w, h = input.size()
        hi = h
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = torch.nn.functional.avg_pool1d(input, self.kernel_size, self.stride)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        pooled = rearrange(pooled, 'b c (h w) -> b c h w', h=224, c=1)
        return pooled


class FuseModule_max(nn.Module):
  """ Interactive fusion module"""
  def __init__(self, in_dim=128):
    super(FuseModule_max, self).__init__()

    self.ir_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
    self.vis_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

    self.sig = nn.Sigmoid()
    self.cap = Channel_Avg_Pooling(in_dim, 1)
    self.cmp = Channel_Max_Pooling(in_dim, 1)
    self.mlp_1 = nn.Linear(in_dim, in_dim//8)
    self.mlp_2 = nn.Linear(in_dim//8, in_dim)

  def forward(self, ir, vis):
    ir_sig = self.sig(ir)
    ir = torch.cat((ir, ir_sig), dim=1)
    ir_c = self.ir_conv(ir)
    ir_c_avg = F.avg_pool2d(ir_c, kernel_size=ir_c.size()[2:])
    ir_c_max = F.max_pool2d(ir_c, kernel_size=ir_c.size()[2:])
    ir_c_mask = ir_c_avg + ir_c_max
    ir_c_mask = ir_c_mask.view(ir_c_mask.size(0), -1)
    ir_c_mask_l = self.mlp_1(ir_c_mask)
    ir_c_mask_l = self.mlp_2(ir_c_mask_l)
    ir_c_mask_l = self.sig(ir_c_mask_l)
    ir_c_mask_l = ir_c_mask_l.unsqueeze(2)
    ir_c_mask_l = ir_c_mask_l.unsqueeze(3)
    ir_feat = ir * ir_c_mask_l

    vis_sig = self.sig(vis)
    vis = torch.cat((vis, vis_sig), dim=1)
    vis_c = self.vis_conv(vis)
    vis_c_avg = F.avg_pool2d(vis_c, kernel_size=vis_c.size()[2:])
    vis_c_max = F.max_pool2d(vis_c, kernel_size=vis_c.size()[2:])
    vis_c_mask = vis_c_avg + vis_c_max
    vis_c_mask = vis_c_mask.view(vis_c_mask.size(0), -1)
    vis_c_mask_l = self.mlp_1(vis_c_mask)
    vis_c_mask_l = self.mlp_2(vis_c_mask_l)
    vis_c_mask_l = self.sig(vis_c_mask_l)
    vis_c_mask_l = vis_c_mask_l.unsqueeze(2)
    vis_c_mask_l = vis_c_mask_l.unsqueeze(3)
    vis_feat = vis * vis_c_mask_l

    max = vis_c_max + ir_c_max
    max = max.view(max.size(0), -1)
    max_c_mask_l = self.mlp_1(max)
    max_c_mask_l = self.mlp_2(max_c_mask_l)
    max_sig_l = self.sig(max_c_mask_l)
    max_sig_l = max_sig_l.unsqueeze(2)
    max_sig_l = max_sig_l.unsqueeze(3)
    ir_feat_m = ir_feat + ir * max_sig_l
    vis_feat_m = vis_feat + vis * max_sig_l

    fused_feat = ir_feat_m + vis_feat_m

    return fused_feat, ir_c_mask, vis_c_mask


class basicfNet(nn.Module):
  def __init__(self, nfeats=64):
    super(basicfNet, self).__init__()

    # body-fuse
    self.fuse = FuseModule_max()
    self.fuse_res = nn.Conv2d(128, nfeats, kernel_size=3, stride=1, padding=1)

    # tail
    self.out_conv_1 = nn.Conv2d(nfeats, 32, kernel_size=3, stride=1, padding=1)
    self.out_conv_2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
    self.out_conv_3 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    self.act1 = nn.LeakyReLU()
    self.act2 = nn.Tanh()
    self.act = nn.Tanh()

  def forward(self, irb_dfeats, visb_dfeats):

    fuse_feats, ir_c_mask, vis_c_mask = self.fuse(irb_dfeats, visb_dfeats)

    fuse_feats = self.fuse_res(fuse_feats)


    out = self.out_conv_1(fuse_feats)
    out = self.act1(out)
    out = self.out_conv_2(out)
    out = self.act1(out)
    out = self.out_conv_3(out)
    out = self.act2(out)


    return fuse_feats, out, ir_c_mask, vis_c_mask


