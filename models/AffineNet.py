import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange

from einops import rearrange
from models.util import MaskedAutoencoderViT_64
from torchvision.ops import DeformConv2d

from torch.autograd import Variable

gpu_use = True
shape = (256, 256)


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

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Channel_Max_Pooling(torch.nn.MaxPool1d):
    def __init__(self, channels, isize):
        super(Channel_Max_Pooling, self).__init__(channels)
        self.kernel_size = channels
        self.stride = isize

    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = torch.nn.functional.max_pool1d(input, self.kernel_size, self.stride,
                                                self.padding, self.dilation, self.ceil_mode,
                                                self.return_indices)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        pooled = rearrange(pooled, 'b c (h w) -> b c h w', h=256, c=1)
        return pooled

class Channel_Avg_Pooling(torch.nn.MaxPool1d):
    def __init__(self, channels, isize):
        super(Channel_Avg_Pooling, self).__init__(channels)
        self.kernel_size = channels
        self.stride = isize

    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = torch.nn.functional.avg_pool1d(input, self.kernel_size, self.stride)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        pooled = rearrange(pooled, 'b c (h w) -> b c h w', h=256, c=1)
        return pooled

#通道
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attn = self.softmax(out)
        out = x * attn + x
        return out


# def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, reuse=False):
#     if pad > 0:
#         if (kernel - stride) % 2 == 0:
#             pad_top = pad
#             pad_bottom = pad
#             pad_left = pad
#             pad_right = pad
#         else:
#             pad_top = pad
#             pad_bottom = kernel - stride - pad_top
#             pad_left = pad
#             pad_right = kernel - stride - pad_left
#
#         if pad_type == 'zero':
#             x = nn.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
#         if pad_type == 'reflect':
#             x = nn.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')
# 	x = nn.layers.conv2d(inputs=x, filters=channels,
# 							 kernel_size=kernel, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05),
# 							 kernel_regularizer=None,
# 							 strides=stride, use_bias=use_bias, reuse=reuse)
# 	return x


def convoffset2D(x, channel):
    x_shape = x.shape
    # x_shape_list = x.get_shape().as_list()
    x_shape_list = x.shape
    # channel = x_shape_list[-1]

    conv_off = nn.Conv2d(channel, channel * 2, kernel_size=3, stride=1, padding=1, bias=False)
    offsets = conv_off(x)
    conv_weights = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
    weights = torch.nn.Sigmoid(conv_weights(x))

    x = to_bc_h_w(x, x_shape)
    offsets = to_bc_h_w_2(offsets, x_shape)
    weights = to_bc_h_w(weights, x_shape)
    x_offset = tf_batch_map_offsets(x, offsets)
    weights = torch.unsqueeze(weights, dim=1)
    weights = to_b_h_w_c(weights, x_shape)
    x_offset = to_b_h_w_c(x_offset, x_shape)
    x_offset = x_offset * weights
    x_offset.set_shape(x_shape_list)

    return x_offset


def apply_affine_trans(img, dtheta):
    "img is a RGB image"
    batchsize = img.shape[0]
    oh = img.shape[1]
    ow = img.shape[2]
    x = torch.linspace(-1.0, 1.0, ow)
    y = torch.linspace(-1.0, 1.0, oh)

    identity_theta = torch.tensor(np.array([1, 0, 0, 0, 1, 0], dtype=np.float32))
    identity_theta = identity_theta.unsqueeze(dim=0)
    identity_theta = identity_theta.repeat(batchsize, 1)

    affine_matrix = torch.tensor(identity_theta + dtheta, dtype=torch.float32)
    num_batch = batchsize
    affine_matrix = torch.reshape(affine_matrix, [num_batch, 2, 3])

    x_t, y_t = torch.meshgrid(x, y)
    # flatten
    x_t_flat = torch.reshape(x_t, [-1])
    y_t_flat = torch.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = torch.ones_like(x_t_flat)
    sampling_grid = torch.stack([x_t_flat, y_t_flat, ones])
    sampling_grid = torch.unsqueeze(sampling_grid, dim=0)
    sampling_grid = torch.repeat(sampling_grid, torch.stack([num_batch, 1, 1]))
    sampling_grid = torch.tensor(sampling_grid, dtype=torch.float32)

    # transform the sampling grid - batch multiply
    batch_grids = torch.mm(affine_matrix, sampling_grid)
    batch_grids = torch.reshape(batch_grids, [num_batch, 2, int(oh), int(ow)])
    resampling_grid = torch.transpose(batch_grids, (0, 2, 3, 1))

    xx, yy = torch.meshgrid(y, x)
    xx = torch.transpose(xx)
    yy = torch.transpose(yy)
    xx = torch.unsqueeze(xx, dim=-1)
    yy = torch.unsqueeze(yy, dim=-1)
    xx = torch.unsqueeze(xx, dim=0)
    yy = torch.unsqueeze(yy, dim=0)
    identity = torch.concat([yy, xx], dim=-1)
    defor = resampling_grid - identity


    if batchsize == 1:
        warped_R, _, _, label = grid_sample(img[0:1, :, :, 0:1], resampling_grid)
        warped_G, _, _, _ = grid_sample(img[0:1, :, :, 1:2], resampling_grid)
        warped_B, _, _, _ = grid_sample(img[0:1, :, :, 2:3], resampling_grid)
    else:
        warped_R, _, _, label = grid_sample(img[:, :, :, 0:1], resampling_grid)
        warped_G, _, _, label = grid_sample(img[:, :, :, 1:2], resampling_grid)
        warped_B, _, _, label = grid_sample(img[:, :, :, 2:3], resampling_grid)
    warped_img = torch.concat([warped_R, warped_G, warped_B], dim=-1)

    return warped_img, label, resampling_grid

def grid_sample(input, grid):
    in_shape = input.shape
    IH = int(in_shape[1])
    IW = int(in_shape[2])


    # out_shape = grid.shape
    # OH = out_shape[1]
    # OW = out_shape[2]

    nor_ix = grid[:, :, :, 0]
    nor_iy = grid[:, :, :, 1]
    ix = ((nor_ix + 1) / 2) * (IW - 1)
    iy = ((nor_iy + 1) / 2) * (IH - 1)
    out, label = bilinear_sampler(input, ix, iy)

    return out, ix, iy, label

def gather_nd(params, indices):
    ''' 4D example params: tensor shaped [n_1, n_2, n_3, n_4] --> 4 dimensional indices: tensor shaped [m_1, m_2, m_3, m_4, 4] --> multidimensional list of 4D indices returns: tensor shaped [m_1, m_2, m_3, m_4] ND_example params: tensor shaped [n_1, ..., n_p] --> d-dimensional tensor indices: tensor shaped [m_1, ..., m_i, d] --> multidimensional list of d-dimensional indices returns: tensor shaped [m_1, ..., m_1] '''
    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1) # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1
    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)
    out = torch.take(params, idx)
    return out.view(out_shape)

def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = x.shape
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = torch.arange(0, batch_size)
    batch_idx = torch.reshape(batch_idx, (batch_size, 1, 1))
    b = torch.repeat(batch_idx, (1, height, width))

    indices = torch.stack([b, y, x], 3)

    return gather_nd(img, indices)

def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    # H = torch.shape(img)[1]
    # W = torch.shape(img)[2]
    H = img.shape[1]
    W = img.shape[2]
    max_y = torch.tensor(H - 1, dtype=torch.int32)
    max_x = torch.tensor(W - 1, dtype=torch.int32)
    zero = torch.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    # x = tf.cast(x, 'float32')
    # y = tf.cast(y, 'float32')
    # x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    # y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = torch.tensor(torch.floor(x), dtype=torch.int32)
    x1 = x0 + 1
    y0 = torch.tensor(torch.floor(y), dtype=torch.int32)
    y1 = y0 + 1

    labelx_min = torch.where(x < torch.tensor(zero, dtype=torch.float32), x = torch.zeros_like(x), y = torch.ones_like(x))
    labelx_max = torch.where(x > torch.tensor(max_x, dtype=torch.float32), x = torch.zeros_like(x), y = torch.ones_like(x))
    labelx = labelx_min * labelx_max
    labely_min = torch.where(y < torch.tensor(zero, dtype=torch.float32), x = torch.zeros_like(y), y = torch.ones_like(y))
    labely_max = torch.where(y > torch.tensor(max_y, dtype=torch.float32), x = torch.zeros_like(y), y = torch.ones_like(y))
    labely = labely_min * labely_max
    label = torch.unsqueeze((labelx * labely), dim=-1)

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = torch.clamp(x0, zero, max_x)
    x1 = torch.clamp(x1, zero, max_x)
    y0 = torch.clamp(y0, zero, max_y)
    y1 = torch.clamp(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = torch.tensor(x0, dtype=torch.float32)
    x1 = torch.tensor(x1, dtype=torch.float32)
    y0 = torch.tensor(y0, dtype=torch.float32)
    y1 = torch.tensor(y1, dtype=torch.float32)

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = torch.unsqueeze(wa, dim=3)
    wb = torch.unsqueeze(wb, dim=3)
    wc = torch.unsqueeze(wc, dim=3)
    wd = torch.unsqueeze(wd, dim=3)
    w = torch.concat([wa, wb, wc, wd], dim=-1)
    w = nn.Softmax(w/0.05)
    wa = torch.unsqueeze(w[:, :, :, 0], dim=-1)
    wb = torch.unsqueeze(w[:, :, :, 1], dim=-1)
    wc = torch.unsqueeze(w[:, :, :, 2], dim=-1)
    wd = torch.unsqueeze(w[:, :, :, 3], dim=-1)

    # compute output
    out = wa*Ia + wb*Ib + wc*Ic + wd*Id
    out = out * label
    return out, label

class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, volsize, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        size = volsize
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids) # y, x, z
        grid = torch.unsqueeze(grid, 0)  #add batch
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """

        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...].clone() / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode='border', align_corners=True), new_locs


# def multi_affine(dtheta1, dtheta2):
# 	batchsize = dtheta1.shape[0]
# 	identity_theta = tf.constant(np.array([1, 0, 0, 0, 1, 0], dtype=np.float32))
# 	identity_theta = tf.expand_dims(identity_theta, axis=0)
# 	identity_theta = tf.tile(identity_theta, [batchsize, 1])
#
# 	zeros = tf.constant(np.array([0, 0, 0], dtype=np.float32))
# 	zeros = tf.expand_dims(tf.expand_dims(zeros, axis=0), axis=0)
# 	zeros = tf.tile(zeros, [batchsize, 1, 1])
# 	affine_matrix1 = tf.cast(identity_theta + dtheta1, 'float32')
# 	affine_matrix1 = tf.reshape(affine_matrix1, [batchsize, 2, 3])
# 	affine_matrix1 = tf.concat([affine_matrix1, zeros], axis=1)
# 	affine_matrix2 = tf.cast(identity_theta + dtheta2, 'float32')
# 	affine_matrix2 = tf.reshape(affine_matrix2, [batchsize, 2, 3])
# 	affine_matrix2 = tf.concat([affine_matrix2, zeros], axis=1)
#
# 	matrix = tf.matmul(affine_matrix2, affine_matrix1)
#
# 	return tf.reshape(matrix[:, 0:2, :], [batchsize, 6]) - identity_theta

def reduce_std(x, dim=None, keepdims=False):
    return math.sqrt(reduce_var(x, dim=dim, keepdims=keepdims))

def reduce_var(x, dim=None, keepdims=False):
    m = torch.mean(x, dim=dim, keep_dims=True)
    devs_squared = torch.pow(x - m)
    return torch.mean(devs_squared, dim=dim, keep_dims=keepdims)

def NCC(img1, img2):
    h = img1.shape[1]
    w = img2.shape[2]
    mean1 = torch.mean(img1, dim=[1,2])
    mean2 = torch.mean(img2, dim=[1,2])
    mean1= torch.unsqueeze(mean1,dim=1)
    mean1 = torch.unsqueeze(mean1, dim = 2)
    mean2= torch.unsqueeze(mean2,dim=1)
    mean2 = torch.unsqueeze(mean2, dim = 2)
    mean1 = torch.repeat(mean1, [1, int(h), int(w), 1])
    mean2 = torch.repeat(mean2, [1, int(h), int(w), 1])
    dimg1 = img1-mean1
    dimg2 = img2-mean2
    numerator = dimg1 * dimg2

    std1 = reduce_std(img1, dim=[1,2], keepdims=True)
    std1 = torch.repeat(std1, [1, int(h), int(w), 1])
    std2 = reduce_std(img2, dim=[1, 2], keepdims=True)
    std2 = torch.repeat(std2, [1, int(h), int(w), 1])
    denominator = std1 * std2

    return torch.mean(numerator / denominator)


def to_bc_h_w_2(x, x_shape):
    # """(b, h, w, 2c) -> (b*c, h, w, 2)"""
    """(b, h, w, 2c) -> (b*c, h, w, 2)"""
    # x = x.permute(0, 3, 1, 2)
    # x = x.reshape(x_shape[0], x_shape[3], 2, x_shape[1], x_shape[2])
    # x = x.permute(0, 1, 3, 4, 2)
    print('hhhh',x.shape)
    x = x.reshape(-1, x_shape[1], x_shape[1], 2)
    print('hhhh',x.shape)
    # x = x.reshape(-1, x_shape[2], x_shape[3], 2)
    return x


def to_bc_h_w(x, x_shape):
    # """(b, h, w, c) -> (b*c, h, w)"""
    """(b, c, h, w) -> (b*c, h, w)"""
    # x = x.permute(0, 3, 1, 2)
    x = x.reshape(-1, x_shape[2], x_shape[3])
    return x

def to_b_h_w_c(x, x_shape):
    """(b*c, h, w) -> (b, h, w, c)"""
    x = torch.reshape(x, (-1, x_shape[3], x_shape[1], x_shape[2]))
    x = torch.transpose(x, [0, 2, 3, 1])
    return x

def tf_flatten(a):
    """Flatten tensor"""
    return torch.reshape(a, [-1])

def tf_repeat(a, repeats, axis=0):
    a = torch.unsqueeze(a, -1)
    a = torch.tile(a, [1, repeats])
    a = tf_flatten(a)
    return a

def tf_batch_map_offsets(input, offsets):
    """:param input: tf.Tensor, shape=(b, h, w)
	:param offsets: tf.Tensor, shape=(b, h, w, 2)
	:return:
	"""
    # input_shape = torch.shape(input)
    input_shape = input.shape
    batch_size = input_shape[0]
    input_size_h = input_shape[1]
    input_size_w = input_shape[2]
    offsets = torch.reshape(offsets, (batch_size, -1, 2))
    grid_x, grid_y = torch.meshgrid(torch.arange(input_size_w), torch.arange(input_size_h))
    grid = torch.stack([grid_y, grid_x], axis=-1)
    grid = torch.tensor(grid, dtype=torch.float32)
    grid = torch.reshape(grid, (-1, 2))
    grid = torch.unsqueeze(grid, dim=0)
    grid = torch.repeat(grid, multiples=[batch_size, 1, 1])
    coords = offsets + grid
    mapped_vals = tf_batch_map_coordinates(input, coords)
    return mapped_vals

def tf_batch_map_coordinates(input, coords):
    """
    Batch version of tf_map_coordinates
    :param input: tf.Tensor. shape = (b, h, w)
    :param coords: tf.Tensor. shape = (b, n_points, 2)
    :return:
    """
    # input_shape = torch.shape(input)
    input_shape = input.shape
    batch_size = input_shape[0]
    input_size_h = input_shape[1]
    input_size_w = input_shape[2]
    # n_coords = torch.shape(coords)[1]
    n_coords = coords.shape[1]
    coords_w = torch.clamp(coords[..., 1], 0, torch.tensor(input_size_w, dtype=torch.float32) - 1)
    coords_h = torch.clamp(coords[..., 0], 0, torch.tensor(input_size_h, dtype=torch.float32) - 1)
    coords = torch.stack([coords_h, coords_w], axis=-1)
    coords_tl = torch.tensor(torch.floor(coords), dtype=torch.int32)
    coords_br = torch.tensor(torch.ceil(coords), dtype=torch.int32)
    coords_bl = torch.stack([coords_br[..., 0], coords_tl[..., 1]], axis=-1)
    coords_tr = torch.stack([coords_tl[..., 0], coords_br[..., 1]], axis=-1)
    idx = tf_repeat(torch.arange(batch_size), n_coords)

    def _get_vals_by_coords(input, coords):
        indices = torch.stack([idx, tf_flatten(coords[..., 0]), tf_flatten(coords[..., 1])], axis=-1)
        vals = gather_nd(input, indices)
        vals = torch.reshape(vals, (batch_size, n_coords))
        return vals

    vals_tl = _get_vals_by_coords(input, coords_tl)
    vals_br = _get_vals_by_coords(input, coords_br)
    vals_bl = _get_vals_by_coords(input, coords_bl)
    vals_tr = _get_vals_by_coords(input, coords_tr)

    coords_offset_tl = coords - torch.tensor(coords_tl, dtype=torch.float32)
    vals_t = vals_tl + (vals_tr - vals_tl) * coords_offset_tl[..., 1]
    vals_b = vals_bl + (vals_br - vals_bl) * coords_offset_tl[..., 1]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_tl[..., 0]

    return mapped_vals

class Affine_Generator(nn.Module):
    # def __init__(self, scope_name, is_training):
    def __init__(self):
        super(Affine_Generator, self).__init__()
    #     self.scope = scope_name
    #     self.is_training = is_training

        self.conv1 = nn.Conv2d(128, 16, kernel_size=9, stride=1, padding=4, bias=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        # self.offset1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.offset1 = nn.Conv2d(16, 18, kernel_size=3, stride=1, padding=1, bias=True)
        self.dfconv1 = DeformConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=9, stride=1, padding=4, bias=True)
        self.offset2 = nn.Conv2d(32, 18, kernel_size=3, stride=1, padding=1, bias=True)
        self.dfconv2 = DeformConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=9, stride=1, padding=4, bias=True)
        self.offset3 = nn.Conv2d(64, 18, kernel_size=3, stride=1, padding=1, bias=True)
        self.dfconv3 = DeformConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3, bias=True)

        # self.conv5 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv5 = nn.Conv2d(64, 2, kernel_size=7, stride=1, padding=3, bias=True)

        # self.fc = nn.Linear(128, 6) #11111111111111111
        self.fc = nn.Linear(128, 2)

    def forward(self, img_a, img_b):
        x = torch.cat((img_a, img_b), dim=1)
        # print('x', x.shape)
        feat_1 = self.conv1(x)
        # print('feat_1111111111111111111111111111', feat_1.shape)
        feat_1 = self.act1(feat_1)
        offset1 = self.offset1(feat_1)
        # print('offset1', offset1.shape)
        # x = to_bc_h_w(x, x.shape)
        # print('x_shape111111111111111111111111111', x.shape)
        # offset1 = to_bc_h_w_2(offset1, x.shape)
        # print('offset1111111111111111111111111111', offset1.shape)
        feat_1 = self.dfconv1(feat_1, offset1)
        # feat_1 = convoffset2D(feat_1, channel=16)
        feat_1 = self.act1(feat_1)
        # feat_1 = F.max_pool2d(feat_1, kernel_size=feat_1.size()[2:])

        feat_2 = self.conv2(feat_1)
        feat_2 = self.act1(feat_2)
        offset2 = self.offset2(feat_2)
        feat_2 = self.dfconv2(feat_2, offset2)
        feat_2 = self.act1(feat_2)
        # feat_2 = F.max_pool2d(feat_2, kernel_size=feat_2.size()[2:])

        feat_3 = self.conv3(feat_2)
        feat_3 = self.act1(feat_3)
        offset3 = self.offset3(feat_3)
        feat_3 = self.dfconv3(feat_3, offset3)
        feat_3 = self.act1(feat_3)
        # feat_3 = F.max_pool2d(feat_3, kernel_size=feat_3.size()[2:])

        feat_4 = self.conv4(feat_3)
        feat_4 = self.act1(feat_4)

        feat_5 = self.conv5(feat_4)
        feat_5 = self.act1(feat_5)
        # feat_5 = F.avg_pool2d(feat_5, kernel_size=feat_5.size()[2:])
        # feat_5 = torch.mean(feat_5, [1, 2])

        # affine = self.fc(feat_5)
        # affine = self.act1(affine)
        affine = feat_5

        return affine


class Affine_Model(nn.Module):
    # def __init__(self, scope_name, is_training):
    def __init__(self):
        super(Affine_Model, self).__init__()

        # self.scope = scope_name
        # self.is_training = is_training

        # self.affine_net = Affine_Generator('affine_net', self.is_training)
        self.affine_net = Affine_Generator()

        self.conv1 = nn.Conv2d(512, 16, kernel_size=9, stride=1, padding=4, bias=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.dfconv1 = DeformConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=9, stride=1, padding=4, bias=True)
        self.dfconv2 = DeformConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=9, stride=1, padding=4, bias=True)
        self.dfconv3 = DeformConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3, bias=True)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3, bias=True)

        self.fc = nn.Linear(128, 6)

        self.inshape = shape
        self.spatial_transform = SpatialTransformer(volsize=self.inshape)

    def forward(self, transfer_ir, transfer_vis, ir_feat, vis_feat):
        # dtheta = self.affine_net.field_model(transfer_vis, transfer_ir)
        dtheta = self.affine_net(transfer_vis, transfer_ir)
        align_vis_feat, label_vis, resampling_grid_vis = apply_affine_trans(vis_feat, dtheta)
        align_vis_feat = align_vis_feat * label_vis.repeat(1, 1, 1, 3)
        # m_warp, disp_pre = self.spatial_transform(transfer_vis, dtheta) # torch.Size([16, 1, 256, 256]) torch.Size([16, 256, 256, 2])
        # f_warp, _ = self.spatial_transform(tgt, (-dtheta))  # torch.Size([16, 1, 256, 256]) torch.Size([16, 256, 256, 2])

        # warped_vis_feat, _, _, _ = grid_sample(vis_feat, resampling_grid) #11111111111111111111
        # align_vis, label_vis, resampling_grid_vis = apply_affine_trans(transfer_vis, dtheta)
        align_ir_feat, label_ir, resampling_grid_ir = apply_affine_trans(ir_feat, -dtheta)

        w_des1 = align_vis_feat * label_vis
        w_des2 = align_ir_feat * label_vis

        # self.d1_loss = -NCC(self.w_des1_d4, self.w_des2_d4)
        # self.loss = self.d1_loss - NCC(self.w_des1, self.w_des2)
        d1_loss = -NCC(w_des1, w_des2)

        # return _, d1_loss, dtheta #111111111111
        return align_ir_feat, align_vis_feat, d1_loss, dtheta
        # return m_warp, f_warp, flow, int_flow1, int_flow2, disp_pre

    def get_identity_grid(self):
        """Returns a sampling-grid that represents the identity transformation."""
        x = torch.repeat(-1.0, 1.0, self.ow) #
        y = torch.repeat(-1.0, 1.0, self.oh)  #
        xx, yy = torch.meshgrid([y, x])  #
        xx = torch.transpose(xx)
        yy = torch.transpose(yy)
        xx = xx.unsqueeze(dim=0)
        yy = yy.unsqueeze(dim=0)
        identity = torch.concat((yy, xx), dim=0).unsqueeze(0)  #

        return identity


class Affine_Model_warp(nn.Module):
    # def __init__(self, scope_name, is_training):
    def __init__(self):
        super(Affine_Model_warp, self).__init__()

        # self.scope = scope_name
        # self.is_training = is_training

        # self.affine_net = Affine_Generator('affine_net', self.is_training)
        self.affine_net = Affine_Generator()

        self.conv1 = nn.Conv2d(512, 16, kernel_size=9, stride=1, padding=4, bias=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.dfconv1 = DeformConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=9, stride=1, padding=4, bias=True)
        self.dfconv2 = DeformConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=9, stride=1, padding=4, bias=True)
        self.dfconv3 = DeformConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3, bias=True)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3, bias=True)

        self.fc = nn.Linear(128, 6)

        self.inshape = shape
        self.spatial_transform = SpatialTransformer(volsize=self.inshape)

        self.warp = Warper2d()

    def forward(self, transfer_ir, transfer_vis, ir_feat_un, vis_feat):
        # dtheta = self.affine_net.field_model(transfer_vis, transfer_ir)
        dtheta = self.affine_net(transfer_ir, transfer_vis)
        align_ir_feat = self.warp(dtheta, ir_feat_un)

        # return _, d1_loss, dtheta #111111111111
        return align_ir_feat, dtheta
        # return m_warp, f_warp, flow, int_flow1, int_flow2, disp_pre

    def get_identity_grid(self):
        """Returns a sampling-grid that represents the identity transformation."""
        x = torch.repeat(-1.0, 1.0, self.ow) #
        y = torch.repeat(-1.0, 1.0, self.oh)  #
        xx, yy = torch.meshgrid([y, x])  #
        xx = torch.transpose(xx)
        yy = torch.transpose(yy)
        xx = xx.unsqueeze(dim=0)
        yy = yy.unsqueeze(dim=0)
        identity = torch.concat((yy, xx), dim=0).unsqueeze(0)  #

        return identity


class Warper2d(nn.Module):
    def __init__(self):
        super(Warper2d, self).__init__()

        """
        warp an image/tensor (im2) back to im1, according to the optical flow
#        img_src: [B, 1, H1, W1] (source image used for prediction, size 32)
        img_smp: [B, 1, H2, W2] (image for sampling, size 44)
        flow: [B, 2, H1, W1] flow predicted from source image pair
        """

    def forward(self, flow, img):
        H, W = flow.size()[2], flow.size()[3]

        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, H, W)
        yy = yy.view(1, H, W)
        grid = torch.cat((xx, yy), 0).float()  # [2, H, W]

        device = img.device

        if img.is_cuda:
            grid = grid.to(device)

        vgrid = Variable(grid, requires_grad=False) + flow
        vgrid[:, 0] = 2.0 * vgrid[:, 0] / (H - 1) - 1.0  # max(W-1,1)
        vgrid[:, 1] = 2.0 * vgrid[:, 1] / (W - 1) - 1.0  # max(H-1,1)

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(img, vgrid)

        return output  # *mask