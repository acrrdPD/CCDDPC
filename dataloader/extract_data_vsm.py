import pathlib

import cv2
import numpy as np
import kornia.utils
import torch.utils.data
import torchvision.transforms.functional
from PIL import Image
import torchvision.transforms as transforms

import torchvision.transforms.functional as TF
from util.utils import randflow, randrot, randfilp
import torch.nn.functional as F

class ExtractDataVSM(torch.utils.data.Dataset):
    """
    Load dataset with ir folder path and vis folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vis_folder: pathlib.Path, ir_reverse_folder: pathlib.Path, vis_reverse_folder: pathlib.Path):
        super(ExtractDataVSM, self).__init__()
        # self.crop = crop
        # gain ir and vis images list
        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vis_list = [x for x in sorted(vis_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

        self.ir_reverse_list = [x for x in sorted(ir_reverse_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vis_reverse_list = [x for x in sorted(vis_reverse_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vis_path = self.vis_list[index]

        ir_reverse_path = self.ir_reverse_list[index]
        vis_reverse_path = self.vis_reverse_list[index]

        assert ir_path.name == vis_path.name, f"Mismatch ir:{ir_path.name} vis:{vis_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vis = self.imread(path=vis_path, flags=cv2.IMREAD_GRAYSCALE)

        ir_reverse = self.imread(path=ir_reverse_path, flags=cv2.IMREAD_GRAYSCALE)
        vis_reverse = self.imread(path=vis_reverse_path, flags=cv2.IMREAD_GRAYSCALE)


        return (ir, vis), (str(ir_path), str(vis_path)), (ir_reverse, vis_reverse), (str(ir_reverse_path), str(vis_reverse_path))

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts


class ExtractTestData(torch.utils.data.Dataset):
    """
    Load dataset with ir folder path and vis folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vis_folder: pathlib.Path):
        super(ExtractTestData, self).__init__()
        # gain ir and vis images list

        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.tif']]
        self.vis_list = [x for x in sorted(vis_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.tif']]

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vis_path = self.vis_list[index]

        assert ir_path.name == vis_path.name, f"Mismatch ir:{ir_path.name} vis:{vis_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vis = self.imread(path=vis_path, flags=cv2.IMREAD_ANYCOLOR)
        # vis = self.imread(path=vis_path, flags=cv2.IMREAD_GRAYSCALE)


        return (ir, vis), (str(ir_path), str(vis_path))

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts


class ExtractTestData_gener(torch.utils.data.Dataset):
    """
    Load dataset with ir folder path and vis folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vis_folder: pathlib.Path):
        super(ExtractTestData_gener, self).__init__()
        # gain ir and vis images list

        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.tif']]
        self.vis_list = [x for x in sorted(vis_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.tif']]

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vis_path = self.vis_list[index]

        assert ir_path.name == vis_path.name, f"Mismatch ir:{ir_path.name} vis:{vis_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        # vis = self.imread(path=vis_path, flags=cv2.IMREAD_ANYCOLOR)
        vis = self.imread(path=vis_path, flags=cv2.IMREAD_GRAYSCALE)

        # ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        # vis = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)


        return (ir, vis), (str(ir_path), str(vis_path))

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts


class ExtractTestData_inten(torch.utils.data.Dataset):
    """
    Load dataset with ir folder path and vis folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vis_folder: pathlib.Path):
        super(ExtractTestData_inten, self).__init__()
        # gain ir and vis images list

        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.tif']]
        self.vis_list = [x for x in sorted(vis_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.tif']]

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vis_path = self.vis_list[index]

        assert ir_path.name == vis_path.name, f"Mismatch ir:{ir_path.name} vis:{vis_path.name}."

        # read image as type Tensor
        ir = (torch.from_numpy(np.array(Image.open(ir_path).convert('L'))) / 255.)
        vis = (torch.from_numpy(np.array(Image.open(vis_path).convert('L'))) / 255.)

        # ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        # vis = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)


        return (ir, vis), (str(ir_path), str(vis_path))

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts


class ExtractTestData_grad(torch.utils.data.Dataset):
    """
    Load dataset with ir folder path and vis folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vis_folder: pathlib.Path):
        super(ExtractTestData_grad, self).__init__()
        # gain ir and vis images list

        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.tif']]
        self.vis_list = [x for x in sorted(vis_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.tif']]

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vis_path = self.vis_list[index]

        assert ir_path.name == vis_path.name, f"Mismatch ir:{ir_path.name} vis:{vis_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_ANYCOLOR)
        # vis = self.imread(path=vis_path, flags=cv2.IMREAD_ANYCOLOR)
        vis = self.imread(path=vis_path, flags=cv2.IMREAD_ANYCOLOR)


        return (ir, vis), (str(ir_path), str(vis_path))

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts


class ExtractTestData_test(torch.utils.data.Dataset):
    """
    Load dataset with ir folder path and vis folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vis_folder: pathlib.Path, grid_folder:pathlib.Path):
        super(ExtractTestData_test, self).__init__()
        # gain ir and vis images list

        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.tif']]
        self.vis_list = [x for x in sorted(vis_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.tif']]
        self.grid_list = [x for x in sorted(grid_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.tif']]


    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vis_path = self.vis_list[index]
        grid_path = self.grid_list[0]

        assert ir_path.name == vis_path.name, f"Mismatch ir:{ir_path.name} vis:{vis_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        # vis = self.imread(path=vis_path, flags=cv2.IMREAD_ANYCOLOR)
        vis = self.imread(path=vis_path, flags=cv2.IMREAD_GRAYSCALE)
        grid = self.imread(path=grid_path, flags=cv2.IMREAD_ANYCOLOR)



        return (ir, vis, grid), (str(ir_path), str(vis_path), str(grid_path))

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts


class ExtractTestData_same(torch.utils.data.Dataset):
    """
    Load dataset with ir folder path and vis folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vis_folder: pathlib.Path, ir_warp_folder: pathlib.Path, vis_warp_folder: pathlib.Path):
        super(ExtractTestData_same, self).__init__()
        # gain ir and vis images list

        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.tif']]
        self.ir_warp_list = [x for x in sorted(ir_warp_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.tif']]
        self.vis_list = [x for x in sorted(vis_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.tif']]
        self.vis_warp_list = [x for x in sorted(vis_warp_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.tif']]

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vis_path = self.vis_list[index]
        ir_warp_path = self.ir_warp_list[index]
        vis_warp_path = self.vis_warp_list[index]

        assert ir_path.name == vis_path.name, f"Mismatch ir:{ir_path.name} vis:{vis_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vis = self.imread(path=vis_path, flags=cv2.IMREAD_GRAYSCALE)
        ir_warp = self.imread(path=ir_warp_path, flags=cv2.IMREAD_GRAYSCALE)
        vis_warp = self.imread(path=vis_warp_path, flags=cv2.IMREAD_GRAYSCALE)


        return (ir, vis), (str(ir_path), str(vis_path)), (ir_warp, vis_warp), (str(ir_warp_path), str(vis_warp_path))

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts


class ExtractTestData_SF(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path):
        super(ExtractTestData_SF, self).__init__()
        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.JPG']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp', '.JPG']]
        self.crop = torchvision.transforms.RandomCrop(224)
        # gain infrared and visible images list

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]

        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vis = self.imread(path=vi_path, flags=cv2.IMREAD_GRAYSCALE)

        vis_ir = torch.cat([vis,ir],dim=1)
        if vis_ir.shape[-1]<=224 or vis_ir.shape[-2]<=224:
            vis_ir=TF.resize(vis_ir,224)

        # flow,disp,_ = randflow(vis_ir,10,0.1,1)  #origin
        flow,disp,_ = randflow(vis_ir,1,0.1,1)  #my train
        # flow,disp,_ = randflow(vis_ir,1,0.01,1)  #test
        vis_ir_warped = F.grid_sample(vis_ir, flow, align_corners=False, mode='bilinear')
        patch = torch.cat([vis_ir, vis_ir_warped, disp.permute(0, 3, 1, 2)], dim=1)

        vis, ir, vis_warped, ir_warped, disp = torch.split(patch, [3,3,3,3,2], dim=1)
        h, w = vis_ir.shape[2],vis_ir.shape[3]
        scale = (torch.FloatTensor([w,h]).unsqueeze(0).unsqueeze(0)-1)/(self.crop.size[0]*1.0-1)
        #print(self.crop.size[0])
        disp = disp.permute(0,2,3,1)*scale
        #vis_warped_ = self.ST(vis.unsqueeze(0),disp.unsqueeze(0))
        #TF.to_pil_image(((vis_warped-vis_warped_).abs()).squeeze(0)).save('error.png')
        #disp_crop = disp[:,h//2-150:h//2+150,w//2-150:w//2+150,:]*scale
        return (ir, vis, ir_warped, vis_warped, disp), (str(ir_path), str(vi_path))

    def __len__(self):
        return len(self.vi_list)


    @staticmethod
    def imread(path, flags=cv2.IMREAD_GRAYSCALE):
        # im_cv = cv2.imread(str(path), flags)
        # assert im_cv is not None, f"Image {str(path)} is invalid."
        # im_ts = kornia.utils.image_to_tensor(im_cv / 255.,keepdim=False).type(torch.FloatTensor)
        img = Image.open(path).convert('RGB')
        im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts


class ExtractTestData_SF0(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path):
        super(ExtractTestData_SF0, self).__init__()
        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.crop = torchvision.transforms.RandomCrop(224)
        # gain infrared and visible images list

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]

        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vis = self.imread(path=vi_path, flags=cv2.IMREAD_GRAYSCALE)

        vis_ir = torch.cat([vis,ir],dim=1)
        if vis_ir.shape[-1]<=224 or vis_ir.shape[-2]<=224:
            vis_ir=TF.resize(vis_ir,224)

        # flow,disp,_ = randflow(vis_ir,10,0.1,1)  #origin
        flow,disp,_ = randflow(vis_ir,0.2,0.01,1)  #my train
        # flow,disp,_ = randflow(vis_ir,1,0.01,1)  #test
        vis_ir_warped = F.grid_sample(vis_ir, flow, align_corners=False, mode='bilinear')
        patch = torch.cat([vis_ir, vis_ir_warped, disp.permute(0, 3, 1, 2)], dim=1)

        vis, ir, vis_warped, ir_warped, disp = torch.split(patch, [3,3,3,3,2], dim=1)
        h, w = vis_ir.shape[2],vis_ir.shape[3]
        scale = (torch.FloatTensor([w,h]).unsqueeze(0).unsqueeze(0)-1)/(self.crop.size[0]*1.0-1)
        #print(self.crop.size[0])
        disp = disp.permute(0,2,3,1)*scale
        #vis_warped_ = self.ST(vis.unsqueeze(0),disp.unsqueeze(0))
        #TF.to_pil_image(((vis_warped-vis_warped_).abs()).squeeze(0)).save('error.png')
        #disp_crop = disp[:,h//2-150:h//2+150,w//2-150:w//2+150,:]*scale
        return (ir, vis, ir_warped, vis_warped, disp), (str(ir_path), str(vi_path))

    def __len__(self):
        return len(self.vi_list)


    @staticmethod
    def imread(path, flags=cv2.IMREAD_GRAYSCALE):
        # im_cv = cv2.imread(str(path), flags)
        # assert im_cv is not None, f"Image {str(path)} is invalid."
        # im_ts = kornia.utils.image_to_tensor(im_cv / 255.,keepdim=False).type(torch.FloatTensor)
        img = Image.open(path).convert('RGB')
        im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts
