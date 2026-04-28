import pathlib

import cv2
import numpy as np
import kornia.utils
import torch.utils.data
import torchvision.transforms.functional
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from dataloader.reg_data import ImageTransform_1, ImageTransform_vari
from dataloader.reg_data import Warper2d, warp2D

class FuseDataVSM(torch.utils.data.Dataset):
    """
    Load dataset with ir folder path and vis folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path, ir_map: pathlib.Path, vi_map: pathlib.Path):
        super(FuseDataVSM, self).__init__()
        # self.crop = crop
        # gain ir and vis images list
        self.ps = 224

        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

        self.ir_map_list = [x for x in sorted(ir_map.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_map_list = [x for x in sorted(vi_map.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    def get_patch(self, ir, vis, ir_map, vis_map):
        H, W = ir.shape[1], ir.shape[2]

        #1.
        x, y = np.random.randint(10, H-10-self.ps+1), np.random.randint(10, W-10-self.ps+1)
        ir_crop = ir[:, x:x+self.ps, y:y+self.ps]
        vis_crop = vis[:, x:x+self.ps, y:y+self.ps]
        ir_map_crop = ir_map[:, x:x+self.ps, y:y+self.ps]
        vis_map_crop = vis_map[:, x:x+self.ps, y:y+self.ps]


        return ir_crop, vis_crop, ir_map_crop, vis_map_crop

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]

        ir_map_path = self.ir_map_list[index]
        vi_map_path = self.vi_map_list[index]

        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vi = self.imread(path=vi_path, flags=cv2.IMREAD_GRAYSCALE)


        ir_map = self.imread(path=ir_map_path, flags=cv2.IMREAD_GRAYSCALE)
        vi_map = self.imread(path=vi_map_path, flags=cv2.IMREAD_GRAYSCALE)


        ir_crop, vis_crop, ir_map_crop, vis_map_crop = self.get_patch(ir, vi, ir_map, vi_map)

        # return ir_crop, vis_crop, ir_reverse_crop, vis_reverse_crop, ir_map_crop, vis_map_crop

        # crop same patch
        # patch = torch.cat([ir, vi, ir_reverse, vi_reverse, ir_map, vi_map], dim=0)
        # patch = torchvision.transforms.functional.to_pil_image(patch)
        # patch = self.crop(patch)
        # patch = torchvision.transforms.functional.to_tensor(patch)
        # ir, vi, ir_reverse, vi_reverse, ir_map, vi_map = torch.chunk(patch, 6, dim=0)

        return (ir_crop, vis_crop), (str(ir_path), str(vi_path)), (ir_map_crop, vis_map_crop)

    # def __getitem__(self, index):
    #     # gain image path
    #     ir_path = self.ir_list[index]
    #     vi_path = self.vi_list[index]
    #
    #     ir_map_path = self.ir_map_list[index]
    #     vi_map_path = self.vi_map_list[index]
    #
    #     assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."
    #
    #     # read image as type Tensor
    #     ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
    #     vi = self.imread(path=vi_path, flags=cv2.IMREAD_GRAYSCALE)
    #
    #     ir_map = self.imread(path=ir_map_path, flags=cv2.IMREAD_GRAYSCALE)
    #     vi_map = self.imread(path=vi_map_path, flags=cv2.IMREAD_GRAYSCALE)
    #
    #
    #     return (ir, vi), (str(ir_path), str(vi_path)), (ir_map, vi_map)

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts


class FuseDataVSM_144(torch.utils.data.Dataset):
    """
    Load dataset with ir folder path and vis folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path):
    # def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path, ir_map: pathlib.Path, vi_map: pathlib.Path):
        super(FuseDataVSM_144, self).__init__()
        # self.crop = crop
        # gain ir and vis images list

        self.ps = 144

        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]


        # self.ir_transform = transforms.Resize([256, 256])
        # self.vis_transform = transforms.Resize([256, 256])

    def get_patch(self, ir, vis):
        H, W = ir.shape[1], ir.shape[2]

        # 1.
        x, y = np.random.randint(10, H - 10 - self.ps + 1), np.random.randint(10, W - 10 - self.ps + 1)
        ir_crop = ir[:, x:x + self.ps, y:y + self.ps]
        vis_crop = vis[:, x:x + self.ps, y:y + self.ps]

        return ir_crop, vis_crop

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]

        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vi = self.imread(path=vi_path, flags=cv2.IMREAD_GRAYSCALE)

        ir_crop, vis_crop = self.get_patch(ir, vi)

        # vi = self.vis_transform(vi)
        # ir = self.ir_transform(ir)

        # crop same patch
        # patch = torch.cat([ir, vi], dim=0)
        # patch = torchvision.transforms.functional.to_pil_image(patch)
        # patch = self.crop(patch)
        # patch = torchvision.transforms.functional.to_tensor(patch)
        # ir, vi = torch.chunk(patch, 2, dim=0)


        return (ir_crop, vis_crop), (str(ir_path), str(vi_path))

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts


class FuseDataVSM_224(torch.utils.data.Dataset):
    """
    Load dataset with ir folder path and vis folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path):
        super(FuseDataVSM_224, self).__init__()
        # self.crop = crop
        # gain ir and vis images list
        self.ps = 224

        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

        # self.img_resize = RandomResizedCrop(224)

    # def get_patch(self, ir, vis, ir_map, vis_map):
    #     H, W = ir.shape[1], ir.shape[2]
    #
    #     #1.
    #     x, y = np.random.randint(10, H-10-self.ps+1), np.random.randint(10, W-10-self.ps+1)
    #     ir_crop = ir[:, x:x+self.ps, y:y+self.ps]
    #     vis_crop = vis[:, x:x+self.ps, y:y+self.ps]
    #     ir_map_crop = ir_map[:, x:x+self.ps, y:y+self.ps]
    #     vis_map_crop = vis_map[:, x:x+self.ps, y:y+self.ps]
    #
    #     return ir_crop, vis_crop, ir_map_crop, vis_map_crop

    def get_patch(self, ir, vis):

        H, W = ir.shape[1], ir.shape[2]

        # 1.
        x, y = np.random.randint(10, H - 10 - self.ps + 1), np.random.randint(10, W - 10 - self.ps + 1)
        ir_crop = ir[:, x:x + self.ps, y:y + self.ps]
        vis_crop = vis[:, x:x + self.ps, y:y + self.ps]

        return ir_crop, vis_crop

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]

        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vi = self.imread(path=vi_path, flags=cv2.IMREAD_GRAYSCALE)

        ir_crop, vis_crop = self.get_patch(ir, vi)

        return (ir_crop, vis_crop), (str(ir_path), str(vi_path))

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts


class FuseTestData(torch.utils.data.Dataset):
    """
    Load dataset with ir folder path and vis folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path, ir_map: pathlib.Path, vi_map: pathlib.Path):
        super(FuseTestData, self).__init__()
        # gain ir and vis images list
        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

        self.ir_map_list = [x for x in sorted(ir_map.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_map_list = [x for x in sorted(vi_map.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]

        ir_map_path = self.ir_map_list[index]
        vi_map_path = self.vi_map_list[index]

        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vi = self.imread(path=vi_path, flags=cv2.IMREAD_GRAYSCALE)

        ir_map = self.imread(path=ir_map_path, flags=cv2.IMREAD_GRAYSCALE)
        vi_map = self.imread(path=vi_map_path, flags=cv2.IMREAD_GRAYSCALE)

        return (ir, vi), (str(ir_path), str(vi_path)), (ir_map, vi_map)

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts


class FuseTestData_crop(torch.utils.data.Dataset):
    """
    Load dataset with ir folder path and vis folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path):
        super(FuseTestData_crop, self).__init__()
        # self.crop = crop
        # gain ir and vis images list
        self.ps = 224

        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

        self.image_trans = ImageTransform_vari()
        self.warp = Warper2d()

        # self.img_resize = RandomResizedCrop(224)

    # def get_patch(self, ir, vis, ir_map, vis_map):
    #     H, W = ir.shape[1], ir.shape[2]
    #
    #     #1.
    #     x, y = np.random.randint(10, H-10-self.ps+1), np.random.randint(10, W-10-self.ps+1)
    #     ir_crop = ir[:, x:x+self.ps, y:y+self.ps]
    #     vis_crop = vis[:, x:x+self.ps, y:y+self.ps]
    #     ir_map_crop = ir_map[:, x:x+self.ps, y:y+self.ps]
    #     vis_map_crop = vis_map[:, x:x+self.ps, y:y+self.ps]
    #
    #     return ir_crop, vis_crop, ir_map_crop, vis_map_crop

    def get_patch(self, ir, vis, ir_warp, vis_warp, disp):

        H, W = ir.shape[1], ir.shape[2]

        # 1.
        x, y = np.random.randint(10, H - 10 - self.ps + 1), np.random.randint(10, W - 10 - self.ps + 1)
        ir_crop = ir[:, x:x + self.ps, y:y + self.ps]
        vis_crop = vis[:, x:x + self.ps, y:y + self.ps]
        ir_warp_crop = ir_warp[:, x:x + self.ps, y:y + self.ps]
        vis_warp_crop = vis_warp[:, x:x + self.ps, y:y + self.ps]
        disp_crop = disp[:, x:x + self.ps, y:y + self.ps]

        return ir_crop, vis_crop, ir_warp_crop, vis_warp_crop, disp_crop

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]

        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vi = self.imread(path=vi_path, flags=cv2.IMREAD_GRAYSCALE)

        # iw, disp = self.image_trans(ir)
        disp = self.image_trans(ir)
        # disp = disp.permute(0, 3, 1, 2) * 224
        disp = disp.permute(0, 3, 1, 2) * 512
        ir = ir.unsqueeze(0)
        vi = vi.unsqueeze(0)
        ir_warp = self.warp(disp, ir)
        vis_warp = self.warp(disp, vi)
        ir = ir.squeeze(0)
        vi = vi.squeeze(0)
        ir_warp = ir_warp.squeeze(0)
        vis_warp = vis_warp.squeeze(0)

        disp = disp.squeeze(0)

        ir_crop, vis_crop, ir_warp_crop, vis_warp_crop, disp_crop = self.get_patch(ir, vi, ir_warp, vis_warp, disp)

        return (ir_crop, vis_crop), (str(ir_path), str(vi_path)), (ir_warp_crop, vis_warp_crop), disp_crop

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts


class FuseTestData_crop_fus(torch.utils.data.Dataset):
    """
    Load dataset with ir folder path and vis folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path, ir_map: pathlib.Path, vi_map: pathlib.Path):
        super(FuseTestData_crop_fus, self).__init__()
        # self.crop = crop
        # gain ir and vis images list
        self.ps = 224

        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

        self.ir_map_list = [x for x in sorted(ir_map.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_map_list = [x for x in sorted(vi_map.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

        self.image_trans = ImageTransform_vari()
        self.warp = Warper2d()

        # self.transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip,
        #     transforms.RandomVerticalFlip,
        #     transforms.RandomChoice([
        #         transforms.RandomRotation((0, 0)),
        #         transforms.RandomRotation((90, 90)),
        #         transforms.RandomRotation((180, 180)),
        #         transforms.RandomRotation((270, 270)),
        #     ])
        # ])

        # self.img_resize = RandomResizedCrop(224)

    # def get_patch(self, ir, vis, ir_map, vis_map):
    #     H, W = ir.shape[1], ir.shape[2]
    #
    #     #1.
    #     x, y = np.random.randint(10, H-10-self.ps+1), np.random.randint(10, W-10-self.ps+1)
    #     ir_crop = ir[:, x:x+self.ps, y:y+self.ps]
    #     vis_crop = vis[:, x:x+self.ps, y:y+self.ps]
    #     ir_map_crop = ir_map[:, x:x+self.ps, y:y+self.ps]
    #     vis_map_crop = vis_map[:, x:x+self.ps, y:y+self.ps]
    #
    #     return ir_crop, vis_crop, ir_map_crop, vis_map_crop

    def random_rotate(self, image):
        angle = torch.rand(1) * 360  # 随机生戍旋转角度rotated_image = TF.rotate(image，angle.item( ))return rotated_image
        rotated_image = TF. rotate(image, angle.item())
        return rotated_image

    def get_patch(self, ir, vis, ir_warp, vis_warp, ir_map, vi_map, disp):

        H, W = ir.shape[1], ir.shape[2]

        # 1.
        x, y = np.random.randint(10, H - 10 - self.ps + 1), np.random.randint(10, W - 10 - self.ps + 1)
        ir_crop = ir[:, x:x + self.ps, y:y + self.ps]
        vis_crop = vis[:, x:x + self.ps, y:y + self.ps]
        ir_warp_crop = ir_warp[:, x:x + self.ps, y:y + self.ps]
        vis_warp_crop = vis_warp[:, x:x + self.ps, y:y + self.ps]
        ir_map_crop = ir_map[:, x:x + self.ps, y:y + self.ps]
        vis_map_crop = vi_map[:, x:x + self.ps, y:y + self.ps]
        disp_crop = disp[:, x:x + self.ps, y:y + self.ps]

        return ir_crop, vis_crop, ir_warp_crop, vis_warp_crop, ir_map_crop, vis_map_crop, disp_crop

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]

        ir_map_path = self.ir_map_list[index]
        vi_map_path = self.vi_map_list[index]

        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vi = self.imread(path=vi_path, flags=cv2.IMREAD_GRAYSCALE)

        ir_map = self.imread(path=ir_map_path, flags=cv2.IMREAD_GRAYSCALE)
        vi_map = self.imread(path=vi_map_path, flags=cv2.IMREAD_GRAYSCALE)

        # # random flip
        # cat = torch.cat((ir, vi, ir_map, vi_map), dim=0)
        # # cat = transforms.ToPILImage(cat)
        # cat_flip = self.random_rotate(cat)
        # ir = cat_flip[:1, :, :]
        # vi = cat_flip[1:2, :, :]
        # ir_map = cat_flip[2:3, :, :]
        # vi_map = cat_flip[3:4, :, :]

        # iw, disp = self.image_trans(ir)
        disp = self.image_trans(ir)
        # disp = disp.permute(0, 3, 1, 2) * 224
        disp = disp.permute(0, 3, 1, 2) * 512
        ir = ir.unsqueeze(0)
        # ir = ir.unsqueeze(0)
        vi = vi.unsqueeze(0)
        ir_warp = self.warp(disp, ir)
        vis_warp = self.warp(disp, vi)
        ir = ir.squeeze(0)
        vi = vi.squeeze(0)
        ir_warp = ir_warp.squeeze(0)
        vis_warp = vis_warp.squeeze(0)

        disp = disp.squeeze(0)


        ir_crop, vis_crop, ir_warp_crop, vis_warp_crop, ir_map_crop, vi_map_crop, disp_crop = self.get_patch(ir, vi, ir_warp, vis_warp, ir_map, vi_map, disp)

        return (ir_crop, vis_crop), (str(ir_path), str(vi_path)), (ir_warp_crop, vis_warp_crop), (ir_map_crop, vi_map_crop), disp_crop

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts

