# import numpy as np
# import random
# import torch
# import torchvision.transforms.functional as F
# from torchvision.transforms import (
#     Resize, Compose, ToTensor, Normalize, CenterCrop, RandomCrop, ColorJitter,
#     RandomApply, GaussianBlur, RandomGrayscale, RandomResizedCrop,
#     RandomHorizontalFlip
# )
# from torchvision.transforms.functional import InterpolationMode
# from PIL import Image, ImageEnhance
# from .autoaugment import SVHNPolicy, CIFAR10Policy, ImageNetPolicy
# from .randaugment import RandAugment, RandAugment2, RandAugmentFixMatch
#
# AVAI_CHOICES = [
#     "random_flip",
#     "random_resized_crop",
#     "normalize",
#     "instance_norm",
#     "random_crop",
#     "random_translation",
#     "center_crop",  # This has become a default operation during testing
#     "cutout",
#     "imagenet_policy",
#     "cifar10_policy",
#     "svhn_policy",
#     "randaugment",
#     "randaugment_fixmatch",
#     "randaugment2",
#     "gaussian_noise",
#     "colorjitter",
#     "randomgrayscale",
#     "gaussian_blur",
#     "light_variations"
# ]
#
# INTERPOLATION_MODES = {
#     "bilinear": InterpolationMode.BILINEAR,
#     "bicubic": InterpolationMode.BICUBIC,
#     "nearest": InterpolationMode.NEAREST,
# }
#
#
#
# class Random2DTranslation:
#     """Given an image of (height, width), we resize it to
#     (height*1.125, width*1.125), and then perform random cropping.
#
#     Args:
#         height (int): target image height.
#         width (int): target image width.
#         p (float, optional): probability that this operation takes place.
#             Default is 0.5.
#         interpolation (int, optional): desired interpolation. Default is
#             ``torchvision.transforms.functional.InterpolationMode.BILINEAR``
#     """
#
#     def __init__(
#         self, height, width, p=0.5, interpolation=InterpolationMode.BILINEAR
#     ):
#         self.height = height
#         self.width = width
#         self.p = p
#         self.interpolation = interpolation
#
#     def __call__(self, img):
#         if random.uniform(0, 1) > self.p:
#             return F.resize(
#                 img=img,
#                 size=[self.height, self.width],
#                 interpolation=self.interpolation
#             )
#
#         new_width = int(round(self.width * 1.125))
#         new_height = int(round(self.height * 1.125))
#         resized_img = F.resize(
#             img=img,
#             size=[new_height, new_width],
#             interpolation=self.interpolation
#         )
#         x_maxrange = new_width - self.width
#         y_maxrange = new_height - self.height
#         x1 = int(round(random.uniform(0, x_maxrange)))
#         y1 = int(round(random.uniform(0, y_maxrange)))
#         croped_img = F.crop(
#             img=resized_img,
#             top=y1,
#             left=x1,
#             height=self.height,
#             width=self.width
#         )
#
#         return croped_img
#
#
# class LightVariations:
#     def __init__(self, min_factor=1.0, max_factor=1.6, p=0.6):
#         assert min_factor >= 1.0, "min_factor should be >= 1.0"
#         assert max_factor >= min_factor, "max_factor should be >= min_factor"
#         self.min_factor = min_factor
#         self.max_factor = max_factor
#         self.p = p
#
#     def __call__(self, img):
#
#         if random.random() < self.p:
#             factor = random.uniform(self.min_factor, self.max_factor)
#             enhancer = ImageEnhance.Brightness(img)
#             img = enhancer.enhance(factor)
#         return img
#
# class InstanceNormalization:
#     """Normalize data using per-channel mean and standard deviation.
#
#     Reference:
#         - Ulyanov et al. Instance normalization: The missing in- gredient
#           for fast stylization. ArXiv 2016.
#         - Shu et al. A DIRT-T Approach to Unsupervised Domain Adaptation.
#           ICLR 2018.
#     """
#
#     def __init__(self, eps=1e-8):
#         self.eps = eps
#
#     def __call__(self, img):
#         C, H, W = img.shape
#         img_re = img.reshape(C, H * W)
#         mean = img_re.mean(1).view(C, 1, 1)
#         std = img_re.std(1).view(C, 1, 1)
#         return (img-mean) / (std + self.eps)
#
#
# class Cutout:
#     """Randomly mask out one or more patches from an image.
#
#     https://github.com/uoguelph-mlrg/Cutout
#
#     Args:
#         n_holes (int, optional): number of patches to cut out
#             of each image. Default is 1.
#         length (int, optinal): length (in pixels) of each square
#             patch. Default is 16.
#     """
#
#     def __init__(self, n_holes=1, length=16):
#         self.n_holes = n_holes
#         self.length = length
#
#     def __call__(self, img):
#         """
#         Args:
#             img (Tensor): tensor image of size (C, H, W).
#
#         Returns:
#             Tensor: image with n_holes of dimension
#                 length x length cut out of it.
#         """
#         h = img.size(1)
#         w = img.size(2)
#
#         mask = np.ones((h, w), np.float32)
#
#         for n in range(self.n_holes):
#             y = np.random.randint(h)
#             x = np.random.randint(w)
#
#             y1 = np.clip(y - self.length // 2, 0, h)
#             y2 = np.clip(y + self.length // 2, 0, h)
#             x1 = np.clip(x - self.length // 2, 0, w)
#             x2 = np.clip(x + self.length // 2, 0, w)
#
#             mask[y1:y2, x1:x2] = 0.0
#
#         mask = torch.from_numpy(mask)
#         mask = mask.expand_as(img)
#         return img * mask
#
#
# class GaussianNoise:
#     """Add gaussian noise."""
#
#     def __init__(self, mean=0, std=0.15, p=0.5):
#         self.mean = mean
#         self.std = std
#         self.p = p
#
#     def __call__(self, img):
#         if random.uniform(0, 1) > self.p:
#             return img
#         noise = torch.randn(img.size()) * self.std + self.mean
#         return img + noise
#
#
# def build_transform(cfg, is_train=True, choices=None):
#     """Build transformation function.
#
#     Args:
#         cfg (CfgNode): config.
#         is_train (bool, optional): for training (True) or test (False).
#             Default is True.
#         choices (list, optional): list of strings which will overwrite
#             cfg.INPUT.TRANSFORMS if given. Default is None.
#     """
#     if cfg.INPUT.NO_TRANSFORM:
#         print("Note: no transform is applied!")
#         return None
#
#     if choices is None:
#         choices = cfg.INPUT.TRANSFORMS
#
#     for choice in choices:
#         assert choice in AVAI_CHOICES
#
#     target_size = f"{cfg.INPUT.SIZE[0]}x{cfg.INPUT.SIZE[1]}"
#
#     normalize = Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
#
#     if is_train:
#         return _build_transform_train(cfg, choices, target_size, normalize)
#     else:
#         return _build_transform_test(cfg, choices, target_size, normalize)
#
#
# def _build_transform_train(cfg, choices, target_size, normalize):
#     print("Building transform_train")
#     tfm_train = []
#
#     interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
#     input_size = cfg.INPUT.SIZE
#
#     # Make sure the image size matches the target size
#     conditions = []
#     conditions += ["random_crop" not in choices]
#     conditions += ["random_resized_crop" not in choices]
#     if all(conditions):
#         print(f"+ resize to {target_size}")
#         tfm_train += [Resize(input_size, interpolation=interp_mode)]
#
#     if "light_variations" in choices:
#         print("+ LightVariations")
#         tfm_train += [LightVariations()]
#
#     if "random_translation" in choices:
#         print("+ random translation")
#         tfm_train += [Random2DTranslation(input_size[0], input_size[1])]
#
#     if "random_crop" in choices:
#         crop_padding = cfg.INPUT.CROP_PADDING
#         print(f"+ random crop (padding = {crop_padding})")
#         tfm_train += [RandomCrop(input_size, padding=crop_padding)]
#
#     if "random_resized_crop" in choices:
#         s_ = cfg.INPUT.RRCROP_SCALE
#         print(f"+ random resized crop (size={input_size}, scale={s_})")
#         tfm_train += [
#             RandomResizedCrop(input_size, scale=s_, interpolation=interp_mode)
#         ]
#
#     if "random_flip" in choices:
#         print("+ random flip")
#         tfm_train += [RandomHorizontalFlip()]
#
#     if "imagenet_policy" in choices:
#         print("+ imagenet policy")
#         tfm_train += [ImageNetPolicy()]
#
#     if "cifar10_policy" in choices:
#         print("+ cifar10 policy")
#         tfm_train += [CIFAR10Policy()]
#
#     if "svhn_policy" in choices:
#         print("+ svhn policy")
#         tfm_train += [SVHNPolicy()]
#
#     if "randaugment" in choices:
#         n_ = cfg.INPUT.RANDAUGMENT_N
#         m_ = cfg.INPUT.RANDAUGMENT_M
#         print(f"+ randaugment (n={n_}, m={m_})")
#         tfm_train += [RandAugment(n_, m_)]
#
#     if "randaugment_fixmatch" in choices:
#         n_ = cfg.INPUT.RANDAUGMENT_N
#         print(f"+ randaugment_fixmatch (n={n_})")
#         tfm_train += [RandAugmentFixMatch(n_)]
#
#     if "randaugment2" in choices:
#         n_ = cfg.INPUT.RANDAUGMENT_N
#         print(f"+ randaugment2 (n={n_})")
#         tfm_train += [RandAugment2(n_)]
#
#     if "colorjitter" in choices:
#         b_ = cfg.INPUT.COLORJITTER_B
#         c_ = cfg.INPUT.COLORJITTER_C
#         s_ = cfg.INPUT.COLORJITTER_S
#         h_ = cfg.INPUT.COLORJITTER_H
#         print(
#             f"+ color jitter (brightness={b_}, "
#             f"contrast={c_}, saturation={s_}, hue={h_})"
#         )
#         tfm_train += [
#             ColorJitter(
#                 brightness=b_,
#                 contrast=c_,
#                 saturation=s_,
#                 hue=h_,
#             )
#         ]
#
#     if "randomgrayscale" in choices:
#         print("+ random gray scale")
#         tfm_train += [RandomGrayscale(p=cfg.INPUT.RGS_P)]
#
#     if "gaussian_blur" in choices:
#         print(f"+ gaussian blur (kernel={cfg.INPUT.GB_K})")
#         gb_k, gb_p = cfg.INPUT.GB_K, cfg.INPUT.GB_P
#         tfm_train += [RandomApply([GaussianBlur(gb_k)], p=gb_p)]
#
#     print("+ to torch tensor of range [0, 1]")
#     tfm_train += [ToTensor()]
#
#     if "cutout" in choices:
#         cutout_n = cfg.INPUT.CUTOUT_N
#         cutout_len = cfg.INPUT.CUTOUT_LEN
#         print(f"+ cutout (n_holes={cutout_n}, length={cutout_len})")
#         tfm_train += [Cutout(cutout_n, cutout_len)]
#
#     if "normalize" in choices:
#         print(
#             f"+ normalization (mean={cfg.INPUT.PIXEL_MEAN}, std={cfg.INPUT.PIXEL_STD})"
#         )
#         tfm_train += [normalize]
#
#     if "gaussian_noise" in choices:
#         print(
#             f"+ gaussian noise (mean={cfg.INPUT.GN_MEAN}, std={cfg.INPUT.GN_STD})"
#         )
#         tfm_train += [GaussianNoise(cfg.INPUT.GN_MEAN, cfg.INPUT.GN_STD)]
#
#     if "instance_norm" in choices:
#         print("+ instance normalization")
#         tfm_train += [InstanceNormalization()]
#
#     tfm_train = Compose(tfm_train)
#
#     return tfm_train
#
#
# def _build_transform_test(cfg, choices, target_size, normalize):
#     print("Building transform_test")
#     tfm_test = []
#
#     interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
#     input_size = cfg.INPUT.SIZE
#
#     print(f"+ resize the smaller edge to {max(input_size)}")
#     tfm_test += [Resize(max(input_size), interpolation=interp_mode)]
#
#     print(f"+ {target_size} center crop")
#     tfm_test += [CenterCrop(input_size)]
#
#     print("+ to torch tensor of range [0, 1]")
#     tfm_test += [ToTensor()]
#
#     if "normalize" in choices:
#         print(
#             f"+ normalization (mean={cfg.INPUT.PIXEL_MEAN}, std={cfg.INPUT.PIXEL_STD})"
#         )
#         tfm_test += [normalize]
#
#     if "instance_norm" in choices:
#         print("+ instance normalization")
#         tfm_test += [InstanceNormalization()]
#
#     tfm_test = Compose(tfm_test)
#
#     return tfm_test

import numpy as np
import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, CenterCrop, RandomCrop, ColorJitter,
    RandomApply, GaussianBlur, RandomGrayscale, RandomResizedCrop,
    RandomHorizontalFlip
)
from torchvision.transforms.functional import InterpolationMode

from .autoaugment import SVHNPolicy, CIFAR10Policy, ImageNetPolicy
from .randaugment import RandAugment, RandAugment2, RandAugmentFixMatch

AVAI_CHOICES = [
    "random_flip",
    "random_crop",
    "random_resized_crop",
    "center_crop",
    "colorjitter",
    "normalize",
    "gaussian_noise",
    "instance_norm",
    "cutout",
    "randaugment",
    "randaugment_fixmatch",
    "randaugment2",
    "cifar10",
    "svhn",
    "imagenet",
    "randomgrayscale",
    "gaussian_blur",
    "resize",
    "data_pollution",
]


INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


def build_transform(cfg, is_train=True):
    """Build transformation function.

    It expects a config object that has at least the following attributes:

        - INPUT.SIZE (list): height and width.
        - INPUT.PIXEL_MEAN (list): pixel mean.
        - INPUT.PIXEL_STD (list): pixel std.
        - INPUT.INTERPOLATION (str): interpolation method.
        - INPUT.TRANSFORMS (list): list of transform names to use.

    Args:
        cfg (CfgNode): config.
        is_train (bool, optional): training or test. Default is True.
    """
    # Build normalization function.
    pixel_mean = np.array(cfg.INPUT.PIXEL_MEAN)
    pixel_std = np.array(cfg.INPUT.PIXEL_STD)
    normalize = Normalize(mean=pixel_mean, std=pixel_std)

    # Build transformation function.
    transforms = cfg.INPUT.TRANSFORMS
    transforms = [t.lower() for t in transforms]
    choices = transforms

    for t in choices:
        if t not in AVAI_CHOICES:
            raise ValueError(
                f"Transform '{t}' not found. "
                f"Available choices are {AVAI_CHOICES}."
            )

    height, width = cfg.INPUT.SIZE
    target_size = (height , width)

    if is_train:
        return _build_transform_train(cfg, choices, target_size, normalize)
    else:
        return _build_transform_test(cfg, choices, target_size, normalize)


def _build_transform_train(cfg, choices, target_size, normalize):
    print("Building transform_train")
    tfm_train = []

    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
    input_size = cfg.INPUT.SIZE

    # Make sure the image size matches the target size
    conditions = []
    conditions += ["random_crop" not in choices]
    conditions += ["random_resized_crop" not in choices]
    if all(conditions):
        print(f"+ resize to {target_size}")
        tfm_train += [Resize(input_size, interpolation=interp_mode)]

    if "random_crop" in choices:
        print(f"+ random crop to {target_size}")
        tfm_train += [RandomCrop(input_size)]

    if "center_crop" in choices:
        print(f"+ center crop to {target_size}")
        tfm_train += [CenterCrop(input_size)]

    if "random_resized_crop" in choices:
        print(f"+ random resized crop to {target_size}")
        tfm_train += [RandomResizedCrop(input_size, interpolation=interp_mode)]

    if "random_flip" in choices:
        print("+ random flip")
        tfm_train += [RandomHorizontalFlip()]

    if "cifar10" in choices:
        print("+ cifar10 policy")
        tfm_train += [CIFAR10Policy()]

    if "svhn" in choices:
        print("+ svhn policy")
        tfm_train += [SVHNPolicy()]

    if "randaugment" in choices:
        n_ = cfg.INPUT.RANDAUGMENT_N
        m_ = cfg.INPUT.RANDAUGMENT_M
        print(f"+ randaugment (n={n_}, m={m_})")
        tfm_train += [RandAugment(n_, m_)]

    if "randaugment_fixmatch" in choices:
        n_ = cfg.INPUT.RANDAUGMENT_N
        print(f"+ randaugment_fixmatch (n={n_})")
        tfm_train += [RandAugmentFixMatch(n_)]

    if "randaugment2" in choices:
        n_ = cfg.INPUT.RANDAUGMENT_N
        print(f"+ randaugment2 (n={n_})")
        tfm_train += [RandAugment2(n_)]

    if "colorjitter" in choices:
        b_ = cfg.INPUT.COLORJITTER_B
        c_ = cfg.INPUT.COLORJITTER_C
        s_ = cfg.INPUT.COLORJITTER_S
        h_ = cfg.INPUT.COLORJITTER_H
        print(
            f"+ color jitter (brightness={b_}, "
            f"contrast={c_}, saturation={s_}, hue={h_})"
        )
        tfm_train += [
            ColorJitter(
                brightness=b_,
                contrast=c_,
                saturation=s_,
                hue=h_,
            )
        ]

    if "randomgrayscale" in choices:
        print("+ random gray scale")
        tfm_train += [RandomGrayscale(p=cfg.INPUT.RGS_P)]

    if "gaussian_blur" in choices:
        print(f"+ gaussian blur (kernel={cfg.INPUT.GB_K})")
        gb_k, gb_p = cfg.INPUT.GB_K, cfg.INPUT.GB_P
        tfm_train += [RandomApply([GaussianBlur(gb_k)], p=gb_p)]

    print("+ to torch tensor of range [0, 1]")
    tfm_train += [ToTensor()]

    if "cutout" in choices:
        cutout_n = cfg.INPUT.CUTOUT_N
        cutout_len = cfg.INPUT.CUTOUT_LEN
        print(f"+ cutout (n_holes={cutout_n}, length={cutout_len})")
        tfm_train += [Cutout(cutout_n, cutout_len)]

    if "normalize" in choices:
        print(
            f"+ normalization (mean={cfg.INPUT.PIXEL_MEAN}, std={cfg.INPUT.PIXEL_STD})"
        )
        tfm_train += [normalize]

    if "gaussian_noise" in choices:
        print(
            f"+ gaussian noise (mean={cfg.INPUT.GN_MEAN}, std={cfg.INPUT.GN_STD})"
        )
        tfm_train += [GaussianNoise(cfg.INPUT.GN_MEAN, cfg.INPUT.GN_STD)]

    if "instance_norm" in choices:
        print("+ instance normalization")
        tfm_train += [InstanceNormalization()]

#    # ===== 数据污染（保持你的“随机变黑”思路）=====
#    if "data_pollution" in choices:
#        # 只对被标记为需要污染的客户端生效
#        do_pollute = (
#            hasattr(cfg, "TRAINER")
#            and hasattr(cfg.TRAINER, "PROMPTKD")
#            and getattr(cfg.TRAINER.PROMPTKD, "CLIENT_DATA_POLLUTION", False)
 #       )
 #       if do_pollute:
 #           # 概率从 yaml 里的 POLLUTION_PERCENTAGE 读
 #          prob = getattr(cfg.TRAINER.PROMPTKD, "POLLUTION_PERCENTAGE", 0.0)
#            print(f"+ data_pollution (prob={prob})")
#            tfm_train += [RandomBlackout(p=prob)]
 #       else:
  #          print("+ data_pollution skipped (CLIENT_DATA_POLLUTION=False)")
    # ============================================

    tfm_train = Compose(tfm_train)

    return tfm_train


def _build_transform_test(cfg, choices, target_size, normalize):
    print("Building transform_test")
    tfm_test = []

    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
    input_size = cfg.INPUT.SIZE

    print(f"+ resize the smaller edge to {max(input_size)}")
    tfm_test += [
        Resize(max(input_size), interpolation=interp_mode),
        CenterCrop(input_size),
    ]

    print("+ to torch tensor of range [0, 1]")
    tfm_test += [ToTensor()]

    print(
        f"+ normalization (mean={cfg.INPUT.PIXEL_MEAN}, std={cfg.INPUT.PIXEL_STD})"
    )
    tfm_test += [normalize]

    tfm_test = Compose(tfm_test)

    return tfm_test


class RandomBlackout:
    """
    随机将输入图像全黑。

    参数:
        p (float): 每次调用时将图像变成全黑的概率，范围为 [0, 1]。
    """

    def __init__(self, p=0.5):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p 应该在 [0.0, 1.0] 范围内")
        self.p = p

    def __call__(self, img):
        # 随机决定是否执行黑屏
        if random.random() < self.p:
            # 假设输入是 PIL.Image，先转为 tensor 方便处理
            if isinstance(img, torch.Tensor):
                # 如果已经是 Tensor，就直接生成全零 Tensor，尺寸与输入相同
                return torch.zeros_like(img)
            else:
                # 如果是 PIL 图像，先转 tensor 再置零
                tensor = ToTensor()(img)
                return torch.zeros_like(tensor)
        else:
            # 不执行黑屏，直接返回原图
            return img


class GaussianNoise(object):
    """
    Add gaussian noise.
    """

    def __init__(self, mean=0.0, std=1.0):
        self._mean = mean
        self._std = std

    def __call__(self, img):
        return img + torch.randn(img.size()) * self._std + self._mean


class InstanceNormalization(object):
    """
    Perform instance normalization.
    """

    def __call__(self, tensor):
        return F.instance_norm(tensor)


class Cutout(object):
    """
    Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): number of patches to cut out of each image.
        length (int): length of the patches.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).

        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class RandomResizeHeight(object):
    """Resize an image to a randomly chosen width or height with a random pick
    between the two scales.

    Args:
        height (int): desired image height.
        width (int): desired image width.
        p (float, optional): probability of applying the scaling. Default is 0.5.
        interpolation (int, optional): desired interpolation. Default is
            ``torchvision.transforms.functional.InterpolationMode.BILINEAR``
    """

    def __init__(
        self, height, width, p=0.5, interpolation=InterpolationMode.BILINEAR
    ):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return F.resize(
                img, (self.height, self.width), interpolation=self.interpolation
            )

        scale_y = random.uniform(0.5, 1.5)
        scale_x = random.uniform(0.5, 1.5)
        new_height = int(self.height * scale_y)
        new_width = int(self.width * scale_x)

        return F.resize(
            img, (new_height, new_width), interpolation=self.interpolation
        )


class TensorZeroPollution(object):
    """
    把输入的图像 tensor 以概率 prob 变成全 0（黑图）。
    这里假设 x 是 C×H×W 的 tensor，范围在 [0,1] 或 [0,255] 都没关系，
    全 0 对应“全黑”。
    """

    def __init__(self, prob: float = 0.0):
        self.prob = float(prob)

    def __call__(self, x):
        # x: torch.Tensor (C, H, W)
        if not isinstance(x, torch.Tensor):
            # 如果万一传进来的是 PIL，就直接原样返回，避免挂掉
            return x

        if random.random() < self.prob:
            return torch.zeros_like(x)
        return x