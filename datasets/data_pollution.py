# import random
#
# import torch
# from PIL import Image
# from torchvision import transforms
#
#
# def read_image(path):
#     """Read image from path using ``PIL.Image``.
#
#     Args:
#         path (str): path to an image.
#
#     Returns:
#         PIL image
#     """
#     return Image.open(path).convert("RGB")
#
#
# class RandomBlackout:
#     """以50%的概率将图像转换为完全黑色的图像。"""
#
#     def __init__(self):
#         pass
#
#     def __call__(self, img):
#         # 创建一个全零的张量（全黑图像）
#         black_img = torch.zeros_like(img)
#         return black_img
#
#
# def create_black_image(image_path, size=(224, 224)):
#     """
#     创建一个全黑的图像并保存到指定路径
#     """
#     black_image = Image.new("RGB", size, (0, 0, 0))  # 创建全黑图像
#     black_image.save(image_path)  # 保存到指定路径
#     return image_path
#
#
# def data_pollution(train, percentage):
#     if not (0 <= percentage <= 1):
#         raise ValueError("Percentage must be between 0 and 1.")  # percentage 需要是0到1之间的浮动值
#
#     num_samples = len(train)
#     num_polluted = int(num_samples * percentage)
#
#     # 随机选择要污染的索引
#     polluted_indices = random.sample(range(num_samples), num_polluted)
#
#     # 先全部初始化污染标记为False
#     for data in train:
#         data._pollution = False
#
#     # 对污染的样本进行黑图处理
#     for index in polluted_indices:
#         data = train[index]
#         data._pollution = True  # 标记为污染
#         # # 打印样本的属性，检查每个数据的结构
#         # print(f"Data structure : {dir(data)}")  # 逐个打印每个样本的数据结构
#         # 检查图像路径并加载图像
#         if hasattr(data, 'impath'):  # 检查是否有 'impath' 属性
#             # 创建一个全黑图像并保存到文件
#             black_image_path = f"/home/his/zby/save/black_image_{index}.jpg"
#             create_black_image(black_image_path)  # 创建并保存全黑图像
#             # 替换图像路径为全黑图像的路径
#             data.impath = black_image_path  # 更新 imapth 为全黑图像的路径
#             print("客户端已经污染")
#         else:
#             raise AttributeError(f"Sample at index {index} does not have 'impath' attribute.")
#     return train

from PIL import Image
import torch
import random


def create_black_pil(size=(224, 224)):
    """创建一个黑色 PIL Image"""
    return Image.new("RGB", size, (0, 0, 0))


def create_black_tensor(tensor_like):
    """根据原 tensor 的 shape 自动生成黑张量"""
    return torch.zeros_like(tensor_like)


def data_pollution(train, percentage, black_size=(224, 224)):
    if not (0 <= percentage <= 1):
        raise ValueError("Percentage must be between 0 and 1.")

    num_samples = len(train)
    num_polluted = int(num_samples * percentage)
    polluted_indices = random.sample(range(num_samples), num_polluted)

    # 初始化所有样本的污染标记
    for data in train:
        data._pollution = False

    for index in polluted_indices:
        data = train[index]
        data._pollution = True

        # 1. 如果已有 PIL 图像
        if hasattr(data, "img") and data.img is not None:
            data.img = create_black_pil(black_size)

        # 2. 如果已有 tensor 图像
        elif hasattr(data, "pixel") and data.pixel is not None:
            data.pixel = create_black_tensor(data.pixel)

        # 3. 既没有 img，也没有 pixel —— 从 impath 加载！
        elif hasattr(data, "impath"):
            try:
                img = Image.open(data.impath).convert("RGB")
                # 覆写为黑图（不修改 impath）
                data.img = create_black_pil(black_size)
            except Exception as e:
                raise RuntimeError(
                    f"无法从 impath 加载图片：index={index}, path={data.impath}. 错误：{e}"
                )
        else:
            raise AttributeError(
                f"Sample at index {index} has no 'img'/'pixel'/'impath', 无法污染。"
            )

        print(f"样本 {index} 已被污染（黑图注入，不修改路径）")

    return train
