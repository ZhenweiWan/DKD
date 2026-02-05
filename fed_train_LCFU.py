import argparse
import copy
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from datasets.data_pollution import data_pollution
# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.sun397
import datasets.tinyimagenet200


import trainers.independentVL
import trainers.promptkd
from federated_learning import aggregation_model_parameter
from datasets.fixed_ratio_pollution import FixedRatioPollutedDataset


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    # if args.second_phase:
    #     cfg.TRAIN.SECOND_PHASE = args.second_phase


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, am
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9  # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for PromptSRC
    cfg.TRAINER.PROMPTSRC = CN()
    cfg.TRAINER.PROMPTSRC.N_CTX_VISION = 4  # number of context vectors at the vision branch
    cfg.TRAINER.PROMPTSRC.N_CTX_TEXT = 4  # number of context vectors at the language branch
    cfg.TRAINER.PROMPTSRC.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROMPTSRC.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT = 25
    cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT = 10
    cfg.TRAINER.PROMPTSRC.GPA_MEAN = 15
    cfg.TRAINER.PROMPTSRC.GPA_STD = 1

    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting(J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.TEST.NO_TEST = False

    # KD
    # cfg.MODEL.BACKBONE.TEACHER_NAME = "ViT/L-14"
    # cfg.MODEL.BACKBONE.PROJECT_LAYER = 2
    # cfg.MODEL.BACKBONE.CE_WEIGHT = 0.0

    cfg.TRAINER.MODAL = "base2novel"
    cfg.TRAINER.PROMPTKD = CN()
    cfg.TRAINER.PROMPTKD.N_CTX_VISION = 4  # number of context vectors at the vision branch
    cfg.TRAINER.PROMPTKD.N_CTX_TEXT = 4  # number of context vectors at the language branch
    cfg.TRAINER.PROMPTKD.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PROMPTKD.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PROMPTKD.PROJECT_LAYER = 2
    cfg.TRAINER.PROMPTKD.CE_WEIGHT = 0.0
    cfg.TRAINER.PROMPTKD.KD_WEIGHT = 1.0
    cfg.TRAINER.PROMPTKD.TEMPERATURE = 1.0
    cfg.TRAINER.PROMPTKD.TEACHER_NAME = "ViT/L-14"
    cfg.TRAINER.PROMPTKD.STRONG = False
    cfg.TRAINER.PROMPTKD.ROUND = 1
    cfg.TRAINER.PROMPTKD.NUM_CLIENT = 10
    cfg.TRAINER.PROMPTKD.DATA_POLLUTION = False
    cfg.TRAINER.PROMPTKD.CLIENT_DATA_POLLUTION = False
    cfg.TRAINER.PROMPTKD.POLLUTION_CLIENT_ID = []
    cfg.TRAINER.PROMPTKD.POLLUTION_PERCENTAGE = 0.5
    cfg.TRAINER.PROMPTKD.LIGHT_VARIATIONS = False
    cfg.TRAINER.PROMPTKD.COLOUR_VARIATIONS = False
    cfg.TRAINER.PROMPTKD.DIFFERENT_NUM_SHOT = False
    cfg.TRAINER.PROMPTKD.CLIENT_SHOTS = []
    cfg.TRAINER.PROMPTKD.SECOND_PHASE = False


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    # cfg.freeze()

    return cfg


def generate_color_jitter_params(brightness=0.3, contrast=0.3, saturation=0.3):
    params = {
        'brightness': random.uniform(1 - brightness, 1 + brightness),
        'contrast': random.uniform(1 - contrast, 1 + contrast),
        'saturation': random.uniform(1 - saturation, 1 + saturation),
    }
    return params


def apply_fixed_ratio_pollution(clients_trainer, global_cfg):
    """
    对 yaml 中指定的污染客户端，应用“固定比例污染”：
    - 固定选出 ratio 比例的样本 index
    - 这些样本在整个训练过程中永远输出全 0 张量
    """
    if not global_cfg.TRAINER.PROMPTKD.DATA_POLLUTION:
        print("[FixedPollution] DATA_POLLUTION = False，不启用固定比例污染")
        return

    pollution_ids = global_cfg.TRAINER.PROMPTKD.POLLUTION_CLIENT_ID
    ratio = float(global_cfg.TRAINER.PROMPTKD.POLLUTION_PERCENTAGE)

    print(f"[FixedPollution] 将对客户端 {pollution_ids} 使用固定比例污染 ratio={ratio}")

    for cid in pollution_ids:
        if cid not in clients_trainer:
            print(f"[FixedPollution] client_{cid} 不在 clients_trainer 中，跳过")
            continue

        trainer = clients_trainer[cid]
        old_loader = getattr(trainer, "train_loader_x", None)
        if old_loader is None:
            print(f"[FixedPollution] client_{cid} 没有 train_loader_x，跳过")
            continue

        base_dataset = old_loader.dataset
        polluted_dataset = FixedRatioPollutedDataset(base_dataset, ratio)

        # ✅ 关键改动：重新建一个 DataLoader，而不是改老的 .dataset
        new_loader = DataLoader(
            polluted_dataset,
            batch_size=old_loader.batch_size,
            sampler=old_loader.sampler,  # 复用原来的采样策略（保持 shuffle 等一致）
            num_workers=old_loader.num_workers,
            collate_fn=old_loader.collate_fn,
            pin_memory=old_loader.pin_memory,
            drop_last=old_loader.drop_last,
            timeout=old_loader.timeout,
            worker_init_fn=old_loader.worker_init_fn,
            multiprocessing_context=old_loader.multiprocessing_context,
            generator=getattr(old_loader, "generator", None),
            prefetch_factor=getattr(old_loader, "prefetch_factor", 2),
            persistent_workers=getattr(old_loader, "persistent_workers", False),
        )

        trainer.train_loader_x = new_loader
        print(f"[FixedPollution] client_{cid} 已应用固定比例污染。")


def split_data_by_dirichlet(total_data, num_client, alpha):
    """
    使用 Dirichlet 分布将 total_data 分配到 num_client 个客户端。
    Args:
        total_data (list of Datum): 所有训练数据
        num_client (int): 客户端数量
        alpha (float): Dirichlet 分布的浓度参数（越小越偏）

    Returns:
        list of list: 每个客户端的数据列表
    """

    # 按 label 分组，一个label对应一个list
    label_to_data = defaultdict(list)
    for item in total_data:
        label_to_data[item.label].append(item)

    # 为每个客户端准备一个空列表，后面往里塞样本
    client_data = [[] for _ in range(num_client)]

    for label, items in label_to_data.items():
        # 打乱数据
        random.shuffle(items)

        # 使用 Dirichlet 分布决定每个客户端分得该类样本的比例
        proportions = np.random.dirichlet([alpha] * num_client)

        # 根据比例分配样本
        data_idx = 0
        for client_idx, prop in enumerate(proportions):
            num_samples = int(prop * len(items))
            client_data[client_idx].extend(items[data_idx: data_idx + num_samples])
            data_idx += num_samples

        # 处理余数（未分配的样本）——分配给客户端中数据最少的
        if data_idx < len(items):
            leftovers = items[data_idx:]
            for item in leftovers:
                min_client = min(range(num_client), key=lambda x: len(client_data[x]))
                client_data[min_client].append(item)

    return client_data


def creat_clients_trainer(args, cfg):
    num_client = cfg.TRAINER.PROMPTKD.NUM_CLIENT
    pollution_client_ID = cfg.TRAINER.PROMPTKD.POLLUTION_CLIENT_ID
    clients_args = {}
    clients_cfg = {}
    clients_trainer = {}
    unlearning_trainer_cfg = {}
    new_unlearning_trainer_cfg = {}
    new_class_trainer_cfg = {}
    new_class_trainer = {}
    for i in range(num_client):
        # 给每一个client设置不同的seed来控制抽到的数据
        args.seed = args.seed + (i + 1)
        clients_args[i] = copy.deepcopy(args)
        clients_args[i].output_dir = clients_args[i].output_dir + "_client" + str(i)

        # 生成客户端cfg文件
        clients_cfg[i] = setup_cfg(clients_args[i])
        clients_cfg[i].TRAINER.PROMPTKD.CLIENT_ID = i
        if clients_cfg[i].SEED >= 0:
            set_random_seed(clients_cfg[i].SEED)
        setup_logger(clients_cfg[i].OUTPUT_DIR)
        unlearning_trainer_cfg[i] = copy.deepcopy(clients_cfg[i])
        new_unlearning_trainer_cfg[i] = copy.deepcopy(clients_cfg[i])
        new_unlearning_trainer_cfg[i].DATASET.SUBSAMPLE_CLASSES = "new"

        # 这个缩进要和 for 循环平齐，表示整个 for 执行完之后再打印
        print("[Debug] num_client =", num_client)
        print("[Debug] clients_trainer keys =", list(clients_trainer.keys()))

        # 添加数据污染
        if cfg.TRAINER.PROMPTKD.DATA_POLLUTION is True:
            if not (len(pollution_client_ID) <= num_client):
                raise ValueError("polluting clients exceeds the number of clients")

            if i in pollution_client_ID:
                clients_cfg[i].TRAINER.PROMPTKD.CLIENT_DATA_POLLUTION = True
                print(f"客户端 {i} 需要污染")

                transforms_list = list(cfg.INPUT.TRANSFORMS)

                # 只给被选中的客户端多一个 "data_pollution"
                if "data_pollution" not in transforms_list:
                    transforms_list.append("data_pollution")

                clients_cfg[i].INPUT.TRANSFORMS = transforms_list
                print(f"[Debug] client_{i} INPUT.TRANSFORMS = {clients_cfg[i].INPUT.TRANSFORMS}")
            else:
                print(f"[Debug] client_{i} 无污染, INPUT.TRANSFORMS = {clients_cfg[i].INPUT.TRANSFORMS}")

        # 添加亮度变化
        if cfg.TRAINER.PROMPTKD.LIGHT_VARIATIONS is True:
            clients_cfg[i].TRAINER.PROMPTKD.CLIENT_LIGHT_VARIATIONS = True
            clients_cfg[i].INPUT.LIGHT_VARIATIONS_FACTOR = cfg.INPUT.LIGHT_VARIATIONS_FACTOR
            cfg.INPUT.LIGHT_VARIATIONS_FACTOR = cfg.INPUT.LIGHT_VARIATIONS_FACTOR + 0.3

        # 添加颜色变换
        if cfg.TRAINER.PROMPTKD.COLOUR_VARIATIONS is True:
            clients_cfg[i].TRAINER.PROMPTKD.CLIENT_COLOUR_VARIATIONS = True
            color_params = generate_color_jitter_params()
            clients_cfg[i].INPUT.COLORJITTER_B = color_params['brightness']
            clients_cfg[i].INPUT.COLORJITTER_C = color_params['contrast']
            clients_cfg[i].INPUT.COLORJITTER_S = color_params['saturation']

        # 不同客户端不同数据量
        if cfg.TRAINER.PROMPTKD.DIFFERENT_NUM_SHOT is True:
            if len(cfg.TRAINER.PROMPTKD.CLIENT_SHOTS) > cfg.TRAINER.PROMPTKD.NUM_CLIENT:
                raise ValueError("CLIENT_SHOTS_LIST > NUM_CLIENT")
            clients_cfg[i].DATASET.NUM_SHOTS = cfg.TRAINER.PROMPTKD.CLIENT_SHOTS[i]
            unlearning_trainer_cfg[i].DATASET.NUM_SHOTS = cfg.TRAINER.PROMPTKD.CLIENT_SHOTS[i]
            new_unlearning_trainer_cfg[i].DATASET.NUM_SHOTS = cfg.TRAINER.PROMPTKD.CLIENT_SHOTS[i]

        print("-----------------create client_{}---------------".format(i))
        print(clients_cfg[i])
        clients_trainer[i] = build_trainer(clients_cfg[i])

        new_class_trainer_cfg[i] = copy.deepcopy(clients_cfg[i])
        new_class_trainer_cfg[i].DATASET.SUBSAMPLE_CLASSES = "new"

    if args.dirichlet:

        total_train_data = []
        # dirichlet_client_num = num_client - len(pollution_client_ID)
        dirichlet_client_num = num_client
        for c_index in range(dirichlet_client_num):

            temp_train_data = clients_trainer[c_index].get_train_x_dataset()
            for item in temp_train_data:
                total_train_data.append(item)

        alpha = 0.3  # 划分比例 小于1代表更偏向非IID，比例从这里修改 0.3/0.5
        new_client_data = split_data_by_dirichlet(total_train_data, dirichlet_client_num, alpha)

        # ======= 打印每个客户端的类别分布(检查点) =======
        from collections import Counter
        for c_index in range(dirichlet_client_num):
            data = new_client_data[c_index]

            # Datum 对象的标签字段有的叫 label，有的叫 y，这里兼容一下
            labels = []
            for d in data:
                if hasattr(d, "label"):
                    labels.append(d.label)
                elif hasattr(d, "y"):
                    labels.append(d.y)
                else:
                    raise ValueError("找不到样本的标签字段(label / y)，请检查 Datum 定义")

            counter = Counter(labels)
            print(f"[Dirichlet] client {c_index}: "
                  f"samples={len(labels)}, "
                  f"num_classes={len(counter)}, "
                  f"label_counts={dict(counter)}")

        # Oxford Pets = 8 Oxford Flowers = 21 Stanford Cars = 40 SUN397 = 80
        # new_client_data = split_data_label_quantity(total_train_data, dirichlet_client_num, 21)
        # new_client_data = split_data_quantity_skew(total_train_data, dirichlet_client_num, 0.5)

        print(f"[dirichlet]使用 DIRICHLET 划分客户端数据（alpha={alpha}）")
        print("[debug]", type(clients_trainer[0].dm.dataset))
        for c_index in range(dirichlet_client_num):
            clients_trainer[c_index].dm.dataset.set_train_x(new_client_data[c_index])
            clients_trainer[c_index].rebulid_train_loader_x()

    return clients_trainer, clients_args, unlearning_trainer_cfg, new_class_trainer, new_unlearning_trainer_cfg


def debug_print_client_tensors(clients_trainer, max_batch=1, max_img_per_batch=1):
    """
    打印每个客户端 train_loader_x 里加载到的图像张量。
    - clients_trainer: 你的客户端 trainer 字典，例如 {0: trainer0, 1: trainer1, ...}
    - max_batch: 每个客户端最多看多少个 batch
    - max_img_per_batch: 每个 batch 打印多少张图
    """
    # 自动根据字典 key 推出有多少个 client
    client_ids = sorted(list(clients_trainer.keys()))
    print(f"[DebugTensor] 将检查客户端: {client_ids}")

    for cid in client_ids:
        trainer = clients_trainer[cid]
        loader = getattr(trainer, "train_loader_x", None)
        if loader is None:
            print(f"[DebugTensor] client_{cid} 没有 train_loader_x，跳过")
            continue

        print(f"\n[DebugTensor] ===== Client {cid} =====")
        for b_idx, batch in enumerate(loader):
            # ---- 根据 batch 类型解析 img 和 label ----
            if isinstance(batch, dict):
                imgs = batch.get("img", None)
                labels = batch.get("label", None)
            elif isinstance(batch, (list, tuple)):
                imgs, labels = batch[0], batch[1]
            else:
                print(f"[DebugTensor] client_{cid} batch_{b_idx} 类型未知: {type(batch)}")
                break

            if imgs is None:
                print(f"[DebugTensor] client_{cid} batch_{b_idx} 中没有 img 字段")
                break

            # 打印整体统计信息
            print(
                f"[DebugTensor] client_{cid} batch_{b_idx}: "
                f"imgs.shape = {tuple(imgs.shape)}, "
                f"min = {imgs.min().item():.4f}, "
                f"max = {imgs.max().item():.4f}, "
                f"mean = {imgs.mean().item():.4f}"
            )

            # 打印前几张具体 tensor（避免刷屏）
            for k in range(min(max_img_per_batch, imgs.size(0))):
                print(f"[DebugTensor] client_{cid} batch_{b_idx} image_{k} tensor：")
                print(imgs[k])

            if b_idx + 1 >= max_batch:
                break


def set_data_pollution_dir(OUTPUT_DIR, output_dir, cfg):
    OUTPUT_DIR = OUTPUT_DIR + '_' + \
                 str(cfg.TRAINER.PROMPTKD.POLLUTION_CLIENT_ID) + '_' + \
                 str(cfg.TRAINER.PROMPTKD.POLLUTION_PERCENTAGE)
    output_dir = cfg.OUTPUT_DIR + '_' + \
                 str(cfg.TRAINER.PROMPTKD.POLLUTION_CLIENT_ID) + '_' + \
                 str(cfg.TRAINER.PROMPTKD.POLLUTION_PERCENTAGE)

    return OUTPUT_DIR, output_dir


def client_new_class_test(test_model_trainer, param_model_trainer):
    print('---------------client new class test------------------')
    test_model_trainer.model.to(test_model_trainer.device)
    test_model_trainer.load_learnable_Parameters(param_model_trainer.get_learnable_Parameters())
    test_model_trainer.test()
    test_model_trainer.model.to("cpu")


def global_new_class_test(test_model_trainer, param_model_trainers):
    print('---------------global new class test------------------')
    test_model_trainer.model.to(test_model_trainer.device)
    test_model_trainer.load_learnable_Parameters(aggregation_model_parameter(param_model_trainers))
    test_model_trainer.test()
    test_model_trainer.model.to("cpu")


def main(args):
    args.output_dir = args.output_dir + "_server"
    global_cfg = setup_cfg(args)
    num_client = global_cfg.TRAINER.PROMPTKD.NUM_CLIENT

    local_round = global_cfg.OPTIM.MAX_EPOCH

    if global_cfg.TRAINER.PROMPTKD.DATA_POLLUTION is True:
        global_cfg.OUTPUT_DIR, args.output_dir = set_data_pollution_dir(global_cfg.OUTPUT_DIR, args.output_dir,
                                                                        global_cfg)

    if global_cfg.TRAINER.PROMPTKD.DIFFERENT_NUM_SHOT is True:
        for i in range(num_client):
            global_cfg.OUTPUT_DIR = global_cfg.OUTPUT_DIR + '_' + str(global_cfg.TRAINER.PROMPTKD.CLIENT_SHOTS[i])
            args.output_dir = args.output_dir + '_' + str(global_cfg.TRAINER.PROMPTKD.CLIENT_SHOTS[i])
    setup_logger(global_cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and global_cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print("-----------create server----------")
    global_trainer = build_trainer(global_cfg)
    # global_trainer.load_model("/home/his/zby/PromptKD-main/output_first/first_phase/oxford_pets/shots_16/PromptKD/seed_1")

    new_class_global_trainer_cfg = copy.deepcopy(global_cfg)
    new_class_global_trainer_cfg.DATASET.SUBSAMPLE_CLASSES = "new"
    new_class_global_trainer = build_trainer(new_class_global_trainer_cfg)
    new_class_global_trainer.model.to("cpu")

    clients_trainer, clients_args, unlearning_trainer_cfg, new_class_trainer, new_unlearning_trainer_cfg = creat_clients_trainer(
        args, global_cfg)

    apply_fixed_ratio_pollution(clients_trainer, global_cfg)

    # # ====== [Debug] 打印/导出“污染后张量统计”（全体样本）======
    # if global_cfg.TRAINER.PROMPTKD.DATA_POLLUTION:
    #     pollution_ids = global_cfg.TRAINER.PROMPTKD.POLLUTION_CLIENT_ID  # 你的污染客户端列表 :contentReference[oaicite:2]{index=2}
    #     os.makedirs(global_cfg.OUTPUT_DIR, exist_ok=True)
    #
    #     for cid in pollution_ids:
    #         if cid not in clients_trainer:
    #             print(f"[PollutionStat] client_{cid} 不在 clients_trainer，跳过")
    #             continue
    #
    #         trainer = clients_trainer[cid]
    #         loader = getattr(trainer, "train_loader_x", None)
    #         if loader is None:
    #             print(f"[PollutionStat] client_{cid} 没有 train_loader_x，跳过")
    #             continue
    #
    #         ds = loader.dataset  # 这里就是 FixedRatioPollutedDataset 包装后的 dataset（在 apply_fixed_ratio_pollution 里替换的） :contentReference[oaicite:3]{index=3}
    #
    #         # 你要的是“全部样本统计”，建议写 CSV，控制台会非常长
    #         csv_path = os.path.join(global_cfg.OUTPUT_DIR, f"pollution_stats_client{cid}.csv")
    #
    #         if hasattr(ds, "print_all_pollution_stats"):
    #             print(f"[PollutionStat] 开始导出 client_{cid} => {csv_path}")
    #             ds.print_all_pollution_stats(print_before=False, to_csv=csv_path)
    #         else:
    #             print(
    #                 f"[PollutionStat] loader.dataset 没有 print_all_pollution_stats()，"
    #                 f"请确认你已经把带该方法的 FixedRatioPollutedDataset 版本覆盖到 datasets/fixed_ratio_pollution.py"
    #             )

    # debug_print_client_tensors(clients_trainer, max_batch=1, max_img_per_batch=1)
    print_args(args, global_cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    if args.eval_only:
        print("eval_only")
        if global_cfg.TRAINER.PROMPTKD.DIFFERENT_NUM_SHOT is True:
            print("different num shot eval")
            print("different num shot eval")
            different_shots = ''
            for i in range(num_client):
                different_shots += '_' + str(global_cfg.TRAINER.PROMPTKD.CLIENT_SHOTS[i])

            if global_cfg.TRAINER.PROMPTKD.DATA_POLLUTION is True:
                for i in range(num_client):
                    clients_args[i].model_dir = clients_args[i].model_dir + "_server" + '_' \
                                                + str(global_cfg.TRAINER.PROMPTKD.POLLUTION_CLIENT_ID) + '_' \
                                                + str(global_cfg.TRAINER.PROMPTKD.POLLUTION_PERCENTAGE) \
                                                + different_shots + "_client" + str(i)
                    clients_trainer[i].load_model(clients_args[i].model_dir, epoch=args.load_epoch)

            else:
                for i in range(num_client):
                    clients_args[i].model_dir = clients_args[
                                                    i].model_dir + "_server" + different_shots + "_client" + str(i)
                    clients_trainer[i].load_model(clients_args[i].model_dir, epoch=args.load_epoch)
        else:
            for i in range(num_client):
                clients_args[i].model_dir = clients_args[i].model_dir + "_server_client" + str(i)
                clients_trainer[i].load_model(clients_args[i].model_dir, epoch=args.load_epoch)

        global_trainer.load_learnable_Parameters(aggregation_model_parameter(clients_trainer))
        for i in range(num_client):
            print("----------client_{} eval--------".format(i))
            clients_trainer[i].test()
            client_new_class_test(new_class_trainer[i], clients_trainer[i])

        print("----------global eval--------")
        global_trainer.test()
        global_new_class_test(new_class_global_trainer, clients_trainer)

        return

    if not args.no_train:
        # -----------------
        # 联邦训练阶段（带历史精度保存）
        # -----------------
        for r in range(global_cfg.TRAINER.PROMPTKD.ROUND):
            print(f"================ Round {r + 1}/{global_cfg.TRAINER.PROMPTKD.ROUND} ================")

            # ① 各客户端加载全局参数、重置优化器
            for i in range(num_client):
                clients_trainer[i].load_learnable_Parameters(global_trainer.get_learnable_Parameters(mode="VPT"))
                clients_trainer[i].reset_training()

            # ② 本地训练
            print("---------  开始本地训练阶段 ---------")
            for i in range(num_client):
                print(f"Client_{i} local training start ...")
                clients_trainer[i].fed_train()
                # ----------------------
                clients_trainer[i].save_image_feature()
                # ----------------------

            # ③ 联邦聚合
            print("--------- 联邦聚合 ---------")
            global_trainer.load_learnable_Parameters(aggregation_model_parameter(clients_trainer, "VPT"))
            print("[FedAvg] Global model aggregated.")
            global_acc = float(global_trainer.test())
            print(f"全局准确率 {global_acc:.4f}")



            # ④ 后聚合在每个客户端上测试并记录精度
            print(f"--------- 第 {r + 1} 轮评估 ---------")
            for i in range(num_client):
                clients_trainer[i].model.to(clients_trainer[i].device)
                acc = float(clients_trainer[i].test())
                # 初始化或追加历史精度记录
                if not hasattr(clients_trainer[i], "mqa_history"):
                    clients_trainer[i].mqa_history = []
                clients_trainer[i].mqa_history.append(acc)
                # 只保留最近10轮结果
                if len(clients_trainer[i].mqa_history) > 10:
                    clients_trainer[i].mqa_history = clients_trainer[i].mqa_history[-10:]
                print(f"Client_{i} accuracy history (last 10): {clients_trainer[i].mqa_history}")

        verbose = True
        print("[MQA] LQC 检测开始（轮内中位数对齐 + gap 大小(相对) + 多轮计次）")

        low_data_client_list = []

        # 超参数设置
        T_max = 10  # 只看最初T_max 轮（例如最近 5 轮）
        rel_gap_ratio = 0.05  # 相对阈值：gap > med_t * rel_gap_ratio（即低于中位数超过 5%）
        min_round = 3  # 至少有这么多历史轮数才参与检测
        min_clients = 2  # 至少需要这么多有效客户端才检测

        # 1) 收集各客户端历史精度
        client_histories = {
            cid: getattr(clients_trainer[cid], "mqa_history", [])
            for cid in range(num_client)
        }

        max_len = max((len(h) for h in client_histories.values()), default=0)
        if max_len == 0:
            print("[MQA] 所有客户端历史为空，跳过本轮检测。")
        else:
            # 实际使用的轮数 T（不超过 T_max），使用“最近 T 轮”
            T = min(T_max, max_len)
            if verbose:
                print(f"[MQA] 使用最近 {T} 轮进行轮内中位数对齐检测（T_max={T_max}）")

            # 有效客户端：历史长度至少 min_round
            valid_cids = [cid for cid, h in client_histories.items() if len(h) >= min_round]
            if len(valid_cids) < min_clients:
                print(f"[MQA] 有效客户端（历史 ≥ {min_round} 轮）数量不足 {min_clients}，跳过检测。")
            else:
                # 2) 逐轮计算中位数 m_t，并统计每个客户端被判“本轮明显低于中位数”的次数
                #    （只有满足：acc < med_t 且 gap_it > med_t * rel_gap_ratio 才计入）
                below_cnt = {cid: 0 for cid in valid_cids}  # 符合“明显低于”的轮次数

                # 从第 0 轮开始
                start_idx = 0  # 最近 T 轮在 history 中的起始下标

                for t in range(T):
                    round_idx = start_idx + t  # 这一轮在 mqa_history 里的真实下标

                    # 收集这一轮所有有效客户端的精度 acc_{i,round_idx}
                    vals_t = []
                    cid_with_val = []
                    for cid in valid_cids:
                        hist = client_histories[cid]
                        if len(hist) > round_idx:
                            vals_t.append(hist[round_idx])
                            cid_with_val.append(cid)
                    if len(vals_t) == 0:
                        continue

                    vals_t_tensor = torch.tensor(vals_t, dtype=torch.float32)  # 把val_t变成tensor
                    med_t = float(vals_t_tensor.median().item())  # 当轮中位数 m_t

                    if verbose:
                        print(f"[MQA][round {round_idx}] 轮内中位数 med_t={med_t:.4f}")

                    # 计算这一轮每个客户端：若 (1) acc < med_t 且 (2) gap_it > med_t * rel_gap_ratio
                    # 则视为“本轮明显低于中位数”，计次 +1
                    for cid, acc in zip(cid_with_val, vals_t):
                        acc = float(acc)
                        gap_it = max(0.0, med_t - acc)
                        is_obvious = (acc < med_t) and (gap_it > med_t * rel_gap_ratio)
                        if is_obvious:
                            below_cnt[cid] += 1
                        if verbose:
                            print(
                                f"  - client_{cid}: acc={acc:.4f}, gap_it={gap_it:.4f}, "
                                f"{'明显低于中位数' if is_obvious else '未达到明显低于阈值'}"
                            )

                if verbose:
                    print(
                        "[MQA] 各客户端在最近若干轮中被判“明显低于中位数”的次数 below_cnt：",
                        below_cnt
                    )

                # “超过一半轮次”作为出现次数阈值（例如 T=5 → 至少 3 轮）
                min_hits = max(1, T // 2 + 1)

                # 3) 最终判定：若在最近 T 轮中，“明显低于”次数 >= min_hits，
                #    则判定为低质量 / 污染客户端
                for cid in valid_cids:
                    c = below_cnt[cid]
                    if c > min_hits:
                        print(
                            f"[MQA] client_{cid}: below_cnt={c} ≥ {min_hits} → 判定为低质量 / 污染客户端"
                        )
                        low_data_client_list.append(cid)
                    else:
                        if verbose:
                            print(
                                f"[MQA] client_{cid}: below_cnt={c} < {min_hits} → 视为正常客户端"
                            )

                low_data_client_list = sorted(set(low_data_client_list))
                # 写死低质量
                # fixed_low_data_client = [0]

                # temp = low_data_client_list.copy()
                # low_data_client_list = fixed_low_data_client.copy()
                # # print("[识别出低质量客户端列表]", temp)
                # print("[====================]", fixed_low_data_client)
                print("[MQA] 最终识别低质量客户端列表:", low_data_client_list)

        print("\n=== Client Accuracy History (Last 10 Rounds) ===")
        # for cid in range(num_client):
        # print(f"Client_{cid}: {clients_trainer[cid].mqa_history}")
        for client_index in low_data_client_list:
            print("低质量客户端{}开始忘却".format(client_index))
            clients_trainer[client_index].model.to(clients_trainer[client_index].device)
            clients_trainer[client_index].model = clients_trainer[client_index].unlearning_process()
            # clients_trainer[client_index].model = copy.deepcopy(clients_trainer[client_index].model)

            # print("---------before unlearning-----------")
            # clients_trainer[client_index].test()

        for i in range(num_client):
            print("聚合前")
            clients_trainer[i].test()
        # print(global_trainer)
        print("-----------聚合后全局模型---------")
        global_trainer.load_learnable_Parameters(aggregation_model_parameter(clients_trainer, "VPT"))
        for i in range(num_client):
            print("聚合后")
            clients_trainer[i].load_learnable_Parameters(global_trainer.get_learnable_Parameters(mode="VPT"))
            clients_trainer[i].test()

        for j in range(3):
            print(f"-----------------recover round {j}----------------")
            # for i in range(len(low_data_client_list), num_client):
            for i in range(num_client):
                if i not in low_data_client_list:
                    clients_trainer[i].load_learnable_Parameters(global_trainer.get_learnable_Parameters(mode="VPT"))
                    clients_trainer[i].test()
                    clients_trainer[i].reset_training()
            print("-----------------start train--------------------")
            # for i in range(len(low_data_client_list), num_client):
            for i in range(num_client):
                if i not in low_data_client_list:
                    print("-------client_{} is training-------".format(i))
                    print("-------client_{}_fed_first_phase--------".format(i))

                    clients_trainer[i].fed_train()

            new_clients_trainer = [clients_trainer[i] for i in range(num_client) if i not in low_data_client_list]
            global_trainer.load_learnable_Parameters(aggregation_model_parameter(new_clients_trainer, "VPT"))

        for i in range(num_client):
            if i not in low_data_client_list:
                clients_trainer[i].load_learnable_Parameters(global_trainer.get_learnable_Parameters(mode="VPT"))
                clients_trainer[i].test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--dirichlet", action="store_true", help="use Dirichlet client data split"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument("--second-phase", action="store_true", help="second-phase")
    parser.add_argument("--data-detection", action="store_true", help="evaluation only")
    args = parser.parse_args()
    main(args)
    print("运行完毕")

# CUDA_VISIBLE_DEVICES=1 bash scripts/federated-promptkd/different_numshot_first_train.sh oxford_pets 1
# CUDA_VISIBLE_DEVICES=1 bash scripts/federated-promptkd/different_numshot_first_train.sh stanford_cars 1
# CUDA_VISIBLE_DEVICES=1 bash scripts/federated-promptkd/fed_train_mqa.sh oxford_pets 3 --dirichlet
