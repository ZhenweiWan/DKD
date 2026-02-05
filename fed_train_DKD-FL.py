import argparse
import copy
from datetime import datetime
import math
import random
from collections import defaultdict

import numpy as np
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet
import datasets.tinyimagenet200

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.maple
import trainers.independentVL
import trainers.src_1
import trainers.promptkd
from federated_learning import aggregation_model_parameter


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



def generate_color_jitter_params(brightness=0.3, contrast=0.3, saturation=0.3):
    params = {
        'brightness': random.uniform(1 - brightness, 1 + brightness),
        'contrast': random.uniform(1 - contrast, 1 + contrast),
        'saturation': random.uniform(1 - saturation, 1 + saturation),
    }
    return params


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
        # 添加数据污染
        if cfg.TRAINER.PROMPTKD.DATA_POLLUTION is True:
            if not (len(pollution_client_ID) <= num_client):
                raise ValueError("polluting clients exceeds the number of clients")

            if i in pollution_client_ID:
                clients_cfg[i].TRAINER.PROMPTKD.CLIENT_DATA_POLLUTION = True
                print("客户端需要污染")
                # transforms_list = list(cfg.INPUT.TRANSFORMS)
                # # transforms_list.append('data_pollution')
                # clients_cfg[i].INPUT.TRANSFORMS = transforms_list

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

    for i in new_class_trainer_cfg:
        print(f"----------------create new class client {i}---------------------")
        new_class_trainer[i] = build_trainer(new_class_trainer_cfg[i])
        new_class_trainer[i].model.to("cpu")

        if args.dirichlet:

            total_train_data = []
            # dirichlet_client_num = num_client - len(pollution_client_ID)
            dirichlet_client_num = num_client
            for c_index in range(dirichlet_client_num):

                temp_train_data = clients_trainer[c_index].get_train_x_dataset()
                for item in temp_train_data:
                    total_train_data.append(item)

            alpha = 1  # 小于1代表更偏向非IID，比例从这里修改 0.3/0.5
            new_client_data = split_data_by_dirichlet(total_train_data, dirichlet_client_num, alpha)

            # ======= 打印每个客户端的类别分布 =======
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
            print("[debug]",type(clients_trainer[0].dm.dataset))
            for c_index in range(dirichlet_client_num):
                clients_trainer[c_index].dm.dataset.set_train_x(new_client_data[c_index])
                clients_trainer[c_index].rebulid_train_loader_x()

        return clients_trainer, clients_args, unlearning_trainer_cfg, new_class_trainer, new_unlearning_trainer_cfg


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

def print_current_time():
    now = datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

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
    print_args(args, global_cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    if args.eval_only:
        print("eval_only")
        if global_cfg.TRAINER.PROMPTKD.DIFFERENT_NUM_SHOT is True:
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
            print_current_time()
            # ① 各客户端加载全局参数、重置优化器
            for i in range(num_client):
                clients_trainer[i].load_learnable_Parameters(global_trainer.get_learnable_Parameters(mode="VPT"))
                clients_trainer[i].reset_training()

            # ② 本地训练
            print("---------  开始本地训练阶段 ---------")
            for i in range(num_client):
                print(f"Client_{i} local training start ...")
                clients_trainer[i].fed_train()

            # ③ 联邦聚合
            print("--------- 联邦聚合 ---------")
            global_trainer.load_learnable_Parameters(aggregation_model_parameter(clients_trainer, "all"))
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
        "--dirichlet", action="store_true",help="use Dirichlet client data split"
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



# CUDA_VISIBLE_DEVICES=1 bash scripts/federated-promptkd/fed_train_only.sh oxford_pets 3
# CUDA_VISIBLE_DEVICES=0 bash scripts/federated-promptkd/fed_train_only.sh stanford_cars 3