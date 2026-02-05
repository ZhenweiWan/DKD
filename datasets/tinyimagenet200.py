import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class TinyImageNet200(DatasetBase):

    dataset_dir = "tiny-imagenet-200"

    def __init__(self, cfg):
        # 根目录
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        # 这里把整个 tiny-imagenet-200 作为前缀，后面保存 split 时会存相对路径
        self.image_dir = self.dataset_dir

        # 第一次运行会自动生成，之后直接读取
        self.split_path = os.path.join(self.dataset_dir, "split_zzh_TinyImageNet200.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_zzh_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            # 已经有 split.json：直接用
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            # 没有 split.json：从 Tiny-ImageNet 的原始结构自动读数据并划分
            train, val, test = self.read_and_split_data()
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        # ===== few-shot 处理 =====
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(
                self.split_fewshot_dir,
                f"shot_{num_shots}-seed_{seed}.pkl"
            )

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                print(
                    f"Generating {num_shots}-shot subset for TinyImageNet200 "
                    f"(seed={seed})"
                )
                # 每类 num_shots 个样本；val 最多 4-shot
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(
                    val, num_shots=min(num_shots, 4)
                )
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        # ===== base / novel 类划分（和 Food101 一样的逻辑）=====
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES

        if cfg.TRAINER.NAME == "PromptKD":
            if getattr(cfg.TRAINER, "MODAL", "") == "base2novel":
                # 训练用所有类；验证用 base；测试用 novel
                train_x, _, _ = OxfordPets.subsample_classes(
                    train, val, test, subsample="all"
                )
                _, _, test_base = OxfordPets.subsample_classes(
                    train, val, test, subsample="base"
                )
                _, _, test_novel = OxfordPets.subsample_classes(
                    train, val, test, subsample="new"
                )
                super().__init__(train_x=train_x, val=test_base, test=test_novel)
            elif getattr(cfg.TRAINER, "MODAL", "") == "cross":
                # 按 SUBSAMPLE_CLASSES 选一半类做 cross-domain
                train, _, test = OxfordPets.subsample_classes(
                    train, val, test, subsample=subsample
                )
                super().__init__(train_x=train, val=test, test=test)
            else:
                # 默认行为：和其他 trainer 一样
                train, _, test = OxfordPets.subsample_classes(
                    train, val, test, subsample=subsample
                )
                super().__init__(train_x=train, val=test, test=test)
        # else:
        #     # 其他 trainer：和 OxfordPets、Food101 一样，val=test
        #     train, _, test = OxfordPets.subsample_classes(
        #         train, val, test, subsample=subsample
        #     )
        #     super().__init__(train_x=train, val=test, test=test)
        #
        # # 方便有的 trainer 取所有类名（和 OxfordPets 一致）
        # self.all_classnames = OxfordPets.get_all_classnames(train, val, test)
        else:
            # 其他 trainer：和 OxfordPets、Food101 一样，val=test
            train, _, test = OxfordPets.subsample_classes(
                train, val, test, subsample=subsample
            )
            super().__init__(train_x=train, val=test, test=test)

        # ========= 构建所有类名列表 =========
        # 有些 trainer（例如 PromptKD）会用到 self.all_classnames
        lab2cname = {}
        for item in train + val + test:
            # 后面的覆盖前面的没关系，同一个 label 对应的 classname 是一样的
            lab2cname[item.label] = item.classname

        # 按 label id 从 0~(C-1) 的顺序排好
        self.all_classnames = [lab2cname[i] for i in range(len(lab2cname))]

    def set_train_x(self, data):
        self._train_x = data

    # =======================
    # Tiny-ImageNet 读数据 & 划分
    # =======================
    def read_and_split_data(self, p_val=0.2):
        """
        从 tiny-imagenet-200 原始结构中读取数据并划分为 train/val/test：

        - 使用 train/ 里的 500*200 张图作为 train+val，
          按每类 80/20 做 train/val 划分（复用 OxfordPets.split_trainval 做分层划分）。
        - 使用官方 val/ 里的 50*200 张图作为 test。
        - test 的标签来自 val/val_annotations.txt。
        """
        # 1) 读取 wnids.txt（200 个 synset id）
        wnids_path = os.path.join(self.dataset_dir, "wnids.txt")
        if not os.path.exists(wnids_path):
            raise FileNotFoundError(
                f"Cannot find wnids.txt at {wnids_path}. "
                f"Please make sure tiny-imagenet-200 is correctly placed."
            )
        with open(wnids_path, "r") as f:
            wnids = [line.strip() for line in f if line.strip()]

        # 2) 读取 words.txt（wnid -> 类别名称）
        words_path = os.path.join(self.dataset_dir, "words.txt")
        wnid2name = {}
        if os.path.exists(words_path):
            with open(words_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 格式：wnid \t description, ...
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    wnid = parts[0].strip()
                    desc = parts[1].split(",")[0].strip()  # 取第一个描述词
                    wnid2name[wnid] = desc.replace(" ", "_").lower()

        def get_classname(wnid):
            if wnid in wnid2name:
                return wnid2name[wnid]
            # 兜底：直接用 wnid
            return wnid

        wnid2label = {wnid: idx for idx, wnid in enumerate(wnids)}

        # 3) 读取 train/ 下所有类别的图片，做 train+val
        trainval_items = []
        train_root = os.path.join(self.dataset_dir, "train")
        for wnid in wnids:
            class_dir = os.path.join(train_root, wnid, "images")
            if not os.path.isdir(class_dir):
                # 有些版本可能是 train/wnid/ 下面直接是图片，也顺带兼容一下
                class_dir = os.path.join(train_root, wnid)
                if not os.path.isdir(class_dir):
                    print(f"[Warning] cannot find images for {wnid} under train/")
                    continue

            for fname in os.listdir(class_dir):
                fname_lower = fname.lower()
                if not (fname_lower.endswith(".jpeg")
                        or fname_lower.endswith(".jpg")
                        or fname_lower.endswith(".png")):
                    continue
                impath = os.path.join(class_dir, fname)
                label = wnid2label[wnid]
                classname = get_classname(wnid)

                # 注意：为了和 split.json 存相对路径的逻辑兼容，
                # 这里只要保证 impath 以 self.image_dir 为前缀即可。
                item = Datum(impath=impath, label=label, classname=classname)
                trainval_items.append(item)

        # 按类别分层划分 train / val（和 OxfordPets/Dataset 一样的策略）
        train, val = OxfordPets.split_trainval(trainval_items, p_val=p_val)

        # 4) 读取官方 val 作为 test
        test_items = []
        val_root = os.path.join(self.dataset_dir, "val")
        val_images_dir = os.path.join(val_root, "images")
        anno_file = os.path.join(val_root, "val_annotations.txt")
        if not os.path.exists(anno_file):
            raise FileNotFoundError(
                f"Cannot find val_annotations.txt at {anno_file}."
            )

        # 格式：image_name  wnid  x  y  w  h（以 tab 分隔）
        with open(anno_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                fname, wnid = parts[0], parts[1]
                if wnid not in wnid2label:
                    # 理论上不会发生，稳一下
                    continue
                label = wnid2label[wnid]
                classname = get_classname(wnid)
                impath = os.path.join(val_images_dir, fname)
                if not os.path.exists(impath):
                    # 有些版本路径可能不同，这里先简单跳过
                    continue
                item = Datum(impath=impath, label=label, classname=classname)
                test_items.append(item)

        print(
            f"TinyImageNet200 loaded: "
            f"{len(train)} train, {len(val)} val, {len(test_items)} test"
        )
        return train, val, test_items