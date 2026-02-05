

import csv
import random
from typing import Iterable, Optional, Any, Dict

import torch
from torch.utils.data import Dataset


class FixedRatioPollutedDataset(Dataset):
    """
    包装原始 dataset，将其中固定比例的样本永远污染（将 img 张量置为 0）
    ratio: 比例（例如 0.99 表示 99% 的样本被污染）

    ✅ 保持原有接口可用：FixedRatioPollutedDataset(base_dataset, ratio)
    ✅ 额外支持（可选，不影响你原代码）：
      - seed: 固定抽样结果，便于复现
      - verbose: 是否打印初始化信息
    """

    def __init__(
        self,
        base_dataset: Dataset,
        ratio: float,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.base = base_dataset
        self.ratio = float(ratio)

        self.N = len(self.base)
        self.num_polluted = int(self.N * self.ratio)

        # 固定抽出污染样本 index（可复现）
        all_indices = list(range(self.N))
        if self.num_polluted > 0:
            rng = random.Random(seed) if seed is not None else random
            self.polluted_indices = set(rng.sample(all_indices, self.num_polluted))
        else:
            self.polluted_indices = set()

        if verbose:
            print(
                f"[FixedRatioPollutedDataset] 样本总数={self.N}, "
                f"污染比例={self.ratio}, 实际污染={self.num_polluted} 张, seed={seed}"
            )

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        sample = self.base[idx]

        # Dassl 一般是 dict: {"img": tensor, "label": tensor, ...}
        if isinstance(sample, dict):
            img = sample["img"]
            if idx in self.polluted_indices and torch.is_tensor(img):
                img = torch.zeros_like(img)

            new_sample = dict(sample)
            new_sample["img"] = img
            return new_sample

        # 或者是 (img, label, ...)
        elif isinstance(sample, (list, tuple)):
            img, label = sample[0], sample[1]
            if idx in self.polluted_indices and torch.is_tensor(img):
                img = torch.zeros_like(img)
            return img, label

        else:
            print(f"[FixedRatioPollutedDataset] 未知 sample 类型: {type(sample)} (idx={idx})")
            return sample


    @staticmethod
    def _label_to_py(label: Any) -> Any:

        if torch.is_tensor(label):
            try:
                if label.numel() == 1:
                    return int(label.item())
            except Exception:
                pass
        return label

    @staticmethod
    def _tensor_stat(x: torch.Tensor) -> Dict[str, Any]:

        x_cpu = x.detach().float().cpu()
        return {
            "shape": tuple(x_cpu.shape),
            "dtype": str(x.dtype),
            "min": float(x_cpu.min().item()),
            "max": float(x_cpu.max().item()),
            "mean": float(x_cpu.mean().item()),
            "std": float(x_cpu.std(unbiased=False).item()),
            "abs_max": float(x_cpu.abs().max().item()),
            "nonzero": int((x_cpu != 0).sum().item()),
            "numel": int(x_cpu.numel()),
            "all_zero": bool(torch.all(x_cpu == 0).item()),
        }

    def print_all_pollution_stats(
        self,
        print_before: bool = True,
        to_csv: Optional[str] = None,
    ):

        writer = None
        f = None
        if to_csv is not None:
            f = open(to_csv, "w", newline="", encoding="utf-8")
            writer = csv.writer(f)
            writer.writerow([
                "idx", "polluted", "label",
                "before_shape", "before_dtype", "before_min", "before_max", "before_mean", "before_std",
                "before_abs_max", "before_nonzero", "before_numel", "before_all_zero",
                "after_shape", "after_dtype", "after_min", "after_max", "after_mean", "after_std",
                "after_abs_max", "after_nonzero", "after_numel", "after_all_zero",
            ])

        polluted_total = 0
        polluted_after_all_zero = 0
        clean_after_all_zero = 0

        for idx in range(len(self)):
            sample = self.base[idx]
            polluted = (idx in self.polluted_indices)

            # 取 img/label
            if isinstance(sample, dict):
                img = sample.get("img", None)
                label = sample.get("label", None)
            elif isinstance(sample, (list, tuple)):
                img = sample[0] if len(sample) > 0 else None
                label = sample[1] if len(sample) > 1 else None
            else:
                print(f"[STAT] idx={idx} unknown sample type: {type(sample)}")
                continue

            label_py = self._label_to_py(label)

            if not torch.is_tensor(img):
                print(f"[STAT] idx={idx} polluted={int(polluted)} label={label_py} img_type={type(img)} (非Tensor，无法统计)")
                continue

            before = self._tensor_stat(img)

            # 构造“污染后”的 img（这里不依赖 __getitem__，避免额外副作用）
            if polluted:
                after_img = torch.zeros_like(img)
            else:
                after_img = img
            after = self._tensor_stat(after_img)

            # 汇总
            if polluted:
                polluted_total += 1
                if after["all_zero"]:
                    polluted_after_all_zero += 1
            else:
                if after["all_zero"]:
                    clean_after_all_zero += 1

            # 打印
            if print_before:
                print(f"[STAT] idx={idx} polluted={int(polluted)} label={label_py}")
                print(f"       before: {before}")
                print(f"       after : {after}")
            else:
                print(f"[STAT] idx={idx} polluted={int(polluted)} label={label_py} after={after}")

            # 写 CSV
            if writer is not None:
                writer.writerow([
                    idx, int(polluted), label_py,
                    before["shape"], before["dtype"], before["min"], before["max"], before["mean"], before["std"],
                    before["abs_max"], before["nonzero"], before["numel"], before["all_zero"],
                    after["shape"], after["dtype"], after["min"], after["max"], after["mean"], after["std"],
                    after["abs_max"], after["nonzero"], after["numel"], after["all_zero"],
                ])


        print("\n===== [Pollution Stats Summary] =====")
        print(f"total_samples = {len(self)}")
        print(f"polluted_samples = {polluted_total} (expect={len(self.polluted_indices)})")
        print(f"polluted_after_all_zero = {polluted_after_all_zero}/{polluted_total}")
        print(f"clean_after_all_zero = {clean_after_all_zero}/{len(self) - polluted_total}")
        print("====================================\n")

        if f is not None:
            f.close()
            print(f"[Saved] CSV => {to_csv}")