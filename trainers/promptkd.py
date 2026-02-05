import copy
import os
import os.path as osp
import types
import time
import numpy
import numpy as np
import timm
import torch
import torch.nn as nn
from torch import optim

from dassl.data.transforms import build_transform

from dassl.data.data_manager import build_data_loader
from torch.nn import functional as F, Identity
from torch.cuda.amp import GradScaler, autocast
from torchvision.transforms import ToTensor

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from torchvision import transforms

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .KD_ours import KD_ours
from .KD_ours_strong import KD_ours_strong
from .imagenet_templates import IMAGENET_TEMPLATES
from tqdm import tqdm
import math

from clip.model import VisionTransformer, convert_weights
from .randaugment import RandAugment

_tokenizer = _Tokenizer()


class Feature_Trans_Module_two_layer(nn.Module):
    def __init__(self, input_dim=100, out_dim=256):
        super(Feature_Trans_Module_two_layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1)
        )

    def forward(self, input_feat):
        final_feat = self.conv1(input_feat.unsqueeze(-1).unsqueeze(-1))

        return final_feat.squeeze(-1).squeeze(-1)


def load_clip_to_cpu_teacher(cfg, zero_shot_model=False):
    backbone_name = cfg.TRAINER.PROMPTKD.TEACHER_NAME
    # url = clip._MODELS[backbone_name]

    if backbone_name == "ViT-B/16":
        model_path = './clip/ViT-B-16.pt'
    elif backbone_name == "ViT-L/14":
        model_path = './clip/ViT-L-14.pt'
    elif backbone_name == "ViT-B/32":
        model_path = './clip/ViT-B-32.pt'
    else:
        print('enter the wrong teacher name.')

    print(f"CLIP Teacher name is {backbone_name}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # We default use PROMPTKD to pretrain our teacher model
    design_details = {"trainer": 'IVLP',
                      "vision_depth": 9,
                      "language_depth": 9,
                      "vision_ctx": 4,
                      "language_ctx": 4}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    # url = clip._MODELS[backbone_name]
    model_path = './clip/ViT-B-16.pt'

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'IVLP',
                      "vision_depth": cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT,
                      "vision_ctx": cfg.TRAINER.PROMPTKD.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.PROMPTKD.N_CTX_TEXT}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # print(f'------prompts size is {prompts.size()}------')
        # print(f'------tokenized prompts size is {tokenized_prompts.size()}------')

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, is_teacher):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.PROMPTKD.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                            "\nPlease use VPT trainer if you want to learn only vision " \
                                                            "branch"
        n_ctx = cfg.TRAINER.PROMPTKD.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PROMPTKD.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.trainer_name = cfg.TRAINER.NAME
        self.train_modal = cfg.TRAINER.MODAL

        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PROMPTKD.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)

        print(f'classnames size is {len(classnames)}')

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        # self.name_lens = name_lens

        if self.train_modal == "base2novel":
            self.register_buffer("token_prefix", embedding[:math.ceil(self.n_cls / 2), :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:math.ceil(self.n_cls / 2), 1 + n_ctx:, :])  # CLS, EOS

            self.register_buffer("token_prefix2", embedding[math.ceil(self.n_cls / 2):, :1, :])  # SOS
            self.register_buffer("token_suffix2", embedding[math.ceil(self.n_cls / 2):, 1 + n_ctx:, :])  # CLS, EOS

        elif self.train_modal == "cross":
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

            self.register_buffer("token_prefix2", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix2", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        # print(f'label is {label}')
        # if label is not None:
        #     prefix = prefix[label]
        #     suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        # print(f'ctx size is {ctx.size()}')

        prefix = self.token_prefix
        # print(f'prefix size is {prefix.size()}')

        suffix = self.token_suffix
        # print(f'suffix size is {suffix.size()}')

        if self.trainer_name == "PromptKD" and self.train_modal == "base2novel":
            # print(f'n_cls is {self.n_cls}')
            prefix = torch.cat([prefix, self.token_prefix2], dim=0)
            suffix = torch.cat([suffix, self.token_suffix2], dim=0)

        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

        self.VPT_image_trans = Feature_Trans_Module_two_layer(512, 718)

        self.cfg = cfg

        self.VPT_image_trans = self.VPT_image_trans.cuda()
        convert_weights(self.VPT_image_trans)

    def forward(self, image, label=None):
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = self.VPT_image_trans(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features, logit_scale


class Adapter1(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter1, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class Adapter2(nn.Module):
    def __init__(self, input_dim, reduction=4, residual_ratio=0.2):
        super(Adapter2, self).__init__()
        self.residual_ratio = residual_ratio
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.to(torch.float32)
        a = self.fc(x)
        x = self.residual_ratio * a + (1 - self.residual_ratio) * x
        return x


class PromptedTimmViT(nn.Module):
    def __init__(self, base_vit, vision_ctx=5, vision_depth=4):
        super().__init__()
        self.vit = base_vit
        self.vision_ctx = vision_ctx
        self.vision_depth = vision_depth
        self.embed_dim = base_vit.pos_embed.shape[-1]
        self.num_patches = base_vit.pos_embed.shape[1] - 1  # excludes cls token

        # Shallow prompt token
        self.prompt_embed = nn.Parameter(torch.randn(1, vision_ctx, self.embed_dim) * 0.02)

        # Deep prompt per transformer layer
        self.deep_prompts = [
            nn.Parameter(torch.randn(vision_ctx, self.embed_dim) * 0.02)
            if i < vision_depth else None
            for i in range(len(base_vit.blocks))
        ]
        self.deep_prompts = nn.ParameterList([p for p in self.deep_prompts if p is not None])

        # Remove classification head
        self.vit.head = nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.vit.patch_embed(x)  # [B, C, H, W] -> [B, N, C]
        cls_token = self.vit.cls_token.expand(B, -1, -1)  # [1, 1, C] -> [B, 1, C]
        x = torch.cat((cls_token, x), dim=1)  # [B, 1+N, C]

        # Add shallow prompt tokens after cls
        prompt = self.prompt_embed.expand(B, -1, -1)  # [B, P, C]
        x = torch.cat([x[:, :1], prompt, x[:, 1:]], dim=1)  # [B, 1+P+N, C]

        # Adjust positional embedding
        pos_embed = self.vit.pos_embed
        if x.size(1) != pos_embed.size(1):
            pos_embed = F.interpolate(pos_embed.permute(0, 2, 1), size=x.size(1), mode='linear',
                                      align_corners=False).permute(0, 2, 1)
        x = x + pos_embed[:, :x.size(1), :]

        x = self.vit.pos_drop(x)

        # Transformer blocks with deep prompts
        for i, blk in enumerate(self.vit.blocks):
            if i < self.vision_depth:
                deep_prompt = self.deep_prompts[i].unsqueeze(0).expand(B, -1, -1)  # [B, P, C]
                x = torch.cat([x, deep_prompt], dim=1)
            x = blk(x)

        x = self.vit.norm(x)
        return x


class CustomViT(nn.Module):
    def __init__(self, base_model, clip_model_teacher):
        super(CustomViT, self).__init__()
        self.logit_scale = clip_model_teacher.logit_scale
        self.base_model = base_model
        # self.adapter = Adapter(base_model.num_features, 4)  # Adapter operates on features
        # convert_weights(self.adapter)
        self.VPT_image_trans = Feature_Trans_Module_two_layer(192, 512)

        self.VPT_image_trans = self.VPT_image_trans.cuda()

        # Remove the original classification head (base_model.head)
        # self.base_model.head = Identity()

    def forward(self, x, label=None):
        logit_scale = self.logit_scale.exp()
        features = self.base_model.forward_features(x)
        # Get the features from ViT
        # adapted_features = self.adapter(features)  # Pass features through adapter
        # adapted_features = adapted_features[:, 0, :]
        features = features[:, 0, :]
        adapted_features = self.VPT_image_trans(features)

        adapted_features = adapted_features / adapted_features.norm(dim=-1, keepdim=True)

        return adapted_features.to(torch.float16), logit_scale


class CustomCLIP_teacher(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, True)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model).cuda()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image=None, label=None):
        prompts = self.prompt_learner()
        # Compute the prompted image and text features
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts.cuda(), tokenized_prompts.cuda())
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits

        logits = logit_scale * image_features @ text_features.t()

        return image_features, text_features, logits


@TRAINER_REGISTRY.register()
class PromptKD(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        # self.train_loader_x = None

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTKD.PREC in ["fp16", "fp32", "amp"]

    def get_train_x_dataset(self):
        return self.dm.dataset.train_x

    def rebulid_train_loader_x(self):
        self.train_loader_x = self.dm.reset_train_loader_x()

    def build_model(self):
        cfg = self.cfg

        classnames = self.dm.dataset.classnames
        self.n_cls = len(classnames)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        # clip_model = load_clip_to_cpu(cfg)
        clip_model_teacher = load_clip_to_cpu_teacher(cfg)

        clip_model = timm.create_model('vit_tiny_patch16_224.augreg_in21k', pretrained=False)

        clip_model.load_state_dict(torch.load("clip/vit_tiny_patch16_224_augreg_in21k.pth"))
        clip_model.eval()

        if cfg.TRAINER.PROMPTKD.PREC == "fp32" or cfg.TRAINER.PROMPTKD.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        # self.model = CustomCLIP(cfg, classnames, clip_model)
        # 添加 prompt 封装器
        prompted_vit = PromptedTimmViT(clip_model, vision_ctx=4, vision_depth=8)
        # self.model = CustomViT(clip_model, clip_model_teacher)
        self.model = CustomViT(prompted_vit, clip_model_teacher)

        self.model_teacher = CustomCLIP_teacher(cfg, classnames, clip_model_teacher)

        if cfg.TRAINER.MODAL == "base2novel":
            # model_path = './teacher_model/' + str(cfg.DATASET.NAME) + '/VLPromptLearner/model-best.pth.tar'

            model_path = 'teacher_model/OxfordPets/VLPromptLearner/model.pth.tar-20'
            # model_path = 'teacher_model/Sun397/VLPromptLearner/model.pth.tar-20'
            # model_path = 'teacher_model/Oxford_flowers/VLPromptLearner/model.pth.tar-20'
            # model_path = 'teacher_model/TinyImageNet200/VLPromptLearner/model.pth.tar-20'

        elif cfg.TRAINER.MODAL == "cross":
            model_path = './teacher_model/ImageNet-xd/VLPromptLearner_large/model.pth.tar-20'

        self.train_modal = cfg.TRAINER.MODAL

        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint["state_dict"]

        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]
        if "prompt_learner.token_prefix2" in state_dict:
            del state_dict["prompt_learner.token_prefix2"]

        if "prompt_learner.token_suffix" in state_dict:
            del state_dict["prompt_learner.token_suffix"]
        if "prompt_learner.token_suffix2" in state_dict:
            del state_dict["prompt_learner.token_suffix2"]

        self.model_teacher.load_state_dict(state_dict, strict=False)
        self.model_teacher.to(self.device)
        self.model_teacher.eval()

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        # for name, param in self.model.named_parameters():
        #     if name_to_update not in name:
        #         # Make sure that VPT prompts are updated
        #         if "VPT" in name:
        #             param.requires_grad_(True)
        #         else:
        #             param.requires_grad_(False)
        #     else:
        #         if "ZS_image_encoder" in name:
        #             param.requires_grad_(False)
        # for param in self.model.base_model.parameters():
        #     param.requires_grad = False
        # self.model.logit_scale.requires_grad = False

        for name, param in self.model.named_parameters():
            if "prompt" not in name and "VPT_image_trans" not in name:
                param.requires_grad = False

        # for name, param in self.model.named_parameters():
        #     if "prompt" not in name:
        #         param.requires_grad = False

        # for name, param in self.model.named_parameters():
        #     param.requires_grad = True

        print("count_learnable_parameters:")
        print(self.count_learnable_parameters())

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer

        self.trainable_list = nn.ModuleList([])
        self.trainable_list.append(self.model)

        self.optim = build_optimizer(self.trainable_list, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH

        self.scaler = GradScaler() if cfg.TRAINER.PROMPTKD.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        self.temperature = cfg.TRAINER.PROMPTKD.TEMPERATURE
        cfgkd = {
            "KD": {
                "TEMPERATURE": 4.0,  # 温度
                "LOSS": {
                    "KD_WEIGHT": 0.1,  # 0.1  第二组 双视角蒸馏损失之间的比例
                    "BT_WEIGHT": 1  # 1
                }
            }
        }

        self.distiller = KD_ours(self.model, self.model_teacher, cfgkd)
        self.distiller_strong = KD_ours_strong(self.model, self.model_teacher, cfgkd)
        self.train_transform_strong = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                RandAugment(2, 10),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

        self.measure_fps()

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        with torch.no_grad():
            tea_image_features, tea_text_features, tea_logits = self.model_teacher(image)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.PROMPTKD.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            if self.cfg.TRAINER.PROMPTKD.SECOND_PHASE:
                image_ft, logit_scale = model(image, label)

                # stu_logits = logit_scale * image_ft @ tea_text_features.t().detach()

                # L_ukd = F.kl_div(
                #     F.log_softmax(stu_logits / self.temperature, dim=1),
                #     F.softmax(tea_logits / self.temperature, dim=1),
                #     reduction='sum',
                # ) * (self.temperature * self.temperature) / stu_logits.numel()  # 求平均
                #
                # loss = self.cfg.TRAINER.PROMPTKD.KD_WEIGHT * L_ukd

                print("=======================================================================")

                stu_logits = logit_scale * image_ft @ tea_text_features.t().detach()

                loss = nn.CrossEntropyLoss()(stu_logits, label)

            # forward
            else:
                trans_images = []
                for imag in image:
                    image_strong = self.train_transform_strong(imag)
                    trans_images.append(image_strong)
                image_weak, image_strong = image, torch.stack(trans_images, dim=0).cuda()
                if self.cfg.TRAINER.PROMPTKD.STRONG:
                    print("666")
                    preds, losses_dict = self.distiller_strong(image_weak=image_weak, image_strong=image_strong,
                                                               target=label)
                else:
                    preds, losses_dict = self.distiller(image_weak=image_weak, target=label)

                # backward
                loss = sum([l.mean() for l in losses_dict.values()])

            optim.zero_grad()
            if optim.__class__.__name__ == 'AdaHessian':
                loss.backward(create_graph=True)
            else:
                loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        # self.get_max_cuda_max_memory()
        return loss_summary

    def NoT_unlearning(self):
        """
        实现 CVPR 2025 NoT (Weight Negation) 算法。
        核心逻辑：通过对特定层权重乘以 -1 来破坏层间协同，随后进行快速微调。
        """
        print("执行 NoT 忘却：正在对指定层进行权重取反...")

        # 根据论文实验，通常对模型的第一层（如投影层）取反效果最好
        # 在你的架构中，VPT_image_trans 是连接特征的关键层
        target_layer_found = False

        # 策略 1: 尝试对 VPT_image_trans（特征转换层）的第一层进行取反
        if hasattr(self.model, 'VPT_image_trans'):
            for name, module in self.model.VPT_image_trans.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    print(f"NoT: 正在对 VPT_image_trans 中的层 '{name}' 进行权重取反")
                    module.weight.data.mul_(-1.0)
                    target_layer_found = True
                    break  # 只对该模块的第一层取反

        # 策略 2: 如果上述没找到，对 Adapter (Mona) 的输入映射层进行取反
        if not target_layer_found and hasattr(self.model, 'adapter'):
            for name, module in self.model.adapter.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    print(f"NoT: 正在对 adapter 中的层 '{name}' 进行权重取反")
                    module.weight.data.mul_(-1.0)
                    target_layer_found = True
                    break

        # 策略 3: 作为保底方案，对 base_model (ViT) 的 patch 投影层取反
        if not target_layer_found:
            # 对于 timm 的 ViT 或 CLIP 的视觉部分，通常是 patch_embed
            for name, param in self.model.named_parameters():
                if 'patch_embed.proj.weight' in name or 'conv1.weight' in name:
                    print(f"NoT: 正在对 base_model 的投影层 '{name}' 进行权重取反")
                    param.data.mul_(-1.0)
                    target_layer_found = True
                    break

        if target_layer_found:
            print("NoT 权重取反完成。请接下来使用保留数据（Retained Data）进行 1-5 轮的 Fine-tuning。")
        else:
            print("警告：未找到合适的取反层，请检查模型构造。")

        return self.model

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_prefix2" in state_dict:
                del state_dict["prompt_learner.token_prefix2"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]
            if "prompt_learner.token_suffix2" in state_dict:
                del state_dict["prompt_learner.token_suffix2"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "train":
            data_loader = self.train_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = self.parse_batch_test(batch)

            with torch.no_grad():
                tea_image_features, tea_text_features, tea_logits = self.model_teacher(image, label)

            image_ft, logit_scale = self.model(image, label)

            if self.train_modal == "base2novel":
                if split == "val":
                    output = logit_scale * image_ft @ tea_text_features[:math.ceil(self.n_cls / 2), :].t()
                elif split == "test":
                    output = logit_scale * image_ft @ tea_text_features[math.ceil(self.n_cls / 2):, :].t()
            elif self.train_modal == "cross":
                output = logit_scale * image_ft @ tea_text_features.t()

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    @torch.no_grad()
    def get_low_dataset(self, dm, cfg):
        tfm = build_transform(cfg, is_train=False)
        data_loader = build_data_loader(
            cfg=cfg,
            sampler_type="SequentialSampler",
            data_source=dm.dataset.train_x,
            batch_size=1,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm,
            is_train=False,
        )
        true_num = 0
        low_quality_dataset = []
        dataset_pseudo_label = copy.deepcopy(dm.dataset.train_x)

        dataset_size = len(dm.dataset.train_x)
        pollution_data_size = 0
        for data in dm.dataset.train_x:
            if data.pollution is True:
                pollution_data_size += 1

        pred_pollution_data_size = 0
        error_pollution_data_size = 0
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            pred = output.max(1)[1]
            if torch.all(pred.eq(label)):
                true_num += 1
            else:
                if dm.dataset.train_x[batch_idx].pollution is True:
                    pred_pollution_data_size += 1
                else:
                    error_pollution_data_size += 1
                dataset_pseudo_label[batch_idx]._label = pred.item()
                low_quality_dataset.append(dm.dataset.train_x[batch_idx])

        print("劣质数据检测准确率：{:.2f}%".format(100 * (pred_pollution_data_size / pollution_data_size)))
        print(
            "优质数据误检测率：{:.2f}%".format(100 * (error_pollution_data_size / (dataset_size - pollution_data_size))))
        return low_quality_dataset, dataset_pseudo_label

    def get_dm(self):
        return self.dm

    @torch.no_grad()
    def measure_fps(self, input_shape=(1, 3, 224, 224), device='cuda', warm_up=10, test_runs=100):
        """
        计算模型的 FPS (Frames Per Second)

        参数:
            model:      待测模型
            input_shape:输入张量的形状 (batch_size, channels, height, width)
            device:     设备 ('cuda' 或 'cpu')
            warm_up:    预热次数 (避免 CUDA 初始化等开销影响)
            test_runs:  正式测试循环次数
        """
        # 1. 准备设备和数据
        model = self.model
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()  # 切换到评估模式

        # 创建随机输入数据
        dummy_input = torch.randn(input_shape).to(device)

        print(f"正在测试 FPS (Device: {device}, Shape: {input_shape})...")

        # 2. 预热 (Warm-up)
        # GPU 刚开始运行会有一些初始化开销，预热可以让硬件进入状态
        for _ in range(warm_up):
            _ = model(dummy_input)

        # 确保预热结束
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # 3. 正式测速
        start_time = time.time()
        for _ in range(test_runs):
            _ = model(dummy_input)

            # 注意：这行代码通常放在循环外，但为了模拟真实流式处理，
            # 有时需要确保每次都执行完。对于纯吞吐量测试，放在循环外即可。
            # 这里为了计算纯粹的 GPU Kernel 时间，我们在最后统一同步。

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        # 4. 计算结果
        total_time = end_time - start_time
        avg_time_per_run = total_time / test_runs

        # FPS = (Batch Size * Test Runs) / Total Time
        # 如果 input_shape[0] 是 Batch Size
        batch_size = input_shape[0]
        fps = (test_runs * batch_size) / total_time

        print(f"----- 测试结果 -----")
        print(f"Total Time:   {total_time:.4f} s")
        print(f"Latency:      {avg_time_per_run * 1000:.2f} ms/batch")
        print(f"FPS:          {fps:.2f}")

        return fps

    def get_max_cuda_max_memory(self):
        used = torch.cuda.max_memory_allocated() / 1024 / 1024
        return print(f"memery is {used:.2f}MB")

    def count_learnable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def print_learnable_parameters(self):
        model = self.model
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Parameter name: {name}, Shape: {param.shape}")

    # def get_learnable_Parameters(self):
    #     model = self.model
    #     return {name: param.data for name, param in model.named_parameters() if param.requires_grad}

    def get_learnable_Parameters(self, mode="all"):
        model = self.model
        filtered_params = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if mode == "VPT" and "VPT" not in name:
                continue
            elif mode == "prompt" and "prompt" not in name:
                continue
            filtered_params[name] = param.data
        return filtered_params

    def load_learnable_Parameters(self, state_dict):
        model = self.model
        model.load_state_dict(state_dict, strict=False)
        # model.load_state_dict(state_dict)

    def reset_training(self):
        cfg = self.cfg
        self.model.train()
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self._models["VLPromptLearner"] = self.model
        self._optims["VLPromptLearner"] = self.optim
        self._scheds["VLPromptLearner"] = self.sched
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTKD.PREC == "amp" else None
        self.previous_model_gpa = None

    def set_train_dataset(self, train):
        for index, data in enumerate(train):
            try:
                self.dm.dataset.train_x[index]._label = data._label
                self.dm.dataset.train_x[index]._impath = data._impath
                self.dm.dataset.train_x[index]._domain = data._domain
                self.dm.dataset.train_x[index]._classname = data._classname
                self.dm.dataset.train_x[index]._pollution = data._pollution
            except:
                pass

    def set_test_dataset(self, test):
        for index, data in enumerate(test):
            self.dm.dataset.test[index]._label = data._label
            self.dm.dataset.test[index]._impath = data._impath
            self.dm.dataset.test[index]._domain = data._domain
            self.dm.dataset.test[index]._classname = data._classname
            self.dm.dataset.test[index]._pollution = data._pollution

    def fed_train(self):
        if next(self.model.parameters()).device.type == 'cpu':
            self.model.to(self.device)
        self.train()
        if self.cfg.TRAINER.PROMPTKD.SECOND_PHASE:
            self.model.to("cpu")
        torch.cuda.empty_cache()

    def get_dataset_len(self):
        return len(self.dm.dataset.train_x)

    def set_model_learnableParam(self):
        # name_to_update = "prompt_learner"
        # for name, param in self.model.named_parameters():
        #     if name_to_update not in name:
        #         # Make sure that VPT prompts are updated
        #         if "VPT" in name:
        #             param.requires_grad_(True)
        #         else:
        #             param.requires_grad_(False)
        #     else:
        #         if "ZS_image_encoder" in name:
        #             param.requires_grad_(False)
        #     if "ctx" in name:
        #         param.requires_grad_(True)
        for name, param in self.model.named_parameters():
            if "prompt" not in name and "VPT_image_trans" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

            # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

    def save_image_feature(self):
        print("saving future")
        model = self.model
        dataloader = self.test_loader
        # model.prompt_learner.training = True
        with torch.no_grad():
            all_feature = []
            feature_label = []
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                image, label = self.parse_batch_train(batch)
                # feature_label.append(label.item())
                feature_label.extend(label.tolist())
                image_feature, logits = model(image, label)

                all_feature.append(image_feature)
            result = torch.cat(all_feature, dim=0)
            print(result.shape)
            print(feature_label)

            result_np = result.cpu().numpy()
            feature_label_np = numpy.array(feature_label)
            np.save('/media/yht/37aa263c-dee5-4f68-ac01-07813ff4a404/wzw/Prompt/images/feature'
                    '/image_features_sun397_r.npy',
                    result_np)
            np.save('/media/yht/37aa263c-dee5-4f68-ac01-07813ff4a404/wzw/Prompt/images/feature'
                    '/image_features_sun397_label_r.npy',
                    feature_label_np)
            print("Features saved to image_features.npy")

    def unlearning_process(self):
        self.reset_training()
        unlearning_model = self.model
        for name, param in unlearning_model.named_parameters():
            if "prompt" not in name and "VPT_image_trans" not in name:
                param.requires_grad = False

        # Double check
        enabled = set()
        for name, param in unlearning_model.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        print(enabled)

        unlearning_optim = optim.SGD(unlearning_model.parameters(), lr=0.0001)

        print("---------start unlearning----------")
        for unlearning_epoch in range(1):
            print("---------第{}轮忘却---------".format(unlearning_epoch + 1))
            for batch_idx, batch in enumerate(self.train_loader_x):
                image, label = self.parse_batch_train(batch)

                trans_images = []
                for imag in image:
                    image_strong = self.train_transform_strong(imag)
                    trans_images.append(image_strong)
                image_weak, image_strong = image, torch.stack(trans_images, dim=0).cuda()
                if self.cfg.TRAINER.PROMPTKD.STRONG:

                    preds, losses_dict = self.distiller_strong(image_weak=image_weak, image_strong=image_strong,
                                                               target=label)
                else:
                    preds, losses_dict = self.distiller(image_weak=image_weak, target=label)

                # backward
                # loss = 1.0 / sum([l.mean() for l in losses_dict.values()])
                loss = - (sum([l.mean() for l in losses_dict.values()]))  # 取负操作

                # print("loss_total:", loss_total.item())

                unlearning_optim.zero_grad()
                loss.backward()
                unlearning_optim.step()

        return unlearning_model

    # def save_data(self, cfg):
    #     if cfg.TRAINER.PROMPTSRC.CLIENT_LIGHT_VARIATIONS is True or cfg.TRAINER.PROMPTSRC.CLIENT_COLOUR_VARIATIONS is True or cfg.TRAINER.PROMPTSRC.CLIENT_DATA_POLLUTION is True:
    #         print("data sample is saving")
    #         for i in tqdm(range(len(self.dm.dataset.train_x))):
    #             if cfg.TRAINER.PROMPTSRC.CLIENT_DATA_POLLUTION is True:
    #                 save_pollution_sample(self.dm.dataset.train_x[i].impath)
    #             if cfg.TRAINER.PROMPTSRC.CLIENT_LIGHT_VARIATIONS is True:
    #                 save_data_sample(self.dm.dataset.train_x[i].impath, cfg, "light", i)
    #             if cfg.TRAINER.PROMPTSRC.CLIENT_COLOUR_VARIATIONS is True:
    #                 save_data_sample(self.dm.dataset.train_x[i].impath, cfg, "colour", i)
    #             if i == 50:
    #                 break

    def model_add_adapter(self):

        # adapter = Adapter2(512, reduction=48, residual_ratio=0.1)
        # adapter1 = Mona(in_dim=192)
        # self.model.adapter1 = adapter1.to(self.device)
        adapter = Mona(in_dim=512)
        self.model.adapter = adapter.to(self.device)
        self.model.forward = types.MethodType(new_forward, self.model)

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.adapter.parameters():
            param.requires_grad = True
        # for param in self.model.adapter1.parameters():
        #     param.requires_grad = True
        print(f"Model's trainable parameters:")

        for name, param in self.model.named_parameters():
            # Check if the parameter requires gradients (i.e., is trainable)
            if param.requires_grad:
                print(f"{name}: {param.shape}")


def new_forward(self, x, label=None):
    logit_scale = self.logit_scale.exp()
    features = self.base_model.forward_features(x)
    # Get the features from ViT
    adapted_features = features[:, 0, :]
    # features = features[:, 0, :]
    # adapted_features = self.adapter1(adapted_features)  # Pass features through adapter
    adapted_features = self.VPT_image_trans(adapted_features)
    adapted_features = self.adapter(adapted_features)  # Pass features through adapter
    adapted_features = adapted_features / adapted_features.norm(dim=-1, keepdim=True)

    return adapted_features.to(torch.float16), logit_scale
