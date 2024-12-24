# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from transformers import CLIPModel, CLIPTokenizer

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class DINOv2BackBone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.body.eval()
        self.num_channels = 384
    
    @torch.no_grad()
    def forward(self, tensor):
        xs = self.body.forward_features(tensor)["x_norm_patchtokens"]
        od = OrderedDict()
        od["0"] = xs.reshape(xs.shape[0], 16, 16, 384).permute(0, 3, 2, 1) # Note: hard-coded for 16x16 patches
        return od

class OneHotBackBone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vocab_dim = 15
        self.output_dim = 512
        self.hidden_dim = 1024
        
        # self.body = nn.Sequential(
        #     nn.Linear(self.vocab_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.output_dim),
        # )
        
        # New Version:!!!!
        self.body = nn.Sequential(
            nn.Linear(self.vocab_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim)
        )
        
        # 使用正交初始化确保task embedding初始就有明显区分
        nn.init.orthogonal_(self.body[0].weight, gain=1.4)  # 增大gain使区分更明显
        nn.init.orthogonal_(self.body[3].weight, gain=1.4)
        
        # 预定义所有可能的语言指令
        self.all_possible_lang = [
            "open the middle drawer of the cabinet",
            "put the bowl on the stove",
            "put the wine bottle on top of the cabinet",
            "open the top drawer and put the bowl inside",
            "put the bowl on top of the cabinet",
            "push the plate to the front of the stove",
            "put the cream cheese in the bowl",
            "turn on the stove",
            "put the bowl on the plate",
            "put the wine bottle on the rack",
        ]
        
        # 为每个语言创建对应的 index
        self.lang_to_index = {lang: idx for idx, lang in enumerate(self.all_possible_lang)}
        
        cache_name = "/home/zhaoyixiu/ISR_project/CLIP/tokenizer"
        self.tokenizer = CLIPTokenizer.from_pretrained(cache_name)
        
        # 预计算每个语言的 token ids
        self.lang_token_ids = {}
        for lang in self.all_possible_lang:
            tokens = self.tokenizer(
                lang, 
                padding='max_length', 
                truncation=True, 
                max_length=25,
                return_tensors="pt"
            )
            self.lang_token_ids[lang] = (tokens.input_ids[0].cuda(), tokens.attention_mask[0].cuda())
    
    def _find_matching_lang(self, input_ids, attention_mask):
        # 对于每个输入的 token 序列，找到匹配的预定义语言
        valid_length = attention_mask.sum().item()
        input_tokens = input_ids[:valid_length]
        
        for lang, (stored_ids, stored_mask) in self.lang_token_ids.items():
            stored_length = stored_mask.sum().item()
            
            # print("lang=", lang)
            # print("valid_length=", valid_length, "stored_length=", stored_length)
            # print("input_tokens=", input_tokens)
            # print("stored_ids=", stored_ids[:valid_length])
            
            if valid_length == stored_length and torch.all(input_tokens == stored_ids[:valid_length]):
                return lang
        return None
    
    @torch.no_grad()
    def forward(self, tokenized_inputs):
        if len(tokenized_inputs["input_ids"].shape) == 3:
            bsz, _, max_len = tokenized_inputs["input_ids"].shape
            tokenized_inputs["input_ids"] = tokenized_inputs["input_ids"].reshape(-1, max_len)
            tokenized_inputs["attention_mask"] = tokenized_inputs["attention_mask"].reshape(-1, max_len)
        
        batch_size = tokenized_inputs["input_ids"].shape[0]
        device = tokenized_inputs["input_ids"].device
        
        # 初始化输出 tensor
        text_features = torch.zeros((batch_size, self.vocab_dim), device=device)
        
        # 对每个序列生成 one-hot 向量
        for i in range(batch_size):
            lang = self._find_matching_lang(
                tokenized_inputs["input_ids"][i],
                tokenized_inputs["attention_mask"][i]
            )
            # print("lang=", lang)
            if lang is not None:
                idx = self.lang_to_index[lang]
                text_features[i, idx] = 1.0
                
        # print("feature 0:", text_features[0][:10])
        
        od = OrderedDict()
        od["text_features"] = self.body(text_features)
        
        # print("text_features=", od["text_features"][0][:10])
        
        return od

class CLIPBackBone(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        super().__init__()
        
        cache_name = "/home/zhaoyixiu/ISR_project/CLIP/model"
        self.body = CLIPModel.from_pretrained(cache_name)
        
        # self.body = CLIPModel.from_pretrained(model_name)
        
        self.body.eval()
    
    @torch.no_grad()
    def forward(self, tokenized_inputs):
        """
        Input: text_inputs (list[str]) - 文本列表
        Output: OrderedDict 包含文本特征的编码
        """
        # tokenized_inputs = self.tokenizer(
        #     text_inputs, 
        #     padding=True, 
        #     truncation=True, 
        #     return_tensors="pt"
        # )
        
        if len(tokenized_inputs["input_ids"].shape) == 3:
            # h, w, d -> hw, d
            bsz, _, max_len = tokenized_inputs["input_ids"].shape
            tokenized_inputs["input_ids"] = tokenized_inputs["input_ids"].reshape(-1, max_len)
            tokenized_inputs["attention_mask"] = tokenized_inputs["attention_mask"].reshape(-1, max_len)
        
        text_features = self.body.get_text_features(**tokenized_inputs)
        
        # text_features = text_features.reshape(bsz, -1, text_features.shape[-1])
        
        od = OrderedDict()
        od["text_features"] = text_features
        
        return od

    
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    # def forward (self, tensor):
    #     xs = self[0](tensor)
    #     pos = self[1](xs)
    #     return xs, pos
    
    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos
    
def build_lang_backbone(args):
    train_backbone = False
    return_interm_layers = args.masks

    if args.lang_backbone == 'CLIP':
        backbone = CLIPBackBone()
    elif args.lang_backbone == 'OneHot':
        backbone = OneHotBackBone()
    else:
        raise NotImplementedError
    
    return backbone

def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    if args.backbone == 'dino_v2':
        backbone = DINOv2BackBone()
    else:
        assert args.backbone in ['resnet18', 'resnet34']
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
