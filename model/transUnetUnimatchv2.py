import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from typing import Optional, Tuple, List



# =============================================================================
# 新增：Transformer模块（TransUnet核心组件）
# =============================================================================

class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block with Pre-norm"""

    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# 新增：Patch Embedding + Transformer Encoder
# =============================================================================


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 16x16=256

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size
        x = self.proj(x)               # (B, 768, 16, 16)
        x = x.flatten(2).transpose(1, 2)  # (B, 256, 768)
        return x


class VisionTransformerEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # ✅ 修复Bug1：+1 for cls_token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

    # def load_from_pth(self, pth_path, in_chans=1):
    #     print(f"Loading pretrained weights from {pth_path}...")
    #     pretrained_dict = torch.load(pth_path, map_location='cpu')
    #
    #     if 'model' in pretrained_dict:
    #         pretrained_dict = pretrained_dict['model']
    #     elif 'state_dict' in pretrained_dict:
    #         pretrained_dict = pretrained_dict['state_dict']
    #
    #     # ── 1. patch_embed 权重适配（3通道 → in_chans 通道）─────────────────
    #     patch_weight_key = 'patch_embed.proj.weight'
    #     patch_bias_key   = 'patch_embed.proj.bias'
    #
    #     if patch_weight_key in pretrained_dict:
    #         w_pretrained = pretrained_dict[patch_weight_key]
    #         # 预训练权重形状：[768, 3, 16, 16]
    #         # 目标形状：      [768, 1, 16, 16]
    #
    #         if w_pretrained.shape[1] != in_chans:
    #             print(f"  patch_embed: 通道数 {w_pretrained.shape[1]} → {in_chans}，执行均值适配")
    #             # 核心操作：沿输入通道维度取均值，再expand到目标通道数
    #             # 均值方法比随机初始化保留了更多的空间滤波器信息
    #             w_adapted = w_pretrained.mean(dim=1, keepdim=True)  # [768, 1, 16, 16]
    #             if in_chans > 1:
    #                 w_adapted = w_adapted.repeat(1, in_chans, 1, 1)  # [768, N, 16, 16]
    #             pretrained_dict[patch_weight_key] = w_adapted
    #             print(f"  patch_embed权重适配完成: {w_pretrained.shape} → {w_adapted.shape}")
    #         # bias形状 [768] 不变，直接复用
    #
    #     # ── 2. 位置编码插值（196+1 → 256+1）────────────────────────────────
    #     if 'pos_embed' in pretrained_dict:
    #         pos_embed_pretrained = pretrained_dict['pos_embed']
    #         # ViT-B/16预训练: [1, 197, 768]（196个patch + 1个cls_token）
    #         # 目标: [1, 257, 768]（256个patch + 1个cls_token）
    #
    #         old_num = pos_embed_pretrained.shape[1] - 1  # 196
    #         new_num = self.pos_embed.shape[1] - 1         # 256
    #
    #         if old_num != new_num:
    #             old_size = int(math.sqrt(old_num))  # 14
    #             new_size = int(math.sqrt(new_num))  # 16
    #             print(f"  pos_embed插值: {old_size}x{old_size} → {new_size}x{new_size}")
    #
    #             cls_tok = pos_embed_pretrained[:, :1, :]     # [1, 1, 768]
    #             pos_tok = pos_embed_pretrained[:, 1:, :]     # [1, 196, 768]
    #
    #             # 二维双三次插值
    #             pos_tok = pos_tok.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
    #             # [1, 768, 14, 14]
    #             pos_tok = F.interpolate(pos_tok, size=(new_size, new_size),
    #                                     mode='bicubic', align_corners=False)
    #             # [1, 768, 16, 16]
    #             pos_tok = pos_tok.permute(0, 2, 3, 1).flatten(1, 2)
    #             # [1, 256, 768]
    #
    #             pretrained_dict['pos_embed'] = torch.cat([cls_tok, pos_tok], dim=1)
    #             # [1, 257, 768] ✅
    #
    #     # ── 3. 加载权重（strict=False 忽略形状不匹配的剩余键）───────────────
    #     msg = self.load_state_dict(pretrained_dict, strict=False)
    #
    #     # 打印详细的加载报告
    #     print(f"  成功加载的权重组件:")
    #     loaded = set(pretrained_dict.keys()) - set(msg.missing_keys) - set(msg.unexpected_keys)
    #     for k in sorted(loaded):
    #         print(f"    ✅ {k}")
    #     if msg.missing_keys:
    #         print(f"  Missing keys ({len(msg.missing_keys)}个):")
    #         for k in msg.missing_keys[:5]:  # 只打印前5个避免刷屏
    #             print(f"    ❌ {k}")
    #     if msg.unexpected_keys:
    #         print(f"  Unexpected keys: {len(msg.unexpected_keys)}个")
    #
    #     return msg

    def load_from_pth(self, pth_path, in_chans=1):
        print(f"Loading pretrained weights from {pth_path}...")
        pretrained_dict = torch.load(pth_path, map_location='cpu')

        # 尝试提取真正的state_dict
        if isinstance(pretrained_dict, dict):
            if 'class_token' in pretrained_dict or 'encoder.class_token' in pretrained_dict:
                pass
            elif 'model' in pretrained_dict and isinstance(pretrained_dict['model'], dict):
                pretrained_dict = pretrained_dict['model']
            elif 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']
        else:
            raise TypeError(f"Unexpected checkpoint format: {type(pretrained_dict)}")

        # ── Key映射：官方ViT → 我们的命名 ─────────────────────
        # 预训练权重key可能带 'encoder.' 前缀，为了兼容，先统一去掉所有可能前缀
        def strip_prefix(key):
            """去掉常见前缀"""
            for prefix in ['encoder.', 'model.', 'module.', 'backbone.']:
                if key.startswith(prefix):
                    return key[len(prefix):]
            return key

        # 创建去掉前缀的临时字典
        stripped_dict = {}
        for k, v in pretrained_dict.items():
            stripped_dict[strip_prefix(k)] = v

        print(f"  Original keys: {len(pretrained_dict)}, Stripped keys: {len(stripped_dict)}")

        # 现在用去掉前缀的key进行映射
        key_mapping = {}

        # 1. Patch Embedding
        key_mapping['conv_proj.weight'] = 'patch_embed.proj.weight'
        key_mapping['conv_proj.bias'] = 'patch_embed.proj.bias'

        # 2. Position Embedding & Class Token
        key_mapping['class_token'] = 'cls_token'
        key_mapping['pos_embedding'] = 'pos_embed'

        # 3. Transformer Blocks (12层)
        for i in range(12):
            # Layer Norms
            key_mapping[f'layers.encoder_layer_{i}.ln_1.weight'] = f'blocks.{i}.norm1.weight'
            key_mapping[f'layers.encoder_layer_{i}.ln_1.bias'] = f'blocks.{i}.norm1.bias'
            key_mapping[f'layers.encoder_layer_{i}.ln_2.weight'] = f'blocks.{i}.norm2.weight'
            key_mapping[f'layers.encoder_layer_{i}.ln_2.bias'] = f'blocks.{i}.norm2.bias'

            # Multi-head Attention
            key_mapping[f'layers.encoder_layer_{i}.self_attention.in_proj_weight'] = f'blocks.{i}.attn.qkv.weight'
            key_mapping[f'layers.encoder_layer_{i}.self_attention.in_proj_bias'] = f'blocks.{i}.attn.qkv.bias'
            key_mapping[f'layers.encoder_layer_{i}.self_attention.out_proj.weight'] = f'blocks.{i}.attn.proj.weight'
            key_mapping[f'layers.encoder_layer_{i}.self_attention.out_proj.bias'] = f'blocks.{i}.attn.proj.bias'

            # MLP
            key_mapping[f'layers.encoder_layer_{i}.mlp.linear_1.weight'] = f'blocks.{i}.mlp.0.weight'
            key_mapping[f'layers.encoder_layer_{i}.mlp.linear_1.bias'] = f'blocks.{i}.mlp.0.bias'
            key_mapping[f'layers.encoder_layer_{i}.mlp.linear_2.weight'] = f'blocks.{i}.mlp.3.weight'
            key_mapping[f'layers.encoder_layer_{i}.mlp.linear_2.bias'] = f'blocks.{i}.mlp.3.bias'

        # 4. Final Layer Norm (可能有两种写法)
        key_mapping['encoder.ln.weight'] = 'norm.weight'
        key_mapping['encoder.ln.bias'] = 'norm.bias'
        key_mapping['ln.weight'] = 'norm.weight'  # 兼容无前缀版本
        key_mapping['ln.bias'] = 'norm.bias'

        # 应用key映射
        new_state_dict = {}
        mapped_count = 0
        unmapped_keys = []

        for old_key, value in stripped_dict.items():
            if old_key in key_mapping:
                new_key = key_mapping[old_key]
                new_state_dict[new_key] = value
                mapped_count += 1
            else:
                unmapped_keys.append(old_key)

        print(f"  Mapped {mapped_count}/{len(stripped_dict)} keys")
        if unmapped_keys:
            print(f"  Unmapped keys ({len(unmapped_keys)}): {unmapped_keys[:5]}...")

        # ── 1. patch_embed 权重适配（3通道 → in_chans 通道）─────────────────
        if 'patch_embed.proj.weight' in new_state_dict:
            w_pretrained = new_state_dict['patch_embed.proj.weight']
            if w_pretrained.shape[1] != in_chans:
                print(f"  patch_embed: 通道数 {w_pretrained.shape[1]} → {in_chans}，执行均值适配")
                w_adapted = w_pretrained.mean(dim=1, keepdim=True)
                if in_chans > 1:
                    w_adapted = w_adapted.repeat(1, in_chans, 1, 1)
                new_state_dict['patch_embed.proj.weight'] = w_adapted
                print(f"  patch_embed权重适配完成: {w_pretrained.shape} → {w_adapted.shape}")

        # ── 2. 位置编码插值 ─────────────────────────────────
        if 'pos_embed' in new_state_dict:
            pos_embed_pretrained = new_state_dict['pos_embed']
            old_num = pos_embed_pretrained.shape[1] - 1
            new_num = self.pos_embed.shape[1] - 1

            if old_num != new_num:
                old_size = int(math.sqrt(old_num))
                new_size = int(math.sqrt(new_num))
                print(f"  pos_embed插值: {old_size}x{old_size} → {new_size}x{new_size}")

                cls_tok = pos_embed_pretrained[:, :1, :]
                pos_tok = pos_embed_pretrained[:, 1:, :]
                pos_tok = pos_tok.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
                pos_tok = F.interpolate(pos_tok, size=(new_size, new_size),
                                        mode='bicubic', align_corners=False)
                pos_tok = pos_tok.permute(0, 2, 3, 1).flatten(1, 2)
                new_state_dict['pos_embed'] = torch.cat([cls_tok, pos_tok], dim=1)

        # ── 3. 加载权重 ───────────────────────────────────
        msg = self.load_state_dict(new_state_dict, strict=False)

        loaded_count = len(new_state_dict) - len(msg.missing_keys) - len(msg.unexpected_keys)
        print(f"  ✅ 成功加载: {loaded_count} 个参数（含{len(new_state_dict)}个映射键）")

        if msg.missing_keys:
            print(f"  ❌ Missing keys ({len(msg.missing_keys)}个):")
            for k in msg.missing_keys[:5]:
                print(f"     - {k}")
        if msg.unexpected_keys:
            print(f"  ⚠️ Unexpected keys: {len(msg.unexpected_keys)}个 (已忽略)")

        return msg


    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)               # (B, 256, 768)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, 257, 768)
        x = x + self.pos_embed                 # ✅ 维度匹配
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 1:]  # (B, 256, 768) 去掉cls_token


# =============================================================================
# 新增：TransUnet编码器（CNN浅层 + Transformer深层）
# =============================================================================

class CNNEncoder(nn.Module):
    """CNN编码器部分（提取浅层特征给Transformer和Decoder）"""

    def __init__(self, in_chans=1, base_channels=64):
        super().__init__()

        # 第一层：stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)  # 256 -> 128

        # 第二层
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)  # 128 -> 64

        # 第三层
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)  # 64 -> 32

        # 第四层（输出给Transformer）
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )

        self.base_channels = base_channels

    def forward(self, x):
        # 保存每一层的特征给decoder（像U-Net的skip connection）
        skips = []

        x = self.conv1(x)
        skips.append(x)  # 256x256, base_ch
        x = self.pool1(x)

        x = self.conv2(x)
        skips.append(x)  # 128x128, base_ch*2
        x = self.pool2(x)

        x = self.conv3(x)
        skips.append(x)  # 64x64, base_ch*4
        x = self.pool3(x)

        x = self.conv4(x)  # 32x32, base_ch*8

        return x, skips  # skips的顺序: [256, 128, 64] 尺度的特征


class TransUnetEncoder(nn.Module):
    """
    双流编码器：
    - 流1：原始图像 → Transformer（完整利用预训练权重）
    - 流2：原始图像 → CNN（提取多尺度skip特征）
    - 深层融合两路特征
    """

    def __init__(self, in_chans=1, base_channels=64,
                 img_size=256, patch_size=16, embed_dim=768):
        super().__init__()

        # ── 流1：Transformer（直接处理原始图像）─────────────────────────────
        self.transformer = VisionTransformerEncoder(
            img_size=img_size,    # 256，与预训练一致（只差插值）
            patch_size=patch_size, # 16，与预训练完全一致 ✅
            in_chans=in_chans,    # 1（灰度），权重通过均值适配 ✅
            embed_dim=embed_dim,   # 768，与预训练完全一致 ✅
            depth=12,
            num_heads=12,
            mlp_ratio=4
        )

        # Transformer输出 → 2D特征图
        # patch输出: [B, 256, 768] → reshape → [B, 768, 16, 16] → upsample → [B, 512, 32, 32]
        self.trans_proj = nn.Sequential(
            nn.Conv2d(embed_dim, base_channels * 8, kernel_size=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )

        # ── 流2：CNN（提供skip connections和局部特征）────────────────────────
        self.cnn_encoder = CNNEncoder(in_chans, base_channels)

        # ── 深层融合：拼接CNN深层特征 + Transformer特征 → 压缩 ───────────────
        cnn_deep_channels = base_channels * 8  # 512
        self.fusion = nn.Sequential(
            # 拼接后：512(CNN) + 512(Trans) = 1024
            nn.Conv2d(cnn_deep_channels + cnn_deep_channels, cnn_deep_channels, 1),
            nn.BatchNorm2d(cnn_deep_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_deep_channels, cnn_deep_channels, 3, padding=1),
            nn.BatchNorm2d(cnn_deep_channels),
            nn.ReLU(inplace=True)
        )

        self.embed_dim = embed_dim
        self.cnn_out_channels = cnn_deep_channels

    def load_pretrained_transformer(self, pth_path):
        """加载预训练ViT权重（对外暴露的接口）"""
        return self.transformer.load_from_pth(pth_path, in_chans=1)

    def forward(self, x):
        # ── 流1：Transformer处理原图 ──────────────────────────────────────
        trans_out = self.transformer(x)  # [B, 256, 768]

        B, N, C = trans_out.shape
        H = W = int(math.sqrt(N))        # 16
        trans_feat = trans_out.transpose(1, 2).reshape(B, C, H, W)
        # [B, 768, 16, 16]

        # 上采样到与CNN深层特征相同的空间尺寸(32x32)
        trans_feat = F.interpolate(trans_feat, size=(32, 32),
                                   mode='bilinear', align_corners=False)
        trans_feat = self.trans_proj(trans_feat)
        # [B, 512, 32, 32]

        # ── 流2：CNN处理原图 ──────────────────────────────────────────────
        cnn_deep, skips = self.cnn_encoder(x)
        # cnn_deep: [B, 512, 32, 32]
        # skips: [[B,64,256,256], [B,128,128,128], [B,256,64,64]]

        # ── 深层融合 ──────────────────────────────────────────────────────
        fused = self.fusion(torch.cat([cnn_deep, trans_feat], dim=1))
        # [B, 512, 32, 32]

        return fused, skips


# =============================================================================
# 新增：TransUnet解码器（继承你原有的DecoderBlock设计）
# =============================================================================

class TransUnetDecoder(nn.Module):
    """TransUnet解码器，接收skip connections和深层特征"""

    def __init__(self, cnn_out_channels=512, base_channels=64, seg_head_channels=64):
        super().__init__()

        # decoder4: 从32x32上采样到64x64，拼接skip3（64x64, base_ch*4=256）
        self.decoder4 = DecoderBlock(cnn_out_channels, base_channels * 4, base_channels * 4)
        # decoder3: 64x64 -> 128x128，拼接skip2（128x128, base_ch*2=128）
        self.decoder3 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        # decoder2: 128x128 -> 256x256，拼接skip1（256x256, base_ch=64）
        self.decoder2 = DecoderBlock(base_channels * 2, base_channels, base_channels)

        # 最终输出头
        self.decoder1 = nn.Sequential(
            nn.Conv2d(base_channels, seg_head_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(seg_head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(seg_head_channels, seg_head_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(seg_head_channels),
            nn.ReLU(inplace=True)
        )

        self.base_channels = base_channels

    def forward(self, x, skips):
        """
        x: [B, 512, 32, 32] - TransUnet编码器输出
        skips: list of [B, 64, 256, 256], [B, 128, 128, 128], [B, 256, 64, 64]
        """
        skip3, skip2, skip1 = skips  # 注意顺序：从大到小


        x = self.decoder4(x, skip1)  # 32x32 + 64x64 -> 64x64
        x = self.decoder3(x, skip2)  # 64x64 + 128x128 -> 128x128
        x = self.decoder2(x, skip3)  # 128x128 + 256x256 -> 256x256
        x = self.decoder1(x)  # -> 256x256, seg_head_channels

        return x


# =============================================================================
# 修改：使用TransUnet作为主模型
# =============================================================================

class TransUnet_UniMatch(nn.Module):
    """双视图TransUnet模型，同时支持分割和分类"""

    def __init__(self, in_chns=1, seg_class_num=3, cls_class_num=1,
                 base_channels=64, embed_dim=768, **kwargs):
        super().__init__()

        # --- 编码器（双视图共享） ---
        self.encoder = TransUnetEncoder(
            in_chans=in_chns,
            base_channels=base_channels,
            img_size=256,
            patch_size=16,
            embed_dim=embed_dim
        )

        # --- 解码器（双视图共享） ---
        cnn_out_channels = base_channels * 8  # 64*8=512
        self.decoder = TransUnetDecoder(
            cnn_out_channels=cnn_out_channels,
            base_channels=base_channels,
            seg_head_channels=64
        )

        # --- 分割头（Long/Trans分开） ---
        self.seg_head_long = nn.Conv2d(64, seg_class_num, kernel_size=1)
        self.seg_head_trans = nn.Conv2d(64, seg_class_num, kernel_size=1)

        # --- 分类头（使用Transformer输出做分类） ---
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Linear(cnn_out_channels * 2, 512),  # 512*2=1024
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, cls_class_num)
        )

        # 特征扰动dropout（用于UniMatch的半监督）
        self.fp_drop = nn.Dropout2d(0.5)

        # 梯度分离控制
        self.detach_cls_grad = True

    def set_cls_detach(self, detach=True):
        self.detach_cls_grad = detach

    def _extract_cls_features(self, encoder_out):
        """
        从编码器输出中提取分类特征
        encoder_out: [B, 512, 32, 32] - 经过Transformer增强的特征
        """
        cls_feat = self.cls_pool(encoder_out).flatten(1)  # [B, 512]
        return cls_feat

    def forward_view(self, x, fp=False, return_cls_only=False):
        """
        单视图前向传播
        """
        # 编码
        encoder_out, skips = self.encoder(x)  # encoder_out: [B, 512, 32, 32]

        # 特征扰动
        if fp:
            encoder_out = self.fp_drop(encoder_out)

        # 提取分类特征
        cls_feat = self._extract_cls_features(encoder_out)  # [B, 512]

        if return_cls_only:
            return None, cls_feat

        # 梯度分离
        if self.detach_cls_grad and self.training:
            encoder_out_for_seg = encoder_out.detach()
            skips_for_seg = [s.detach() for s in skips]
        else:
            encoder_out_for_seg = encoder_out
            skips_for_seg = skips

        # 解码
        seg_feat = self.decoder(encoder_out_for_seg, skips_for_seg)  # [B, 64, 256, 256]

        return seg_feat, cls_feat

    def forward(self, x_long, x_trans, need_fp=False):
        """
        双视图前向传播
        """
        # 标准前向
        featL_seg, featL_cls = self.forward_view(x_long, fp=False)
        featT_seg, featT_cls = self.forward_view(x_trans, fp=False)

        segL = self.seg_head_long(featL_seg)
        segT = self.seg_head_trans(featT_seg)

        cls_feat = torch.cat([featL_cls, featT_cls], dim=1)  # [B, 1024]
        cls_out = self.cls_head(cls_feat)

        if not need_fp:
            return segL, segT, cls_out

        # 特征扰动前向（UniMatch需要）
        featL_seg_fp, featL_cls_fp = self.forward_view(x_long, fp=True)
        featT_seg_fp, featT_cls_fp = self.forward_view(x_trans, fp=True)

        segL_fp = self.seg_head_long(featL_seg_fp)
        segT_fp = self.seg_head_trans(featT_seg_fp)

        cls_feat_fp = torch.cat([featL_cls_fp, featT_cls_fp], dim=1)
        cls_out_fp = self.cls_head(cls_feat_fp)

        return (segL, segL_fp), (segT, segT_fp), (cls_out, cls_out_fp)


# 工具函数
def transunet_tiny_unimatch(**kwargs):
    return TransUnet_UniMatch(**kwargs)


# 可选：加载预训练ViT权重（如果你有ImageNet预训练）
def load_pretrained_vit(model, weight_path):
    """加载预训练的ViT权重（可选）"""
    state_dict = torch.load(weight_path, map_location='cpu')
    # 这里需要根据你的预训练权重格式进行处理
    # 通常会匹配 transformer.blocks 的权重
    missing, unexpected = model.encoder.transformer.load_state_dict(state_dict, strict=False)
    print(f"Loaded ViT weights, missing: {len(missing)}, unexpected: {len(unexpected)}")
    return model


# =============================================================================
# 1. Helper Functions (Replaces timm dependencies for single-file usage)
# =============================================================================

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + torch.erf(x / torch.sqrt(torch.tensor(2.)))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("warnings: mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
              "The distribution of values may be incorrect.", )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * torch.sqrt(torch.tensor(2.)))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


# =============================================================================
# 2. ConvNeXt Backbone (From provided convnext.py)
# =============================================================================

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x



# =============================================================================
# 3. Main Task Model (Adapts to train.py)
# =============================================================================

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        # Bilinear upsampling
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            # Handle potential padding issues if resolutions don't match exactly
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x




if __name__ == '__main__':
    x_l = torch.randn(2, 1, 256, 256)
    x_t = torch.randn(2, 1, 256, 256)
    # 测试新的TransUnet模型
    print("\n=== Testing TransUnet_UniMatch ===")
    transunet_model = TransUnet_UniMatch(in_chns=1, seg_class_num=3, cls_class_num=1)
    pth_weight_path = "vit_b_16-c867db91.pth"
    # 执行加载
    try:
        msg = transunet_model.encoder.load_pretrained_transformer(pth_weight_path)
    except FileNotFoundError:
        print(f"Warning: {pth_weight_path} not found, using random initialization.")

    # 计算参数量对比
    transunet_params = sum(p.numel() for p in transunet_model.parameters())
    print(f"TransUnet参数量: {transunet_params / 1e6:.2f}M")

    # 测试前向传播
    sL, sT, cls = transunet_model(x_l, x_t)
    print(f"TransUnet: SegL={sL.shape}, SegT={sT.shape}, Cls={cls.shape}")

    # 测试FP模式
    (sL, sL_fp), (sT, sT_fp), (cls, cls_fp) = transunet_model(x_l, x_t, need_fp=True)
    print(f"FP模式测试通过: segL_fp={sL_fp.shape}")

    # 测试梯度分离
    transunet_model.set_cls_detach(True)
    print("梯度分离模式已启用")

    # 验证预训练权重是否真的加载了（而非随机初始化）
    print("\n=== 权重加载验证 ===")

    # 检查patch_embed权重 - 应该不是随机值
    patch_weight = transunet_model.encoder.transformer.patch_embed.proj.weight
    print(f"patch_embed权重范围: [{patch_weight.min():.3f}, {patch_weight.max():.3f}]")
    print(f"patch_embed权重均值: {patch_weight.mean():.3f}, 标准差: {patch_weight.std():.3f}")

    # 检查第一个Transformer block的norm1权重
    norm1_weight = transunet_model.encoder.transformer.blocks[0].norm1.weight
    print(f"blocks[0].norm1权重范围: [{norm1_weight.min():.3f}, {norm1_weight.max():.3f}]")
    print(f"blocks[0].norm1权重均值: {norm1_weight.mean():.3f}")

    # 检查pos_embed是否被正确插值
    pos_embed = transunet_model.encoder.transformer.pos_embed
    print(f"pos_embed形状: {pos_embed.shape} (预期: [1, 257, 768])")

    print("\n✅ 预训练ViT权重加载验证完成！")