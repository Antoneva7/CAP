"""
CascadedUniMatch: 临床认知驱动的颈动脉斑块分析模型
- Stage 1 (解剖定位):   轻量UNet → 血管分割 + 提示框生成
- Stage 2 (异常提议):   ROI裁剪 + ConvNeXt → 斑块分割
- Stage 3 (微观分析):   双视角注意力融合
- Stage 4 (证据融合):   多尺度特征 → 脆弱性分类
在UniMatch半监督框架下运行

v2:Stage1只做血管分割，Stage2在血管分割的ROI中只做斑块分割
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

# ─────────────────────────────────────────────────────────
# 1. 工具函数
# ─────────────────────────────────────────────────────────

def get_vessel_bbox(
    vessel_mask: torch.Tensor,
    padding: int = 16,
    min_size: int = 64
) -> torch.Tensor:
    """
    从血管分割mask中提取ROI提示框。

    Args:
        vessel_mask : [B, H, W] long tensor (0=bg, 1=plaque预留, 2=vessel)
        padding     : 边界扩张像素数
        min_size    : bbox最小尺寸（防止退化）

    Returns:
        boxes : [B, 4] int tensor → (y1, x1, y2, x2) in pixel coords
    """
    B, H, W = vessel_mask.shape
    # 取血管区域（class=2）
    vessel_bin = (vessel_mask == 1).float()  # [B,H,W]
    boxes = torch.zeros(B, 4, dtype=torch.long, device=vessel_mask.device)

    for b in range(B):
        fg = vessel_bin[b]  # [H,W]
        if fg.sum() < 1:
            # 无血管预测时，使用全图
            boxes[b] = torch.tensor([0, 0, H, W], device=vessel_mask.device)
            continue

        # 行/列投影
        rows = fg.any(dim=1).nonzero(as_tuple=False).view(-1)  # [num_rows]
        cols = fg.any(dim=0).nonzero(as_tuple=False).view(-1)  # [num_cols]

        y1 = max(0,     rows.min().item() - padding)
        y2 = min(H - 1, rows.max().item() + padding)
        x1 = max(0,     cols.min().item() - padding)
        x2 = min(W - 1, cols.max().item() + padding)

        # 保证最小尺寸
        if (y2 - y1) < min_size:
            cy = (y1 + y2) // 2
            y1 = max(0, cy - min_size // 2)
            y2 = min(H - 1, cy + min_size // 2)
        if (x2 - x1) < min_size:
            cx = (x1 + x2) // 2
            x1 = max(0, cx - min_size // 2)
            x2 = min(W - 1, cx + min_size // 2)

        boxes[b] = torch.tensor([y1, x1, y2, x2], device=vessel_mask.device)

    return boxes

def roi_align_crop(
    feat: torch.Tensor,
    boxes: torch.Tensor,
    output_size: int = 128
) -> torch.Tensor:
    """
    按bbox从特征图裁剪ROI并resize到固定尺寸。
    支持图像级 [B,C,H,W] 或像素级操作。

    Args:
        feat        : [B, C, H, W]
        boxes       : [B, 4] → (y1,x1,y2,x2) in pixel coords of H×W space
        output_size : 输出空间尺寸

    Returns:
        roi_feats : [B, C, output_size, output_size]
    """
    B, C, H, W = feat.shape
    roi_list = []
    for b in range(B):
        y1, x1, y2, x2 = boxes[b].tolist()
        y1, x1 = int(y1), int(x1)
        y2, x2 = int(y2), int(x2)
        # 防止越界
        y1 = max(0, y1); x1 = max(0, x1)
        y2 = min(H, y2); x2 = min(W, x2)
        if y2 <= y1: y2 = y1 + 1
        if x2 <= x1: x2 = x1 + 1

        crop = feat[b:b+1, :, y1:y2, x1:x2]          # [1,C,h,w]
        crop = F.interpolate(
            crop, (output_size, output_size),
            mode='bilinear', align_corners=False
        )                                              # [1,C,os,os]
        roi_list.append(crop)
    return torch.cat(roi_list, dim=0)                 # [B,C,os,os]

def paste_back(
    roi_pred: torch.Tensor,
    boxes: torch.Tensor,
    full_size: Tuple[int, int]
) -> torch.Tensor:
    """
    将ROI预测结果粘贴回原始图像尺寸。

    Args:
        roi_pred  : [B, C, roi_h, roi_w]  softmax logits
        boxes     : [B, 4]  (y1,x1,y2,x2)
        full_size : (H, W)

    Returns:
        full_pred : [B, C, H, W]  仅ROI区域有值，其余填背景logit
    """
    B, C, _, _ = roi_pred.shape
    H, W = full_size
    # 背景初始化：class0=大正值，其余=0（等价于预测背景）
    full_pred = torch.zeros(B, C, H, W, device=roi_pred.device)
    full_pred[:, 0, :, :] = 5.0  # 背景logit偏置

    for b in range(B):
        y1, x1, y2, x2 = boxes[b].tolist()
        y1, x1 = int(y1), int(x1)
        y2, x2 = int(y2), int(x2)
        y1 = max(0, y1); x1 = max(0, x1)
        y2 = min(H, y2); x2 = min(W, x2)
        if y2 <= y1: y2 = y1 + 1
        if x2 <= x1: x2 = x1 + 1

        roi_h, roi_w = y2 - y1, x2 - x1
        resized = F.interpolate(
            roi_pred[b:b+1],
            (roi_h, roi_w),
            mode='bilinear', align_corners=False
        )  # [1,C,roi_h,roi_w]
        full_pred[b, :, y1:y2, x1:x2] = resized[0]

    return full_pred

# ─────────────────────────────────────────────────────────
# 2. 血管分割UNet (Stage 1: 解剖定位)
# ─────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """UNet基础双卷积块"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class VesselUNet(nn.Module):
    """
    轻量UNet：专注血管分割（解剖定位阶段）
    输入:  [B, 1, 256, 256]
    输出:  vessel_logits [B, seg_cls, 256, 256]
            bottleneck_feat [B, 512, 16, 16]  (供分类使用)
    """
    def __init__(self, in_chns: int = 1, seg_cls: int = 2, base_ch: int = 32):
        super().__init__()
        bc = base_ch   # 32

        # Encoder
        self.enc1 = DoubleConv(in_chns, bc)        # 32,  256
        self.enc2 = DoubleConv(bc,      bc * 2)    # 64,  128
        self.enc3 = DoubleConv(bc * 2,  bc * 4)    # 128,  64
        self.enc4 = DoubleConv(bc * 4,  bc * 8)    # 256,  32

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(bc * 8, bc * 16)   # 512, 16

        # Decoder
        self.up4   = nn.ConvTranspose2d(bc * 16, bc * 8, 2, stride=2)
        self.dec4  = DoubleConv(bc * 16, bc * 8)

        self.up3   = nn.ConvTranspose2d(bc * 8, bc * 4, 2, stride=2)
        self.dec3  = DoubleConv(bc * 8, bc * 4)

        self.up2   = nn.ConvTranspose2d(bc * 4, bc * 2, 2, stride=2)
        self.dec2  = DoubleConv(bc * 4, bc * 2)

        self.up1   = nn.ConvTranspose2d(bc * 2, bc, 2, stride=2)
        self.dec1  = DoubleConv(bc * 2, bc)

        # Segmentation head
        self.seg_head = nn.Conv2d(bc, seg_cls, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits       : [B, seg_cls, H, W]
            bottleneck   : [B, 512,    H/16, W/16]
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bn = self.bottleneck(self.pool(e4))          # [B,512,16,16]

        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        logits = self.seg_head(d1)                   # [B, seg_cls, H, W]
        return logits, bn                            # bn用于分类辅助

# ─────────────────────────────────────────────────────────
# 3. 双视角注意力融合模块 (Stage 3: 微观分析)
# ─────────────────────────────────────────────────────────

class CrossViewAttention(nn.Module):
    """
    双视角交叉注意力：
    纵轴特征 Query 横轴特征 Key/Value → 增强斑块感知

    基于简化的 MHSA，减少计算量。
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # Q来自当前视角，K/V来自对方视角
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out    = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.norm      = nn.LayerNorm(dim)

    def forward(self, x_self: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_self  : [B, C, H, W] 当前视角特征
            x_other : [B, C, H, W] 对方视角特征
        Returns:
            out     : [B, C, H, W] 增强后的特征
        """
        B, C, H, W = x_self.shape
        N = H * W

        # Flatten spatial → token序列
        xs = x_self.flatten(2).permute(0, 2, 1)    # [B, N, C]
        xo = x_other.flatten(2).permute(0, 2, 1)   # [B, N, C]

        Q = self.q_proj(xs).reshape(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        K = self.k_proj(xo).reshape(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        V = self.v_proj(xo).reshape(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)

        attn = (Q @ K.transpose(-2, -1)) * self.scale   # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out(out)

        # 残差 + LN
        out = self.norm(out + xs)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)  # [B,C,H,W]
        return out

class DualViewFusion(nn.Module):
    """
    双视角特征融合（Stage 3: 微观分析 + Stage 4: 证据融合）
    - 双向交叉注意力
    - 通道注意力门控
    - 特征融合输出
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        # 双向交叉注意力
        self.cross_l2t = CrossViewAttention(dim, num_heads)   # 纵→横
        self.cross_t2l = CrossViewAttention(dim, num_heads)   # 横→纵

        # 通道注意力门控（SE-style）
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim * 2, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim * 2),
            nn.Sigmoid()
        )
        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        feat_long: torch.Tensor,
        feat_trans: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            feat_long_fused  : [B, dim, H, W]
            feat_trans_fused : [B, dim, H, W]
        """
        # 双向交叉注意力增强
        enh_long  = self.cross_l2t(feat_long, feat_trans)   # 纵轴感知横轴
        enh_trans = self.cross_t2l(feat_trans, feat_long)   # 横轴感知纵轴

        # 拼接后通道门控
        cat_feat = torch.cat([enh_long, enh_trans], dim=1)  # [B, 2C, H, W]
        gates = self.gate(cat_feat).unsqueeze(-1).unsqueeze(-1)  # [B, 2C, 1, 1]
        gated = cat_feat * gates

        # 分离并更新各视角
        gl, gt = gated.chunk(2, dim=1)  # 各 [B, C, H, W]
        feat_long_out  = enh_long  + gl
        feat_trans_out = enh_trans + gt

        return feat_long_out, feat_trans_out

# ─────────────────────────────────────────────────────────
# 4. ConvNeXt 斑块分析头 (Stage 2+3: 异常提议+微观分析)
# ─────────────────────────────────────────────────────────

# 复用 convnext.py 中的组件
try:
    from model.convnext import ConvNeXt, LayerNorm, DecoderBlock
except ImportError:
    from convnext import ConvNeXt, LayerNorm, DecoderBlock

class PlaqueConvNeXtDecoder(nn.Module):
    """
    ConvNeXt编码器 + 轻量解码器，专注ROI内斑块分割。
    输入: ROI裁剪后的图像 [B, 1, roi_size, roi_size]
    输出:
        plaque_logits [B, seg_cls, roi_size, roi_size]
        deep_features [B, 768, roi_size//32, roi_size//32]
    """
    def __init__(
        self,
        in_chns: int = 1,
        seg_cls: int = 2,
        encoder_pth: Optional[str] = None,
        roi_size: int = 128
    ):
        super().__init__()

        # ── Encoder ──
        self.encoder = ConvNeXt(
            in_chans=in_chns,
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768],
            drop_path_rate=0.1,  # ROI阶段稍小dropout
        )
        if encoder_pth is not None:
            self.encoder.load_pretrained_weights(encoder_pth, in_chans=in_chns)

        dims = [96, 192, 384, 768]

        # ── Decoder ──
        self.dec4 = DecoderBlock(dims[3], dims[2], dims[2])   # 768→384
        self.dec3 = DecoderBlock(dims[2], dims[1], dims[1])   # 384→192
        self.dec2 = DecoderBlock(dims[1], dims[0], dims[0])   # 192→96

        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(dims[0], 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 斑块分割头
        self.plaque_head = nn.Conv2d(64, seg_cls, kernel_size=1)

        # Feature Perturbation (UniMatch)
        self.fp_drop = nn.Dropout2d(0.5)

    def _decode(self, feats):
        f0, f1, f2, f3 = feats
        x = self.dec4(f3, f2)
        x = self.dec3(x, f1)
        x = self.dec2(x, f0)
        x = self.dec1(x)
        return x, f3   # seg_feat, deep_feat

    def forward(
        self,
        roi_img: torch.Tensor,
        fp: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            plaque_logits : [B, seg_cls, roi_h, roi_w]
            deep_feat     : [B, 768, roi_h//32, roi_w//32]
        """
        feats = self.encoder(roi_img)
        if fp:
            feats = [self.fp_drop(f) for f in feats]

        seg_feat, deep_feat = self._decode(feats)
        logits = self.plaque_head(seg_feat)

        return logits, deep_feat

# ─────────────────────────────────────────────────────────
# 5. 脆弱性分类头 (Stage 4: 证据融合)
# ─────────────────────────────────────────────────────────

class VulnerabilityClassifier(nn.Module):
    """
    证据融合分类器：
    聚合来自4个来源的特征：
      - 纵轴深层CNN特征 (ConvNeXt)
      - 横轴深层CNN特征 (ConvNeXt)
      - 纵轴UNet瓶颈特征
      - 横轴UNet瓶颈特征

    输出: 脆弱性 logit [B, cls_num]
    """
    def __init__(
        self,
        convnext_dim: int = 768,
        unet_dim: int = 512,
        cls_num: int = 1,
        dropout: float = 0.5
    ):
        super().__init__()
        total_dim = (convnext_dim + unet_dim) * 2  # 两个视角各两种特征

        self.pool = nn.AdaptiveAvgPool2d(1)

        # 分层融合：先各视角内部融合，再跨视角融合
        view_dim = convnext_dim + unet_dim  # 1280

        # 视角内融合 MLP
        self.view_mlp_long = nn.Sequential(
            nn.Linear(view_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.view_mlp_trans = nn.Sequential(
            nn.Linear(view_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # 跨视角融合 + 分类
        self.cls_head = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, cls_num)
        )

    def forward(
        self,
        deep_long: torch.Tensor,    # [B, 768, h, w]
        deep_trans: torch.Tensor,   # [B, 768, h, w]
        unet_long: torch.Tensor,    # [B, 512, h2, w2]
        unet_trans: torch.Tensor,   # [B, 512, h2, w2]
    ) -> torch.Tensor:
        # 全局平均池化
        dl = self.pool(deep_long).flatten(1)    # [B, 768]
        dt = self.pool(deep_trans).flatten(1)   # [B, 768]
        ul = self.pool(unet_long).flatten(1)    # [B, 512]
        ut = self.pool(unet_trans).flatten(1)   # [B, 512]

        # 视角内融合
        feat_long  = self.view_mlp_long(torch.cat([dl, ul], dim=1))   # [B, 512]
        feat_trans = self.view_mlp_trans(torch.cat([dt, ut], dim=1))  # [B, 512]

        # 跨视角融合 + 分类
        fused  = torch.cat([feat_long, feat_trans], dim=1)             # [B, 1024]
        logits = self.cls_head(fused)                                   # [B, cls_num]
        return logits

# ─────────────────────────────────────────────────────────
# 6. 主模型：CascadedUniMatch
# ─────────────────────────────────────────────────────────

class CascadedUniMatch(nn.Module):
    """
    级联双阶段模型（兼容原 train.py 的 UniMatch 框架）

    前向接口与原 ConvNeXt_UniMatch 完全一致:
        forward(x_long, x_trans, need_fp=False)

    Stage 1: VesselUNet  → 血管分割 + 提示框
    Stage 2: ROI裁剪     → 双视角 ROI 图像
    Stage 3: PlaqueConvNeXt + DualViewFusion → 斑块分割
    Stage 4: VulnerabilityClassifier         → 脆弱性分类
    """
    def __init__(
        self,
        in_chns:       int   = 1,
        seg_class_num: int   = 3,
        cls_class_num: int   = 1,
        roi_size:      int   = 128,
        encoder_pth:   Optional[str] = None,
        unet_base_ch:  int   = 32,
        attn_heads:    int   = 4,      # 注意力头数（ROI特征图较小时用4）
        **kwargs
    ):
        super().__init__()
        self.roi_size      = roi_size
        self.seg_class_num = seg_class_num
        self.vessel_cls = 2
        self.plaque_cls = 2

        # ── Stage 1: 血管UNet (双视角共享权重) ──
        self.vessel_unet = VesselUNet(
            in_chns=in_chns,
            seg_cls=2,
            base_ch=unet_base_ch
        )

        # ── Stage 2+3: 斑块ConvNeXt (双视角共享编码器，独立解码头) ──
        self.plaque_encoder_long = PlaqueConvNeXtDecoder(
            in_chns=in_chns,
            seg_cls=2,
            encoder_pth=encoder_pth,
            roi_size=roi_size
        )
        # 横轴视角共享同一个ConvNeXt编码器权重
        self.plaque_encoder_trans = PlaqueConvNeXtDecoder(
            in_chns=in_chns,
            seg_cls=2,
            encoder_pth=None,           # 不重复加载
            roi_size=roi_size
        )
        # 共享编码器权重（仅训练一份编码器）
        self.plaque_encoder_trans.encoder = self.plaque_encoder_long.encoder

        # ── Stage 3: 双视角注意力融合 ──
        # ConvNeXt最深层dim=768，但ROI=128时空间尺寸=128/32=4
        self.dual_view_fusion = DualViewFusion(dim=768, num_heads=attn_heads)

        # ── Stage 4: 脆弱性分类 ──
        self.vulnerability_cls = VulnerabilityClassifier(
            convnext_dim=768,
            unet_dim=unet_base_ch * 16,  # 512
            cls_num=cls_class_num,
            dropout=0.5
        )

        # ── 梯度解耦控制 ──
        self.detach_stage2_grad = True   # 早期阶段2梯度不更新Stage1

    def set_cls_detach(self, detach: bool = True):
        """兼容 train.py 的梯度控制接口"""
        self.detach_stage2_grad = detach

    # ──────────────────────────────────────────────────────
    # 内部流程：单视角Stage1前向
    # ──────────────────────────────────────────────────────
    def _stage1_forward(self, x: torch.Tensor):
        """
        Args:
            x : [B, 1, H, W]
        Returns:
            vessel_logits : [B, seg_cls, H, W]
            vessel_pred   : [B, H, W] (argmax)
            boxes         : [B, 4]
            unet_bn       : [B, 512, H/16, W/16]
        """
        vessel_logits, unet_bn = self.vessel_unet(x)   # 血管分割
        vessel_pred = vessel_logits.argmax(dim=1)       # [B,H,W]
        boxes = get_vessel_bbox(vessel_pred, padding=16, min_size=64)
        return vessel_logits, vessel_pred, boxes, unet_bn

    # ──────────────────────────────────────────────────────
    # 内部流程：Stage2+3 斑块分割 + 双视角融合
    # ──────────────────────────────────────────────────────
    def _stage2_forward(
        self,
        x_long:   torch.Tensor,
        x_trans:  torch.Tensor,
        boxes_l:  torch.Tensor,
        boxes_t:  torch.Tensor,
        fp:       bool = False
    ):
        """
        Args:
            x_long/x_trans : [B,1,H,W]
            boxes_l/boxes_t : [B,4]
            fp : Feature Perturbation (UniMatch)

        Returns:
            plaque_logits_long  : [B, seg_cls, H, W]  (paste-back到原图尺寸)
            plaque_logits_trans : [B, seg_cls, H, W]
            deep_long           : [B, 768, 4, 4]      (ROI深层特征，用于分类)
            deep_trans          : [B, 768, 4, 4]
        """
        H, W = x_long.shape[-2:]

        # ROI裁剪
        roi_long  = roi_align_crop(x_long,  boxes_l, self.roi_size)   # [B,1,roi,roi]
        roi_trans = roi_align_crop(x_trans, boxes_t, self.roi_size)   # [B,1,roi,roi]

        # Stage 2: ConvNeXt斑块分割
        logits_l_roi, deep_l = self.plaque_encoder_long(roi_long,   fp=fp)
        logits_t_roi, deep_t = self.plaque_encoder_trans(roi_trans,  fp=fp)

        # Stage 3: 双视角注意力融合（在深层特征空间）
        deep_l_fused, deep_t_fused = self.dual_view_fusion(deep_l, deep_t)

        # 融合后的特征重新过分割头（微调）
        # 注意: 由于deep层空间很小(4x4)，直接用pool后MLP做分类
        # 分割logits已在ROI空间计算完毕，融合主要用于分类

        # Paste-back到原图尺寸
        plaque_logits_long  = paste_back(logits_l_roi, boxes_l, (H, W))
        plaque_logits_trans = paste_back(logits_t_roi, boxes_t, (H, W))

        return plaque_logits_long, plaque_logits_trans, deep_l_fused, deep_t_fused

    # ──────────────────────────────────────────────────────
    # 完整单次前向（一组视角）
    # ──────────────────────────────────────────────────────
    def _full_forward(
        self,
        x_long:  torch.Tensor,
        x_trans: torch.Tensor,
        fp:      bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        完整的四阶段前向传播。
        Returns dict with keys:
            vessel_long, vessel_trans   : Stage1 logits  [B, seg_cls, H, W]
            plaque_long, plaque_trans   : Stage2+3 logits [B, seg_cls, H, W]
            cls_out                     : Stage4 logit   [B, cls_num]
            boxes_long, boxes_trans     : ROI boxes      [B, 4]
        """
        # ── Stage 1 ──
        vessel_l, pred_l, boxes_l, unet_bn_l = self._stage1_forward(x_long)
        vessel_t, pred_t, boxes_t, unet_bn_t = self._stage1_forward(x_trans)

        # 梯度解耦：早期阶段2不更新阶段1参数
        if self.detach_stage2_grad and self.training:
            x_long_s2  = x_long.detach()
            x_trans_s2 = x_trans.detach()
            boxes_l_s2 = boxes_l.detach()
            boxes_t_s2 = boxes_t.detach()
            unet_bn_l_cls = unet_bn_l.detach()
            unet_bn_t_cls = unet_bn_t.detach()
        else:
            x_long_s2  = x_long
            x_trans_s2 = x_trans
            boxes_l_s2 = boxes_l
            boxes_t_s2 = boxes_t
            unet_bn_l_cls = unet_bn_l
            unet_bn_t_cls = unet_bn_t

        # ── Stage 2+3 ──
        plaque_l, plaque_t, deep_l, deep_t = self._stage2_forward(
            x_long_s2, x_trans_s2,
            boxes_l_s2, boxes_t_s2,
            fp=fp
        )

        # ── Stage 4 ──
        cls_out = self.vulnerability_cls(
            deep_long  = deep_l,
            deep_trans = deep_t,
            unet_long  = unet_bn_l_cls,
            unet_trans = unet_bn_t_cls,
        )

        return {
            "vessel_long":  vessel_l,
            "vessel_trans": vessel_t,
            "plaque_long":  plaque_l,
            "plaque_trans": plaque_t,
            "cls_out":      cls_out,
            "boxes_long":   boxes_l,
            "boxes_trans":  boxes_t,
        }

    # ──────────────────────────────────────────────────────
    # 公开接口：兼容 train.py
    # ──────────────────────────────────────────────────────
    def forward(
        self,
        x_long:   torch.Tensor,
        x_trans:  torch.Tensor,
        need_fp:  bool = False,
    ):
        """
        兼容原 train.py 接口。

        Returns (need_fp=False):
            seg_long  : [B, seg_cls, H, W]  ← 融合血管+斑块的最终分割
            seg_trans : [B, seg_cls, H, W]
            cls_out   : [B, cls_num]

        Returns (need_fp=True):
            (seg_long, seg_long_fp), (seg_trans, seg_trans_fp), (cls_out, cls_out_fp)
        """
        # ── 标准前向 ──
        out = self._full_forward(x_long, x_trans, fp=False)

        # 融合两阶段分割结果（Stage1血管 + Stage2斑块联合输出）
        seg_long  = self._merge_seg(out["vessel_long"],  out["plaque_long"])
        seg_trans = self._merge_seg(out["vessel_trans"], out["plaque_trans"])
        cls_out   = out["cls_out"]

        if not need_fp:
            return seg_long, seg_trans, cls_out

        # ── Feature Perturbation 前向（UniMatch） ──
        out_fp = self._full_forward(x_long, x_trans, fp=True)
        seg_long_fp  = self._merge_seg(out_fp["vessel_long"],  out_fp["plaque_long"])
        seg_trans_fp = self._merge_seg(out_fp["vessel_trans"], out_fp["plaque_trans"])
        cls_out_fp   = out_fp["cls_out"]

        return (
            (seg_long,  seg_long_fp),
            (seg_trans, seg_trans_fp),
            (cls_out,   cls_out_fp),
        )

    def _merge_seg(
        self,
        vessel_logits: torch.Tensor,
        plaque_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        融合Stage1（血管：2类）和Stage2（斑块：2类）→ 3类输出

        最终类别映射：
        - class 0: 背景
        - class 1: 斑块
        - class 2: 血管

        策略：
        - 背景：两阶段背景logit的几何平均
        - 斑块：直接使用Stage2的斑块logit
        - 血管：Stage1的血管logit - Stage2的斑块logit（去除重叠）
        """
        B, _, H, W = vessel_logits.shape

        # 提取各自的logit
        vessel_bg = vessel_logits[:, 0:1]  # [B, 1, H, W]
        vessel_fg = vessel_logits[:, 1:2]  # [B, 1, H, W] 血管

        plaque_bg = plaque_logits[:, 0:1]  # [B, 1, H, W]
        plaque_fg = plaque_logits[:, 1:2]  # [B, 1, H, W] 斑块

        # 1. 背景：两者背景logit取几何平均（都认为是背景才是背景）
        bg = (vessel_bg + plaque_bg) / 2.0

        # 2. 斑块：直接使用Stage2输出（Stage2是斑块专家）
        plq = plaque_fg

        # 3. 血管：Stage1血管 - Stage2斑块（避免双重计数）
        #    斑块在血管内部，需要从血管区域中扣除斑块
        vsl = vessel_fg - plaque_fg * 0.5  # 保守扣除，避免过度抑制

        # 最终输出：[B, 3, H, W]  (背景, 斑块, 血管)
        return torch.cat([bg, plq, vsl], dim=1)
