"""
CascadedUniMatch: 临床认知驱动的颈动脉斑块分析模型
- Stage 1 (解剖定位):   轻量UNet → 血管分割 + 提示框生成
- Stage 2 (异常提议):   注意力引导 + ConvNeXt → 斑块分割
- Stage 3 (微观分析):   双视角注意力融合
- Stage 4 (证据融合):   多尺度特征 → 脆弱性分类
在UniMatch半监督框架下运行

v2:Stage1只做血管分割，Stage2在血管分割的ROI中只做斑块分割
v3:ROI裁剪改为注意力引导
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

class VesselAttentionGate(nn.Module):
    """
    基于血管分割mask生成软注意力权重，引导斑块分析关注血管区域
    同时保留周围组织信息
    """

    def __init__(self, smooth_kernel_size=15, sigma=3.0):
        super().__init__()
        # 高斯平滑核（扩展注意力范围到血管周围）
        self.smooth_kernel_size = smooth_kernel_size
        self.sigma = sigma

        # 可学习的注意力调制器
        self.attention_modulator = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def _gaussian_smooth(self, x):
        """高斯平滑"""
        kernel_size = self.smooth_kernel_size
        sigma = self.sigma

        # 创建高斯核
        ax = torch.arange(kernel_size, device=x.device) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)

        # 应用卷积
        return F.conv2d(x, kernel, padding=kernel_size // 2)

    def forward(self, vessel_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vessel_mask: [B, H, W] long tensor (0=bg, 1=vessel)
        Returns:
            attention_map: [B, 1, H, W] 软注意力权重，血管区域高权重，
                          周围区域随距离衰减但保留信息
        """
        B, H, W = vessel_mask.shape

        # 1. 生成二值mask
        vessel_binary = (vessel_mask == 1).float().unsqueeze(1)  # [B,1,H,W]

        # 2. 高斯平滑扩展感受野（模拟医生关注周围组织）
        smooth_mask = self._gaussian_smooth(vessel_binary)

        # 3. 可学习调制（网络自适应学习最优注意力分布）
        attention = self.attention_modulator(
            torch.cat([vessel_binary, smooth_mask], dim=1)
        )

        # 4. 确保注意力中心强度，边缘衰减
        # 中心区域权重 0.8-1.0，周围区域 0.3-0.7
        attention = attention * 0.7 + vessel_binary * 0.3

        return attention

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
    ConvNeXt编码器 + 轻量解码器，专注注意力内斑块分割。
    """
    def __init__(
        self,
        in_chns: int = 1,
        seg_cls: int = 2,
        encoder_pth: Optional[str] = None,
    ):
        super().__init__()

        # ── Encoder ──
        self.encoder = ConvNeXt(
            in_chans=in_chns,
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768],
            drop_path_rate=0.1,  # 注意力引导阶段稍小dropout
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
        attention_map: torch.Tensor,
        fp: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            roi_img: [B, 1, 256, 256] 完整输入图像
            attention_map: [B, 1, 256, 256] 软注意力权重
        """
        # 注意力加权输入（早期融合）
        attended_img = roi_img * attention_map

        feats = self.encoder(attended_img)

        if fp:
            feats = [self.fp_drop(f) for f in feats]

        attended_feats = []
        for i, feat in enumerate(feats):
            scale = feat.shape[-1] / attention_map.shape[-1]
            attn_resized = F.interpolate(
                attention_map,
                size=feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            attended_feats.append(feat * (0.5 + 0.5 * attn_resized))

        seg_feat, deep_feat = self._decode(attended_feats)
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
    Stage 2: 注意力引导
    Stage 3: PlaqueConvNeXt + DualViewFusion → 斑块分割
    Stage 4: VulnerabilityClassifier         → 脆弱性分类
    """
    def __init__(
        self,
        in_chns:       int   = 1,
        seg_class_num: int   = 3,
        cls_class_num: int   = 1,
        encoder_pth:   Optional[str] = None,
        unet_base_ch:  int   = 32,
        attn_heads:    int   = 4,      # 注意力头数
        **kwargs
    ):
        super().__init__()
        self.seg_class_num = seg_class_num
        self.vessel_cls = 2
        self.plaque_cls = 2

        # ── Stage 1: 血管UNet (双视角共享权重) ──
        self.vessel_unet = VesselUNet(
            in_chns=in_chns,
            seg_cls=2,
            base_ch=unet_base_ch
        )

        # 血管注意力生成器（双视角共享权重）
        self.vessel_attention = VesselAttentionGate(
            smooth_kernel_size=15,  # 关注血管周围15像素范围
            sigma=3.0
        )

        # ── Stage 2+3: 斑块ConvNeXt (双视角共享编码器，独立解码头) ──
        self.plaque_encoder_long = PlaqueConvNeXtDecoder(
            in_chns=in_chns,
            seg_cls=2,
            encoder_pth=encoder_pth,
        )
        # 横轴视角共享同一个ConvNeXt编码器权重
        self.plaque_encoder_trans = PlaqueConvNeXtDecoder(
            in_chns=in_chns,
            seg_cls=2,
            encoder_pth=None,           # 不重复加载
        )
        # 共享编码器权重（仅训练一份编码器）
        self.plaque_encoder_trans.encoder = self.plaque_encoder_long.encoder

        # ── Stage 3: 双视角注意力融合 ──
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
            vessel_attn   : [B, 1, H, W]
            unet_bn       : [B, 512, H/16, W/16]
        """
        vessel_logits, unet_bn = self.vessel_unet(x)   # 血管分割
        vessel_pred = vessel_logits.argmax(dim=1)       # [B,H,W]

        return vessel_logits, vessel_pred, unet_bn

    # ──────────────────────────────────────────────────────
    # 内部流程：Stage2+3 斑块分割 + 双视角融合
    # ──────────────────────────────────────────────────────
    def _stage2_forward(
        self,
        x_long:   torch.Tensor,
        x_trans:  torch.Tensor,
        vessel_pred_l:  torch.Tensor,
        vessel_pred_t:  torch.Tensor,
        fp:       bool = False
    ):

        # 生成软注意力
        attn_long = self.vessel_attention(vessel_pred_l)  # [B,1,256,256]
        attn_trans = self.vessel_attention(vessel_pred_t)

        # ConvNeXt斑块分割（注意力引导）
        logits_l, deep_l = self.plaque_encoder_long(
            x_long, attn_long, fp=fp
        )
        logits_t, deep_t = self.plaque_encoder_trans(
            x_trans, attn_trans, fp=fp
        )

        # 双视角注意力融合
        deep_l_fused, deep_t_fused = self.dual_view_fusion(deep_l, deep_t)

        return logits_l, logits_t, deep_l_fused, deep_t_fused

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
        """
        # ── Stage 1 ──
        vessel_l, pred_l, unet_bn_l = self._stage1_forward(x_long)
        vessel_t, pred_t, unet_bn_t = self._stage1_forward(x_trans)

        # 梯度解耦：早期阶段2不更新阶段1参数
        if self.detach_stage2_grad and self.training:
            x_long_s2  = x_long.detach()
            x_trans_s2 = x_trans.detach()
            pred_l_s2 = pred_l.detach()
            pred_t_s2 = pred_t.detach()
            unet_bn_l_cls = unet_bn_l.detach()
            unet_bn_t_cls = unet_bn_t.detach()
        else:
            x_long_s2  = x_long
            x_trans_s2 = x_trans
            pred_l_s2 = pred_l
            pred_t_s2 = pred_t
            unet_bn_l_cls = unet_bn_l
            unet_bn_t_cls = unet_bn_t

        # ── Stage 2+3 ──
        plaque_l, plaque_t, deep_l, deep_t = self._stage2_forward(
            x_long_s2, x_trans_s2,
            pred_l_s2, pred_t_s2,
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
