import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


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


class ConvNeXt(nn.Module):
    r""" ConvNeXt Backbone """

    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()
        self.out_indices = out_indices
        self.dims = dims

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),  # Note: Standard ConvNeXt uses stride 4 in stem
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Norm layers for feature extraction
        for i_layer in range(4):
            layer = LayerNorm(dims[i_layer], eps=1e-6, data_format="channels_first")
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return outs

    def forward(self, x):
        return self.forward_features(x)

    def load_pretrained_weights(self, weight_path, in_chans=1):
        """
        加载预训练权重，兼容单通道输入

        Args:
            weight_path: 权重文件路径
            in_chans: 输入通道数（默认为1，因为你的数据是单通道超声图像）
        """
        print(f"Loading pretrained weights from {weight_path}")

        # 加载权重
        state_dict = torch.load(weight_path, map_location='cpu')

        # 如果保存的是完整模型字典（如官方权重直接是state_dict）
        if 'model' in state_dict:
            state_dict = state_dict['model']

        # 处理输入通道不匹配的问题
        # 原始ConvNeXt是3通道输入，你的模型是1通道
        if in_chans != 3:
            # 找到stem层的权重
            stem_weight_key = 'downsample_layers.0.0.weight'  # stem的卷积层

            if stem_weight_key in state_dict:
                # 获取原始3通道权重
                original_weight = state_dict[stem_weight_key]  # [96, 3, 4, 4]

                if in_chans == 1:
                    # 方式1：取RGB三个通道的平均（推荐）
                    new_weight = original_weight.mean(dim=1, keepdim=True)  # [96, 1, 4, 4]
                    state_dict[stem_weight_key] = new_weight
                    print(f"Converted stem weights from {original_weight.shape} to {new_weight.shape}")
                else:
                    # 其他通道数，需要调整（你的情况不需要）
                    pass

        # 加载权重（strict=False允许部分层不匹配）
        # 分类头（cls_head）和分割头（seg_head_*）的权重会随机初始化
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        print(f"Missing keys: {len(missing_keys)} (these will be randomly initialized)")
        print(f"Unexpected keys: {len(unexpected_keys)} (these are ignored)")

        return self


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


class ConvNeXt_UniMatch(nn.Module):
    def __init__(self,
                 in_chns=1,
                 seg_class_num=3,
                 cls_class_num=1,
                 encoder_pth=None,
                 **kwargs):
        super().__init__()

        # --- Encoder (Shared for both views) ---
        # ConvNeXt Tiny Config: dims=[96, 192, 384, 768]
        self.encoder = ConvNeXt(in_chans=in_chns,
                                depths=[3, 3, 9, 3],
                                dims=[96, 192, 384, 768],
                                drop_path_rate=0.2)

        if encoder_pth is not None:
            self.encoder.load_pretrained_weights(encoder_pth, in_chans=in_chns)

        dims = [96, 192, 384, 768]

        # --- Decoder (Shared weights for efficiency) ---
        # 768 -> 384
        self.decoder4 = DecoderBlock(dims[3], dims[2], dims[2])
        # 384 -> 192
        self.decoder3 = DecoderBlock(dims[2], dims[1], dims[1])
        # 192 -> 96
        self.decoder2 = DecoderBlock(dims[1], dims[0], dims[0])
        # 96 -> 64 (Final resolution restoration block)
        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  # stride 4 stem recovery
            nn.Conv2d(dims[0], 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # --- Segmentation Heads (Separate for Long/Trans) ---
        self.seg_head_long = nn.Conv2d(64, seg_class_num, kernel_size=1)
        self.seg_head_trans = nn.Conv2d(64, seg_class_num, kernel_size=1)

        # 简化分类头：直接拼接两个视图的深层特征
        self.cls_pool = nn.AdaptiveAvgPool2d(1)  # 仍保留用于池化
        self.cls_head = nn.Sequential(
            nn.Linear(dims[3] * 2, 512),  # 768*2=1536 -> 512
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),  # 增加dropout率防止过拟合
            nn.Linear(512, cls_class_num)
        )

        # ============= 改动点2：添加梯度分离控制 =============
        # 这个标志控制是否阻止分类梯度回传到编码器
        self.detach_cls_grad = True  # 默认为True，分类梯度不更新编码器

        # Feature perturbation dropout for UniMatch
        self.fp_drop = nn.Dropout2d(0.5)

    def set_cls_detach(self, detach=True):
        """控制分类梯度是否回传到编码器"""
        self.detach_cls_grad = detach

    def _decode(self, features):
        """
        Decodes a list of features [f0, f1, f2, f3] from ConvNeXt
        """
        f0, f1, f2, f3 = features  # 96, 192, 384, 768

        x = self.decoder4(f3, f2)  # -> 384
        x = self.decoder3(x, f1)  # -> 192
        x = self.decoder2(x, f0)  # -> 96
        x = self.decoder1(x)  # -> 64, 256x256
        return x

    # ============= 改动点3：新增独立的分类特征提取方法 =============
    def _extract_cls_features(self, features):
        """
        从编码器特征中提取分类相关的特征，独立于分割解码器
        features: [f0, f1, f2, f3] from encoder
        """
        f0, f1, f2, f3 = features

        # 只使用最深层的特征进行全局平均池化
        cls_feat = self.cls_pool(f3).flatten(1)  # [B, 768]

        return cls_feat

    def forward_view(self, x, fp=False, return_cls_only=False):
        """
        Forward pass for a single view.
        Returns:
           final_seg_features (for seg head),
           deepest_features (for cls head)
        """
        # Encoder features: [c1, c2, c3, c4]
        features = self.encoder(x)

        # Apply Feature Perturbation (UniMatch) on the deepest feature if requested
        # Usually applied to encoder features before decoder
        if fp:
            features = [self.fp_drop(f) for f in features]

        # 提取分类特征（独立分支）
        cls_feat = self._extract_cls_features(features)

        # 如果需要梯度分离
        if self.detach_cls_grad and self.training:
            # 复制特征用于分割，但不让分类梯度影响
            features_for_seg = [f.detach() for f in features]
        else:
            features_for_seg = features

        if return_cls_only:
            return None, cls_feat

        # Decode
        seg_feat = self._decode(features)

        return seg_feat, cls_feat

    def forward(self, x_long, x_trans, need_fp=False):
        """
        Args:
            x_long: (B, 1, 256, 256)
            x_trans: (B, 1, 256, 256)
            need_fp: Boolean, if True return ((segL, segL_fp), (segT, segT_fp), (cls, cls_fp))
        """

        # 1. Standard Forward
        featL_seg, featL_cls = self.forward_view(x_long, fp=False)
        featT_seg, featT_cls = self.forward_view(x_trans, fp=False)

        # Segmentation Outputs
        segL = self.seg_head_long(featL_seg)
        segT = self.seg_head_trans(featT_seg)

        # Classification Output (使用独立的分类特征)
        cls_feat = torch.cat([featL_cls, featT_cls], dim=1)
        cls_out = self.cls_head(cls_feat)

        if not need_fp:
            return segL, segT, cls_out

        # 2. Feature Perturbation Forward
        featL_seg_fp, featL_cls_fp = self.forward_view(x_long, fp=True)
        featT_seg_fp, featT_cls_fp = self.forward_view(x_trans, fp=True)

        segL_fp = self.seg_head_long(featL_seg_fp)
        segT_fp = self.seg_head_trans(featT_seg_fp)

        cls_feat_fp = torch.cat([featL_cls_fp, featT_cls_fp], dim=1)
        cls_out_fp = self.cls_head(cls_feat_fp)

        return (segL, segL_fp), (segT, segT_fp), (cls_out, cls_out_fp)


# Function to match train.py's get_model expectation somewhat,
# though direct instantiation is preferred.
def convnext_tiny_unimatch(**kwargs):
    model = ConvNeXt_UniMatch(**kwargs)
    return model


if __name__ == '__main__':
    # Test Code
    model = ConvNeXt_UniMatch(in_chns=1, seg_class_num=3, cls_class_num=1)
    x_l = torch.randn(2, 1, 256, 256)
    x_t = torch.randn(2, 1, 256, 256)

    # Test normal forward
    sL, sT, cls = model(x_l, x_t)
    print(f"SegL: {sL.shape}, SegT: {sT.shape}, Cls: {cls.shape}")

    # Test FP forward
    (sL, sL_fp), (sT, sT_fp), (cls, cls_fp) = model(x_l, x_t, need_fp=True)
    print(f"FP SegL: {sL_fp.shape}")
