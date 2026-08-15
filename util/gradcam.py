import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional


class GradCAM:
    """GradCAM for dual-view segmentation + classification model"""

    def __init__(self, model, target_layer_names: List[str], device='cuda'):
        """
        Args:
            model: ConvNeXt_UniMatch or Echocare model
            target_layer_names: 要可视化的层名，如 ['decoder4.conv1[0]', 'encoder.stages.2']
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.device = device
        self.model.eval()

        self.gradients = {}
        self.activations = {}

        self._register_hooks(target_layer_names)

    def _register_hooks(self, layer_names: List[str]):
        """注册前向和反向钩子"""
        for name in layer_names:
            # 根据点分隔的路径获取模块
            module = self._get_module_by_name(name)
            if module is None:
                print(f"Warning: Layer {name} not found, skipping")
                continue

            module.register_forward_hook(self._make_forward_hook(name))
            module.register_backward_hook(self._make_backward_hook(name))

    def _get_module_by_name(self, name: str):
        """通过名称获取模块，支持 'decoder4.conv1[0]' 格式"""
        parts = name.split('.')
        module = self.model

        for part in parts:
            # 处理索引访问，如 'conv1[0]'
            if '[' in part and ']' in part:
                base_name = part[:part.index('[')]
                idx = int(part[part.index('[') + 1:part.index(']')])
                module = getattr(module, base_name)[idx]
            else:
                module = getattr(module, part, None)
                if module is None:
                    return None
        return module

    def _make_forward_hook(self, name: str):
        def hook(module, input, output):
            self.activations[name] = output.detach()

        return hook

    def _make_backward_hook(self, name: str):
        def hook(module, grad_in, grad_out):
            self.gradients[name] = grad_out[0].detach()

        return hook

    def _get_cam(self, layer_name: str, class_idx: int = None,
                 logits: torch.Tensor = None) -> np.ndarray:
        """
        计算单个层的 CAM

        Args:
            layer_name: 层名
            class_idx: 目标类别索引 (None 表示使用最大激活)
            logits: 模型输出 (如果不提供，需要手动调用 backward)
        """
        if layer_name not in self.activations:
            raise KeyError(f"Layer {layer_name} not found in activations")

        activations = self.activations[layer_name]  # [B, C, H, W]
        gradients = self.gradients[layer_name]  # [B, C, H, W]

        # 全局平均池化梯度
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # 加权求和
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [B, 1, H, W]

        # ReLU 激活
        cam = F.relu(cam)

        # 归一化到 [0, 1]
        cam = cam.squeeze(1)  # [B, H, W]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam.cpu().numpy()

    def compute_cam_for_segmentation(self, x_long: torch.Tensor, x_trans: torch.Tensor,
                                     target_class: int = None,
                                     view: str = 'long') -> Tuple[np.ndarray, np.ndarray]:
        """
        为分割任务计算 CAM

        Args:
            x_long, x_trans: 输入张量 [1, 1, H, W]
            target_class: 目标类别 (0:背景, 1:斑块, 2:血管), None表示最大激活
            view: 'long' 或 'trans' 选择哪个视图

        Returns:
            cam: [H, W] CAM 热力图
            overlay: 叠加后的可视化图像
        """
        self.model.zero_grad()

        # 前向传播
        segL, segT, cls_out = self.model(x_long, x_trans)

        # 选择目标输出
        if view == 'long':
            seg_out = segL  # [B, 3, H, W]
        else:
            seg_out = segT

        # 选择目标类别
        if target_class is None:
            # 取最大激活的类别
            target_class = torch.argmax(seg_out, dim=1)[0, 0, 0].item()

        # 计算损失并反向传播
        loss = seg_out[0, target_class].mean()
        loss.backward(retain_graph=True)

        # 获取 CAM (使用最后一个 decoder 层)
        cam = self._get_cam(['encoder.stages.3'])  # 或其他层

        return cam[0], target_class

    def compute_cam_for_classification(self, x_long: torch.Tensor, x_trans: torch.Tensor,
                                       target_class: int = 0) -> Tuple[Dict[str, np.ndarray], float]:
        """
        为分类任务计算 CAM

        Returns:
            cams: {'long_encoder_last': cam_array, 'trans_encoder_last': cam_array, ...}
            probability: 预测概率
        """
        self.model.zero_grad()

        # 前向传播
        segL, segT, cls_out = self.model(x_long, x_trans)

        # 分类输出
        cls_prob = torch.sigmoid(cls_out)
        cls_pred = (cls_prob >= 0.5).long()

        # 对目标类别计算梯度
        loss = cls_out[0, target_class]
        loss.backward(retain_graph=True)

        cams = {}

        # 获取多个层的 CAM
        for layer_name in self.activations.keys():
            cam = self._get_cam(layer_name)
            cams[layer_name] = cam[0]

        return cams, cls_prob[0, 0].item()

    def clear(self):
        """清理钩子"""
        self.gradients.clear()
        self.activations.clear()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    将热力图叠加到原始图像上

    Args:
        image: [H, W] 或 [H, W, 3] 灰度或RGB图像，值范围 [0, 1]
        heatmap: [H, W] 热力图，值范围 [0, 1]
        alpha: 透明度

    Returns:
        overlay: [H, W, 3] RGB图像
    """
    # 转换为3通道RGB
    if image.ndim == 2:
        img_rgb = np.stack([image, image, image], axis=2)
    else:
        img_rgb = image.copy()

    # 生成彩色热力图
    heatmap_color = cv2.applyColorMap(np.uint8(heatmap * 255), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

    # 叠加
    overlay = (1 - alpha) * img_rgb + alpha * heatmap_color
    overlay = np.clip(overlay, 0, 1)

    return overlay


def save_cam_visualization(image: np.ndarray, cam: np.ndarray, save_path: str,
                           title: str = None, alpha: float = 0.5):
    """保存CAM可视化结果"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    axes[0].imshow(image, cmap='gray' if image.ndim == 2 else None)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 热力图
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')

    # 叠加图
    overlay = overlay_heatmap(image, cam, alpha)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved CAM visualization to {save_path}")