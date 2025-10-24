# -*- coding: utf-8 -*-
"""
US Encoder Focus Grid with Foreground (Fan) Masking

- 支持 DINOv3 ViT-L/16（或其他 ViT/16）
- 自动生成 US 扇形前景掩膜，避免注意力/范数落在黑色背景
- 掩膜内归一化 + 质量评分，挑选前景关注最强的 8 张
- 自动剔除 register tokens
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from transformers import AutoImageProcessor, AutoModel

# ========= 配置 =========
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
IMG_PATH = "G:\\Dino3Registration\\test_feature\\MRI_image\\Case1-T1.jpeg"
SAVE_PATH = "encoder_focus_grid_masked_US_vitl16.png"
NUM_MAPS = 8           # 四周展示多少张（建议 8）
ALPHA = 0.5            # 叠加透明度
METRIC = "mass"        # 选图指标: 'mass' (ROI内积分) 或 'contrast' (ROI vs BG 对比)
USE_ATT_LAST = True    # True: 只看最后一层注意力；False: 会回退到特征范数或多层

# ========= 小工具 =========
def safe_open(path: str) -> Image.Image:
    """修正你路径里的异常空格并读图"""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        fixed = path.replace("FeatureExtration /", "FeatureExtration/")
        if fixed != path:
            print(f"[info] 修正路径:\n  {path}\n-> {fixed}")
            return Image.open(fixed).convert("RGB")
        raise
#0-1 normalize
def get_patch_size(model) -> int:
    for cfg in [getattr(model.config, "vision_config", None), model.config]:
        if cfg is not None and hasattr(cfg, "patch_size"):
            return int(getattr(cfg, "patch_size"))
    return 16

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def upsample(hm_t: torch.Tensor, H0: int, W0: int) -> np.ndarray:
    """hm_t: (Hp, Wp) -> (H0,W0) numpy"""
    hm = hm_t.unsqueeze(0).unsqueeze(0)  # (1,1,Hp,Wp)
    hm = F.interpolate(hm, size=(H0, W0), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    return hm

def build_fg_mask_from_nonzero(img_pil: Image.Image, bg_value: int = 0, tol: int = 0,
                               min_keep_ratio: float = 0.02) -> np.ndarray:
    """
    用“灰度 != bg_value”作为前景（非二次阈值/自适应），最直接的前景获取。
    - bg_value: 背景像素值（默认0）。如是JPEG有压缩噪声，可配合 tol 使用。
    - tol: 容差；当 tol>0 时，判定条件变为 |gray - bg_value| > tol。
    - min_keep_ratio: 若前景面积极小（可能图像本身没有0背景），则回退为全1（不过滤）。
    返回: (H,W) 0/1 掩膜
    """
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

    if tol > 0:
        mask = (np.abs(gray.astype(np.int16) - int(bg_value)) > int(tol)).astype(np.uint8)
    else:
        mask = (gray != int(bg_value)).astype(np.uint8)

    # 仅保留最大连通域，避免角落/边框零星噪点
    num, lab = cv2.connectedComponents(mask)
    if num > 1:
        areas = [(lab == i).sum() for i in range(1, num)]
        keep = 1 + int(np.argmax(areas))
        mask = (lab == keep).astype(np.uint8)

    # 如果前景太小，认为这张图可能没有“0背景”，则不使用mask（回退全1）
    if mask.sum() < min_keep_ratio * mask.size:
        mask[:] = 1

    return mask


#only normalize inside the mask
def maskwise_normalize(hm: np.ndarray, mask: np.ndarray, eps=1e-8) -> np.ndarray:
    out = np.ones_like(hm, dtype=np.float32) * 0.05  # 背景低权重
    fg = hm[mask == 1]
    if fg.size == 0:
        return out
    mn, mx = float(fg.min()), float(fg.max())
    if mx - mn < eps:
        return out
    out[mask == 1] = (fg - mn) / (mx - mn + eps)
    return out



def score_map(hm: np.ndarray, mask: np.ndarray, metric="mass") -> float:
    """给热力图一个分数，用于选 top-k"""
    area_fg = mask.sum() + 1e-8
    if metric == "mass":
        return float((hm * (mask > 0)).sum() / area_fg)
    # 对比度：前景中位-背景中位，归一到前景STD
    fg = hm[mask == 1]
    bg = hm[mask == 0]
    if fg.size == 0 or bg.size == 0:
        return 0.0
    return float((np.median(fg) - np.median(bg)) / (np.std(fg) + 1e-6))

def render_all_feature_layers(
    img_pil: Image.Image,
    out,
    H0: int, W0: int,
    Hp: int, Wp: int, num_patch: int,
    mask: np.ndarray,
    save_path: str,
    cols: int = 6,           # 每行列数；24层的话 6列×4行正好
    gamma: float = 0.7,      # 显示时的γ校正
    dpi: int = 220,          # 保存分辨率
):
    """
    把所有 hidden_states(除embedding层) 的 patch特征L2范数热力图拼接成一张大图。
    - 自动忽略 CLS + register tokens，只取前 Hp*Wp 个真实 patch。
    - 在 US 扇形前景内归一化，背景抑制为常数。
    - 标注每层前景最大响应位置（红色+号）。
    """
    # 1) 拿到所有层（list[层], 每层 (B, 1+T_all, D)）
    hs_list = list(out.hidden_states)
    assert len(hs_list) >= 2, "hidden_states 里至少应包含 embedding 层和一个 block 输出"
    layers = hs_list[1:]  # 去掉 embedding 层（第0层）

    # 2) 逐层生成热力图
    heatmaps, titles = [], []
    L = len(layers)              # ViT-L 通常是 24
    for li, hs in enumerate(layers):  # li: 0..L-1
        # 只取真实 patch（屏蔽 CLS 与 register）
        pt = hs[:, 1:1 + num_patch, :]          # (1, Hp*Wp, D)
        feat_norm = pt.norm(p=2, dim=-1)[0]     # (Hp*Wp,)

        hm = upsample(feat_norm.reshape(Hp, Wp), H0, W0)  # -> (H0,W0)
        hm = maskwise_normalize(hm, mask)

        # 负向层编号（与你之前风格一致：倒数第1层记为 -1）
        neg_idx = li - L   # li=0 -> -L, li=L-1 -> -1
        heatmaps.append(hm)
        titles.append(f"Feat L2 layer {neg_idx}")

    # 3) 画大网格
    rows = int(np.ceil(L / cols))
    fig_w = cols * 3.0      # 每格约 3 英寸，可按需调
    fig_h = rows * 3.2
    fig = plt.figure(figsize=(fig_w, fig_h))

    for i, (hm, ttl) in enumerate(zip(heatmaps, titles)):
        r, c = divmod(i, cols)
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img_pil)
        show_hm = np.power(hm, gamma)           # γ 校正，抬亮高响应
        ax.imshow(show_hm, cmap="jet", alpha=0.5)

        # 只在前景内找峰值并标注
        hm_fg = hm.copy()
        hm_fg[mask == 0] = 0
        if hm_fg.max() > 0:
            y, x = np.unravel_index(np.argmax(hm_fg), hm_fg.shape)
            ax.plot([x], [y], marker="+", color="red", markersize=8, markeredgewidth=1.8)

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl, fontsize=9)

    plt.tight_layout()
    big_path = os.path.splitext(save_path)[0] + "_ALL_LAYERS.png"
    plt.savefig(big_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] 已保存所有层可视化到: {big_path}")



# ========= 主流程 =========
def main():
    # 读图
    img = safe_open(IMG_PATH)
    W0, H0 = img.size
    print(f"[info] 原图尺寸: {W0}x{H0}")

    # 模型
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    # forward
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        out = model(**inputs, output_attentions=True, output_hidden_states=True)
        #check out full layer number and shape
        print(f"[info] hidden_states 总层数: {len(out.hidden_states)}")
        for i, hs in enumerate(out.hidden_states):
            print(f"  Layer {i:2d}: shape = {tuple(hs.shape)}")

   
    
    # 计算 patch 网格
    _, _, Hs, Ws = inputs["pixel_values"].shape  # (B,C,H',W')
    p = get_patch_size(model)
    Hp, Wp = Hs // p, Ws // p
    num_patch = Hp * Wp

    # 扇形前景掩膜（原图尺寸）
    mask = build_fg_mask_from_nonzero(img, bg_value=0, tol=0, min_keep_ratio=0.02)
    render_all_feature_layers(
    img_pil=img,
    out=out,
    H0=img.size[1], W0=img.size[0],   # 注意 PIL size=(W,H)
    Hp=Hp, Wp=Wp, num_patch=num_patch,
    mask=mask,
    save_path=SAVE_PATH,              # 用它生成 *_ALL_LAYERS.png
    cols=6,                           # 24层→6列×4行
    gamma=0.7,
    dpi=220
)

    heatmaps = []
    titles = []



        
if __name__ == "__main__":
    main()
