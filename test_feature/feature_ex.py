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
IMG_PATH = "G:\\Dino3Registration\\data\\sliced_png\\Case6\\MRI\\coronal\\slice_224.png"

SAVE_PATH = "encoder_focus_grid_masked_MRi_vitl16.png"

NUM_MAPS = 8           # 四周展示多少张（建议 8）
ALPHA = 0.5            # 叠加透明度
METRIC = "contrast"        # 选图指标: 'mass' (ROI内积分) 或 'contrast' (ROI vs BG 对比)
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
    #print total out shapes
    print(f"[info] model output 总形状: {tuple(out.shape)}")

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


    heatmaps = []
    titles = []

    # ========== if use attentions，用 CLS->patch ==========
    if getattr(out, "attentions", None) is not None and out.attentions:
        att = out.attentions[-1] if USE_ATT_LAST else out.attentions[0]  # (B, heads, 1+T_all, 1+T_all)
        B, Hh, Ttok, _ = att.shape
        assert B == 1

        # 只取前 Hp*Wp 个 patch（忽略 register tokens）
        cls2patch = att[:, :, 0, 1:1 + num_patch]  # (1, heads, Hp*Wp)
        # 生成每个 head 的热力图（掩膜内归一化）
        for h in range(cls2patch.shape[1]):
            hm = cls2patch[0, h]  # (Hp*Wp,)
            # 先自身归一化再上采样
            hm = hm / (hm.max() + 1e-8)
            hm = upsample(hm.reshape(Hp, Wp), H0, W0)
            hm = maskwise_normalize(hm, mask)
            heatmaps.append(hm)
            titles.append(f"Attn Head {h}")
        mode = "attn"

    # ========== no attentions，use hidden states l2 norm ==========
    else:
        # code is going there
        hs_list = list(out.hidden_states)  # list[层], 每层 (B, 1+T_all, D)
        # 去掉 embedding 层（第 0 层），从后往前挑 NUM_MAPS*2 个候选，再选 top-k
        hs_list = hs_list[1:]
        candidates = []
        for idx, hs in enumerate(hs_list[-(NUM_MAPS * 2):]):
            pt = hs[:, 1:1 + num_patch, :]          # 只取前 Hp*Wp patch
            feat_norm = pt.norm(p=2, dim=-1)[0]     # (Hp*Wp,)
            hm = upsample(feat_norm.reshape(Hp, Wp), H0, W0)
            hm = maskwise_normalize(hm, mask)
            candidates.append((hm, f"Feat L2 layer {-len(hs_list) + (len(hs_list) - (NUM_MAPS * 2) + idx)}"))

        # 评分并选 top-k
        scored = [(score_map(h, mask, METRIC), i) for i, (h, _) in enumerate(candidates)]
        scored.sort(reverse=True, key=lambda x: x[0])
        keep_idx = [i for _, i in scored[:NUM_MAPS]]
        for i in keep_idx:
            heatmaps.append(candidates[i][0])
            titles.append(candidates[i][1])
        mode = "feat"

    # ========== 画 3×3 网格 ==========
    # 若不足 8 张，重复补足
    if len(heatmaps) < NUM_MAPS:
        heatmaps = heatmaps * (NUM_MAPS // max(1, len(heatmaps)) + 1)
        titles = titles * (NUM_MAPS // max(1, len(titles)) + 1)
    heatmaps = heatmaps[:NUM_MAPS]
    titles = titles[:NUM_MAPS]

    fig = plt.figure(figsize=(10, 10))
    pos = [(1,1),(1,2),(1,3),
           (2,1),       (2,3),
           (3,1),(3,2),(3,3)]

    for i, hm in enumerate(heatmaps[:8]):
        r, c = pos[i]
        ax = plt.subplot(3, 3, (r - 1) * 3 + c)
        ax.imshow(img)
        # γ 校正提升亮部（可调 0.6~0.8）
        show_hm = np.power(hm, 0.7)
        ax.imshow(show_hm, cmap="jet", alpha=ALPHA)
        # 标红最大点（只在前景内查找）
        hm_fg = hm.copy()
        hm_fg[mask == 0] = 0
        if hm_fg.max() > 0:
            y, x = np.unravel_index(np.argmax(hm_fg), hm_fg.shape)
            ax.plot([x], [y], marker="+", color="red", markersize=10, markeredgewidth=2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titles[i])

    ax_mid = plt.subplot(3, 3, 5)
    ax_mid.imshow(img); ax_mid.set_xticks([]); ax_mid.set_yticks([])
    ax_mid.set_title("Original")

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"[done] 已保存: {SAVE_PATH}")
    print(f"[debug] grid={Hp}x{Wp}={num_patch}, patch={p}, mode={mode}, maps={len(heatmaps)}")

if __name__ == "__main__":
    main()
