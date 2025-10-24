import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch.nn.functional as F
import cv2

# ========== 这里用你已有的小工具 ==========
# safe_open, get_patch_size, build_fg_mask_from_nonzero, upsample, maskwise_normalize
# 请在脚本里确保已经定义好上面的函数


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


# ========== PCA 可视化工具 ==========
def pca_maps_from_layer(hs_layer, Hp, Wp, num_patch, H0, W0, mask):
    """
    hs_layer: (1, 1+T_all, D)，某一层 hidden state
    返回: (PC1, PC2, PC3), RGB 合成
    """
    # 取真实 patch 特征
    X = hs_layer[:, 1:1+num_patch, :][0].cpu().numpy()  # (N,D)

    # 中心化
    Xc = X - X.mean(axis=0, keepdims=True)

    # PCA via SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # Vt: (D,D)
    Z = Xc @ Vt[:3, :].T  # (N,3)

    pcs = []
    for k in range(3):
        pc = Z[:, k].reshape(Hp, Wp)
        pc = upsample(torch.from_numpy(pc.astype(np.float32)), H0, W0)
        pc = maskwise_normalize(pc, mask)
        pcs.append(pc)

    # RGB 合成图
    rgb = np.stack(pcs, axis=-1)
    rgb = rgb / (rgb.max(axis=(0,1))+1e-8)  # 每通道归一化
    return pcs, rgb

def visualize_pca_layer(img, pcs, rgb, title_prefix, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for i in range(3):
        axes[i].imshow(img)
        axes[i].imshow(np.power(pcs[i], 0.8), cmap='jet', alpha=0.5)
        axes[i].set_title(f"{title_prefix} - PC{i+1}")
        axes[i].axis('off')
    axes[3].imshow(rgb)
    axes[3].set_title(f"{title_prefix} - RGB(PC1,2,3)")
    axes[3].axis('off')
    plt.tight_layout()
    sp = save_path.replace(".png", f"_{title_prefix}_PCA.png")
    plt.savefig(sp, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] PCA 可视化已保存: {sp}")
    
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

# ========== 主函数 ==========
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
IMG_PATH   = "G:\\Dino3Registration\\test_feature\\MRI_image\\Case1-T1.jpeg"
SAVE_PATH  = "feature _layer_PCA.png"

def main():
    # 1) 读图
    img = safe_open(IMG_PATH)
    W0, H0 = img.size
    print(f"[info] 原图尺寸: {W0}x{H0}")

    # 2) 模型和前向
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    # 3) 计算 patch 网格参数
    _, _, Hs, Ws = inputs["pixel_values"].shape
    p = get_patch_size(model)
    Hp, Wp = Hs // p, Ws // p
    num_patch = Hp * Wp
    print(f"[info] patch_size={p}, Hp={Hp}, Wp={Wp}, num_patch={num_patch}")

    # 4) 生成前景掩膜
    mask = build_fg_mask_from_nonzero(img, bg_value=0, tol=0, min_keep_ratio=0.02)

    # 5) 选一层做 PCA 可视化
    layer_idx_from_end = 10  # 倒数第10层（通常是中层）
    hs_layer = out.hidden_states[-layer_idx_from_end]

    pcs, rgb = pca_maps_from_layer(hs_layer, Hp, Wp, num_patch, H0, W0, mask)
    visualize_pca_layer(img, pcs, rgb, f"Layer_-{layer_idx_from_end}", SAVE_PATH)


if __name__ == "__main__":
    main()
