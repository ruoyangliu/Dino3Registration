import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ====== 配置 ======
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
IMG_PATH = "/home/rl23/Desktop/code/Dinov3FeatureExtration /dinov3/test_feature/US_image/Case1-US-before.jpeg"
SAVE_PATH = "attn_grid.png"     # 输出文件
NUM_HEADS_TO_SHOW = 8           # 展示多少个 head
ALPHA = 0.5                     # 叠加透明度
USE_LAST_LAYER = True           # True: 只看最后一层；False: 也可改为某一层索引

# ====== 小工具 ======
def safe_open(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        fixed = path.replace("FeatureExtration /", "FeatureExtration/")
        if fixed != path:
            print(f"[info] 修正路径:\n  {path}\n->{fixed}")
            return Image.open(fixed).convert("RGB")
        raise

def get_patch_size(model) -> int:
    if hasattr(model.config, "vision_config") and hasattr(model.config.vision_config, "patch_size"):
        return int(model.config.vision_config.patch_size)
    if hasattr(model.config, "patch_size"):
        return int(model.config.patch_size)
    return 16

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12: return np.zeros_like(x)
    return (x - mn) / (mx - mn)

# ====== 主流程 ======
def main():
    # 1) 载图像/模型
    img = safe_open(IMG_PATH)
    W0, H0 = img.size
    print(f"[info] 原图尺寸: {W0}x{H0}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    # 2) 前向，拿 attentions
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    # 3) 计算 patch 网格（允许非方形）
    _, _, Hs, Ws = inputs["pixel_values"].shape
    p = get_patch_size(model)
    Hp, Wp = Hs // p, Ws // p
    num_patch = Hp * Wp

    # 4) 取最后一层的注意力（或你自定义层）
    #    形状: (B, heads, 1+T_all, 1+T_all)
    att_all_layers = out.attentions
    att = att_all_layers[-1] if USE_LAST_LAYER else att_all_layers[0]
    B, Hh, TT, _ = att.shape
    assert B == 1, "这里假设单张图像"

    # 5) 从 CLS -> 前 Hp*Wp 个 patch 的注意力（忽略 register tokens）
    #    注意：有 register tokens，因此只取 1:1+num_patch
    cls_to_patch = att[:, :, 0, 1:1+num_patch]      # (1, heads, Hp*Wp)
    # 可选：按 head 归一化（便于跨 head 对比）
    cls_to_patch = cls_to_patch / (cls_to_patch.max(dim=2, keepdim=True).values + 1e-8)

    # 6) 选择要展示的 heads（简单取前 N 个；也可以根据“峰值/熵”排序）
    heads_to_show = list(range(min(NUM_HEADS_TO_SHOW, Hh)))

    # 7) 生成每个 head 的热力图（插值到原图）
    heatmaps = []
    for h in heads_to_show:
        hm = cls_to_patch[0, h]                    # (Hp*Wp,)
        hm = hm.reshape(1, 1, Hp, Wp)              # (1,1,Hp,Wp)
        hm_up = F.interpolate(hm, size=(H0, W0), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        hm_up = normalize01(hm_up)
        heatmaps.append(hm_up)

    # 8) 画九宫格：四周热力图 + 中间原图；在每个热力图上画红十字（最大点）
    fig = plt.figure(figsize=(10, 10))
    # 布局：3x3，中央(2,2)放原图
    idx_map = [(1,1), (1,2), (1,3),
               (2,1),         (2,3),
               (3,1), (3,2), (3,3)]
    # 先画四周
    for i, hm in enumerate(heatmaps):
        r, c = idx_map[i]
        ax = plt.subplot(3, 3, (r-1)*3 + c)
        ax.imshow(img)
        ax.imshow(hm, cmap="viridis", alpha=ALPHA)
        # 找最大点并画红十字
        yx = np.unravel_index(np.argmax(hm), hm.shape)
        y, x = int(yx[0]), int(yx[1])
        ax.plot([x], [y], marker="+", color="red", markersize=10, markeredgewidth=2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Head {heads_to_show[i]}")

    # 中间原图
    ax_mid = plt.subplot(3, 3, 5)
    ax_mid.imshow(img)
    ax_mid.set_xticks([]); ax_mid.set_yticks([])
    ax_mid.set_title("Original")

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"[done] 已保存: {SAVE_PATH}")
    print(f"[debug] pixel_values={Hs}x{Ws}, patch={p}, grid={Hp}x{Wp}, heads={Hh}")

if __name__ == "__main__":
    main()
