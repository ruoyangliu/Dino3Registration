# -*- coding: utf-8 -*-
"""
DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m) Feature Check
- 提取每个 Transformer block 的 patch tokens
- 自动剔除 CLS 与 register tokens
- 生成每层 (H_p x W_p) 的热力图并保存
"""

import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

"""
ViT-S/16 distilled  facebook/dinov3-vits16-pretrain-lvd1689m
ViT-S+/16 distilled  facebook/dinov3-vits16plus-pretrain-lvd1689m
ViT-B/16 distilled  facebook/dinov3-vitb16-pretrain-lvd1689m
ViT-L/16 distilled  facebook/dinov3-vitl16-pretrain-lvd1689m
ViT-H+/16 distilled  facebook/dinov3-vith16plus-pretrain-lvd1689m
ViT-7B/16   facebook/dinov3-vit7b16-pretrain-lvd1689m
"""




# ========= 可配参数 =========
MODEL_NAME = "facebook/dinov3-vit7b16-pretrain-lvd1689m"   # ViT-L/16 变体
IMAGE_PATH = r"G:\\Dino3Registration\\data\\sliced_png\\Case6\\MRI\\coronal\\slice_224.png"
# save name add the context after model name /dinov3-

SAVE_PATH  = "G:\\Dino3Registration\\test_feature\\dino3\\"+MODEL_NAME.split("/")[-1]+"_featurecheck.png"

IMAGE_SIZE = 224          # 可选: 224 / 448 / 896 / 1024 ...
PATCH_SIZE = 16           # 对应 ViT-*/16
METRIC     = "l2"         # "l2" 或 "mean" 作为热力图度量
CMAP       = "viridis"    # 热力图 colormap
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ========= 载入模型 & 预处理 =========
print(f"Loading model: {MODEL_NAME}")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# 关键：输出所有层的 hidden_states
model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(DEVICE).eval()

# 读取图像并预处理到指定尺寸（HF 会在模型内部插值位置编码）
img = Image.open(IMAGE_PATH).convert("RGB")
inputs = processor(images=img, return_tensors="pt", size={"shortest_edge": IMAGE_SIZE})
pixel_values = inputs["pixel_values"].to(DEVICE)  # [1,3,H,W]

with torch.no_grad():
    outputs = model(pixel_values=pixel_values)
    # hidden_states: tuple(len = 1 + num_blocks), 第0个是embeddings输入后的状态，其后是每个 block 的输出
    hidden_states = outputs.hidden_states

num_blocks = len(hidden_states) - 1
print(f"Total blocks: {num_blocks}")

# 读出 register token 数量（若无该字段，则默认为0）
num_register_tokens = getattr(model.config, "num_register_tokens", 0)
print(f"num_register_tokens: {num_register_tokens}")

# 计算当前特征图的 patch 网格尺寸 (Hp, Wp)
# 注意：经过processor后的实际输入尺寸在 pixel_values 里，可以直接用
_, _, H, W = pixel_values.shape
Hp = H // PATCH_SIZE
Wp = W // PATCH_SIZE
print(f"Patch grid: {Hp} x {Wp} (from {H}x{W} // {PATCH_SIZE})")

# ========= 从每层提取 patch tokens，并生成热力图 =========
def tokens_to_map(patch_tokens: torch.Tensor, metric: str = "l2"):
    """
    patch_tokens: [N_patches, C]
    返回 (Hp, Wp) 的热力图
    """
    if metric == "l2":
        vec = torch.linalg.vector_norm(patch_tokens, ord=2, dim=-1)   # [N]
    else:
        vec = patch_tokens.mean(dim=-1)                               # [N]
    arr = vec.detach().cpu().numpy()
    fmap = arr.reshape(Hp, Wp)
    return fmap

# 组织子图布局
if num_blocks == 12:          # ViT-B
    n_rows, n_cols = 4, 3
elif num_blocks == 24:        # ViT-L
    n_rows, n_cols = 5, 5     # 多留一个空白
else:
    # 自动布局：尽量接近正方形
    n_cols = math.ceil(math.sqrt(num_blocks))
    n_rows = math.ceil(num_blocks / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3.5*n_rows))
axes = np.atleast_2d(axes)

# 逐层可视化
# hidden_states[1:] 对应 block_0 ... block_{L-1} 的输出
for idx, hs in enumerate(hidden_states[1:]):
    # hs: [B, 1 + R + N_patches, C]
    assert hs.dim() == 3 and hs.shape[0] == 1
    B, T, C = hs.shape
    # 剔除 CLS (1) + registers (R)
    start = 1 + num_register_tokens
    patch_tok = hs[0, start:, :]                                   # [N_patches, C]
    # 保险：确保 N_patches == Hp*Wp
    if patch_tok.shape[0] != Hp * Wp:
        # 若出现不一致，尽量用 sqrt 推断；但通常 HF 会对齐
        n = patch_tok.shape[0]
        hp = int(round(math.sqrt(n)))
        wp = n // hp
        if hp * wp != n:
            raise RuntimeError(f"Patch tokens count {n} is not factorizable into a grid.")
        fmap = tokens_to_map(patch_tok, METRIC).reshape(hp, wp)
    else:
        fmap = tokens_to_map(patch_tok, METRIC)

    r, c = divmod(idx, n_cols)
    ax = axes[r, c]
    im = ax.imshow(fmap, cmap=CMAP)
    ax.set_title(f"Block {idx}", fontsize=12)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 多余子图关掉坐标轴
total_axes = n_rows * n_cols
for k in range(num_blocks, total_axes):
    r, c = divmod(k, n_cols)
    axes[r, c].axis("off")

plt.suptitle(
    f"DINOv3 Feature Maps (model={MODEL_NAME}, size={H}x{W}, metric={METRIC})",
    fontsize=14
)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(SAVE_PATH, dpi=300)
print(f"Saved to: {os.path.abspath(SAVE_PATH)}")
