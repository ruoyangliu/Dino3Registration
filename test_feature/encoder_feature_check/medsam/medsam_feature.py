# -*- coding: utf-8 -*-
"""
SAM / MedSAM Feature Check (per-block patch maps)
- 读取 SAM 或 MedSAM 权重
- 抽取 image encoder 每个 Transformer block 的 tokens
- 生成 (64 x 64) 热力图；SAM 没有 CLS/register token
"""

import os, math, torch, numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry

# ===== 配置 =====
SAM_TYPE = "vit_b"  # vit_b / vit_l / vit_h（MedSAM 也对应这三类）
CKPT = r"G:\Dino3Registration\test_feature\encoder_feature_check\medsam\sam_vit_b_01ec64.pth"      # 可换成你的 MedSAM 权重路径
IMG_PATH = r"G:\Dino3Registration\data\sliced_png\Case6\MRI\coronal\slice_224.png"
SAVE_PATH = "sam_features_vitb.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METRIC = "l2"   # "l2" 或 "mean"

# ===== 加载 SAM/MedSAM =====
sam = sam_model_registry[SAM_TYPE](checkpoint=CKPT).to(DEVICE).eval()
enc = sam.image_encoder  # ImageEncoderViT

# ===== 预处理到 SAM 输入大小（长边 1024） =====
img_bgr = cv2.imread(IMG_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h0, w0 = img_rgb.shape[:2]
scale = 1024.0 / max(h0, w0)
img_resized = cv2.resize(img_rgb, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_LINEAR)

# SAM 内部会再做 padding 到 1024x1024
pad_h = 1024 - img_resized.shape[0]
pad_w = 1024 - img_resized.shape[1]
img_pad = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
img_pad = torch.as_tensor(img_pad, device=DEVICE).permute(2,0,1).contiguous()  # [3, H, W]
img_pad = img_pad[None]  # [1,3,H,W]
img_pad = (img_pad.float() - sam.pixel_mean.to(DEVICE)) / sam.pixel_std.to(DEVICE)

# ===== 注册 hook，抓每个 block 的输出 tokens =====
features = []
def hook_fn(module, x, y):
    # y: [B, HW, C]（SAM 没有 cls）
    features.append(y.detach())

hooks = []
for blk in enc.blocks:
    hooks.append(blk.register_forward_hook(hook_fn))

# ===== 前向：得到逐层 tokens =====
with torch.no_grad():
    _ = enc(img_pad)   # 仅跑 image encoder

for h in hooks:
    h.remove()

num_blocks = len(features)
Hp = Wp = 1024 // enc.patch_embed.patch_size  # 一般是 64
print(f"Blocks: {num_blocks}, grid: {Hp} x {Wp}")

# ===== 可视化每层热力图 =====
def to_map(tokens, metric="l2"):
    # tokens: [B, HW, C]
    t = tokens[0]            # [HW, C]
    if metric == "l2":
        v = torch.linalg.vector_norm(t, ord=2, dim=-1)   # [HW]
    else:
        v = t.mean(dim=-1)
    m = v.reshape(Hp, Wp).cpu().numpy()
    return m

n_cols = math.ceil(math.sqrt(num_blocks))
n_rows = math.ceil(num_blocks / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3.5*n_rows))
axes = np.atleast_2d(axes)

for i, feat in enumerate(features):
    r, c = divmod(i, n_cols)
    ax = axes[r, c]
    fmap = to_map(feat, METRIC)
    im = ax.imshow(fmap, cmap="viridis")
    ax.set_title(f"Block {i}", fontsize=10)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 关掉多余子图
for k in range(num_blocks, n_rows*n_cols):
    r, c = divmod(k, n_cols)
    axes[r, c].axis("off")

plt.suptitle(f"SAM/MedSAM Feature Maps (type={SAM_TYPE}, metric={METRIC})")
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig(SAVE_PATH, dpi=300)
print("Saved to:", os.path.abspath(SAVE_PATH))
