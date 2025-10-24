"""
('ViT-B-32', 'openai'), 
('ViT-B-32', 'laion400m_e31'), 
('ViT-B-32', 'laion400m_e32'), 
('ViT-B-32', 'laion2b_e16'), 
('ViT-B-32', 'laion2b_s34b_b79k'), 

('ViT-B-16', 'openai'), 
('ViT-B-16', 'laion400m_e31'), 
('ViT-B-16', 'laion400m_e32'), 
('ViT-B-16', 'laion2b_s34b_b88k'), 

('ViT-L-14', 'openai'), 
('ViT-L-14', 'laion400m_e31'), 
('ViT-L-14', 'laion400m_e32'), 
('ViT-L-14', 'laion2b_s32b_b82k'), 
('ViT-L-14-336', 'openai'), 

('ViT-H-14', 'laion2b_s32b_b79k'), 
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from open_clip.transform import image_transform

# model_name      = "ViT-L-14"
# pretrained_name = "laion2b_s32b_b82k"
# patch_size = 14

# model_name      = "ViT-L-14-336"
# pretrained_name = "openai"
# patch_size = 14

# model_name      = "ViT-B-16"
# pretrained_name = "openai"
# patch_size = 16

model_name      = "ViT-B-16"
pretrained_name = "laion2b_s34b_b88k"
patch_size = 16

image_size = 1024 # 512 or 224, default is 224

# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_name)
image_mean = getattr(model.visual, 'image_mean', None)
image_std  = getattr(model.visual, 'image_std', None)
preprocess = image_transform(
    image_size,
    is_train=False,
    mean=image_mean,
    std=image_std,
)

if image_size == 512 or 1024: # we need to upsample the positional embeddings
    positional_embedding = model.visual.positional_embedding.clone()
    cls_pos = positional_embedding[0:1, :]
    spatial_size = image_size//patch_size
    spatial_pos = F.interpolate(positional_embedding[1:,].reshape(1, 14, 14, 768).permute(0, 3, 1, 2),
                                size=(spatial_size, spatial_size), 
                                mode='bilinear')
    spatial_pos = spatial_pos.reshape(768, spatial_size*spatial_size).permute(1, 0)
    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
    model.visual.positional_embedding = nn.Parameter(positional_embedding)

# 用于存储特征的字典
features = {}

# 定义钩子函数来提取输入特征
def get_transformer_input_features(module, input, block_index):
    features[f'block_{block_index}_input'] = input[0][1:].detach()

# 注册钩子到每个 Transformer Block
for idx, block in enumerate(model.visual.transformer.resblocks):
    # 使用 lambda 函数来确保只传递必要的参数
    block.register_forward_pre_hook(lambda module, input, idx=idx: get_transformer_input_features(module, input, idx))

image = preprocess(Image.open("bird.png")).unsqueeze(0)
print("image shape: {}".format(image.shape))

# 使用模型进行前向传播
with torch.no_grad():
    _ = model.encode_image(image)

# 打印每个 Transformer Block 的输入特征
for block, feature in features.items():
    print(f'{block}: {feature.shape}')

if len(features) == 12: # B
    n_cols = 3
    n_rows = 4
    fig_interval = 3
if len(features) == 24: # L
    n_cols = 5
    n_rows = 5
    fig_interval = 5

# 创建一个大画布来展示当前 Block 的所有特征图
fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 30))
# fig.suptitle(f'Features of {block_name}', fontsize=16)

counter = 0
for block_name, block_features in features.items():
    patch_num = block_features.shape[0]
    h = int(np.sqrt(patch_num))
    feature_map = block_features.squeeze(1).mean(dim=-1)  # 取均值，shape: (196,)
    feature_map = feature_map.reshape(h, h).numpy()  # reshape 为 (14, 14)

    ax = axes[counter//fig_interval, counter%fig_interval]
    im = ax.imshow(feature_map, cmap='viridis')  # 可视化为热力图
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f'Block {counter}', fontsize=32)
    ax.axis('off')

    counter+=1

if len(features) == 24: # L
    aa = axes[4, 4]
    aa.axis('off')

# 调整布局并展示当前 Block 的特征图
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig('all_features_visualization_{}_{}_image_size_1024.png'.format(model_name, pretrained_name), dpi=300)

    

