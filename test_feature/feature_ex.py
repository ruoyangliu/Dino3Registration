# import torch

# REPO_DIR = r"/home/rl23/Desktop/code/Dinov3FeatureExtration /dinov3"  # 你本地克隆的 dinov3 仓库
# W_S = r"/home/rl23/Desktop/code/Dinov3FeatureExtration /dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"  # 你的 .pth 权重

# enc = torch.hub.load(
#     REPO_DIR, 'dinov3_vits16', source='local', weights=W_S
# ).eval().cuda()

# # 然后用你已有的 preprocess / encode_patch_and_cls / PCA / 注册优化即可
from transformers import pipeline
from transformers.image_utils import load_image
import matplotlib.pyplot as plt
import numpy as np
import os

url = "/home/rl23/Desktop/code/Dinov3FeatureExtration /dinov3/test_feature/US_image/Case1-US-before.jpeg"
image = load_image(url)
print("Image loaded:", image.size)
feature_extractor = pipeline(
    model="facebook/dinov3-vitl16-pretrain-lvd1689m",
    task="image-feature-extraction", 
)
#print(feature_extractor.__class__)
print(feature_extractor.image_processor.__class__)
features = feature_extractor(image,return_tensors=True)




#从模型路径获取模型名称
model_name = feature_extractor.model.name_or_path.split('/')[-1]
save_path = '/home/rl23/Desktop/code/Dinov3FeatureExtration /dinov3/test_feature/US_features/' + model_name + '.npy'

# 确保父目录存在
os.makedirs(os.path.dirname(save_path), exist_ok=True)

#np.save(save_path, features_np)
print('Features saved to', save_path)

import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

# url = "/home/rl23/Desktop/code/Dinov3FeatureExtration /dinov3/test_feature/US_image/Case1-US-before.jpeg"
# image = load_image(url)

# pretrained_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
# processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
# model = AutoModel.from_pretrained(
#     pretrained_model_name, 
#     device_map="auto", 
# )

# inputs = processor(images=image, return_tensors="pt").to(model.device)
# with torch.inference_mode():
#     outputs = model(**inputs)

# pooled_output = outputs.pooler_output
# print("Pooled output shape:", pooled_output.shape)
# model_name = pretrained_model_name
# save_path = '/home/rl23/Desktop/code/Dinov3FeatureExtration /dinov3/test_feature/features_facebook/' + model_name + 'pool.npy'

# # 确保父目录存在
# os.makedirs(os.path.dirname(save_path), exist_ok=True)

# np.save(save_path, pooled_output.cpu().numpy())
# print('Features saved to', save_path)