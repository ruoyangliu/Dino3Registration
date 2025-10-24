import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ===== 加载 feature =====
feat = np.load("G:\\Dino3Registration\\test_feature\\encoderTest\\pipelen_encoder_features_case6_mri_slice224.npy")    # shape (1, 201, 1024)
feat = feat.squeeze(0)           # (201, 1024)

# ===== 1️⃣ 去掉 CLS 与 register token =====
cls_token = feat[0:1, :]
patch_tokens = feat[1:197, :]
register_tokens = feat[197:, :]

# ===== 2️⃣ 重塑为二维特征图 =====
grid_h, grid_w = 14, 14
feat_map = patch_tokens.reshape(grid_h, grid_w, 1024)

# ===== 3️⃣ PCA 降维到 3D（用于可视化）=====
flat = feat_map.reshape(-1, 1024)
pca = PCA(n_components=3)
feat_pca = pca.fit_transform(flat)
feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min())
feat_rgb = feat_pca.reshape(grid_h, grid_w, 3)
print(f"[info] PCA 降维后形状: {feat_rgb.shape}")


# ===== 4️⃣ 可视化 =====
plt.figure(figsize=(6,6))
plt.imshow(feat_rgb)
plt.title("DINOv3 Feature Map (PCA RGB)")
plt.axis('off')
plt.show()

print("前3主成分方差占比:", pca.explained_variance_ratio_,
      "\n累计:", pca.explained_variance_ratio_.sum())

# ===== 保存可视化结果 =====
plt.imsave("G:\\Dino3Registration\\test_feature\\encoderTest\\pipelen_encoder_features_case6_mri_slice224_pca_rgb.png", feat_rgb)