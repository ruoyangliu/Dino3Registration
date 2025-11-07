from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import torch, numpy as np

model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
image_path="G:\\Dino3Registration\\data\\upsample_data\\Case6\\MRI\\slice_108_resampled.png"

# 1) 加载模型（等价于原文 build_model_for_eval）
model = AutoModel.from_pretrained(model_name).eval()

# 2) 不使用 pipeline 的自动预处理；若你仍加载了 processor，务必关掉所有自动项
proc = AutoImageProcessor.from_pretrained(model_name)
for k in ["do_resize","do_center_crop","do_rescale","do_normalize"]:
    if hasattr(proc, k): setattr(proc, k, False)

# === 你“先手动上采样”到你想要的输入分辨率（这一步对应原文 Upsample(s)）===
img = Image.open(image_path).convert("RGB")  # 这里 img.size 就是你上采样后的尺寸

# 手动归一化（原文就是手写的，不靠处理器）
x = torch.tensor(np.array(img)).float() / 255.0
x = x.permute(2,0,1).unsqueeze(0)
mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
x = (x - mean) / std

with torch.no_grad():
    out = model(pixel_values=x, output_hidden_states=True)

# 从 hidden_states 取出 patch tokens → 还原为 (Hp, Wp, C)
# （Hp = H_in/patch_size, Wp = W_in/patch_size；动态计算，不写死）
ps = getattr(model.config, "patch_size", getattr(getattr(model.config,"vision_config",None),"patch_size",16))
H, W = img.size[1], img.size[0]
Hp, Wp = H // ps, W // ps
n_patch = Hp * Wp
last = out.hidden_states[-1]
patch_tokens = last[:, -n_patch:, :]          # 去掉 CLS/REG
feat = patch_tokens.reshape(1, Hp, Wp, last.shape[-1])  # (1,Hp,Wp,1024 for ViT-L)
