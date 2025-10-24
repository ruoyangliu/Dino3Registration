from transformers import pipeline
from transformers.image_utils import load_image

url = "G:\\Dino3Registration\\data\\sliced_png\\Case6\\MRI\\coronal\\slice_224.png"
image = load_image(url)
#print image size
print(f"[info] image size: {image.size}")
feature_extractor = pipeline(
    model="facebook/dinov3-vitl16-pretrain-lvd1689m",
    task="image-feature-extraction", 
)
features = feature_extractor(image)
import numpy as np
# Convert features to numpy array and print its shape
features_np = np.array(features)
print(features_np.shape)

#save the features to a npy file
np.save("G:\\Dino3Registration\\test_feature\\encoderTest\\pipelen_encoder_features_case6_mri_slice224.npy", features_np)