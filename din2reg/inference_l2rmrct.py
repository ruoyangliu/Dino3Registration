import glob
import sys, os
import numpy as np
from sympy import use
import torch
from scipy.ndimage import zoom
from dinov2.eval.setup import build_model_for_eval
from dinov2.configs import load_and_merge_config
import torchvision.transforms as tt
import nibabel as nib
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from skimage.measure import label, regionprops
from skimage import morphology

import scipy.ndimage
from scipy.ndimage import map_coordinates
from scipy.ndimage import zoom

from convex_adam_utils import *
import sys
import einops
from utils.img_operations import remove_uniform_intensity_slices, reconstruct_image, to_lungCT_window, clip_and_normalize_image, pca_lowrank_transform, MR_normalize
from utils.img_operations import extract_lung_mask
from utils.convexAdam_3D import convex_adam_3d, convex_adam_3d_w0, convex_adam_3d_interSmooth, convex_adam_3d_param, convex_adam_3d_param_dataSmooth
from utils.data_utils import get_files_mrct
from scipy.ndimage import laplace, gaussian_filter
import utils.img_operations as img_op
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import cdist
from os import path
from PIL import Image
import matplotlib.pyplot as plt
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
if XFORMERS_ENABLED:
    from xformers.ops import memory_efficient_attention, unbind
import csv
import pandas as pd

from utils import img_operations
"""
FILE NOTE:

"""

class dinoReg:

    def __init__(self, device_id=0, lr=1, smooth_weight=10, num_iter =1000, feat_size=(80,80)):
        self.device_id = device_id
        self.patch_size = 16
        # self.transform = tt.Compose([tt.Normalize(mean=0.5, std=0.2)])
        #check
        self.transform = tt.Compose([])
        self.patch_grid_size = 32
        self.patch_margin = 10
        self.src_slice_num = 2
        self.patch_grid_h, self.patch_grid_w = 8, 8
        self.slice_step = 5
        #check  
        # self.img_size = (int(256 / self.patch_size+0.5) * self.patch_size * 3, int(192 / self.patch_size+0.5) * self.patch_size * 3)  #
        # self.img_size = (2080, 2560)  #set size
        # self.img_size = (1120, 1120)  #set size
        self.embed_dim = 384
        self.model = self.load_model()
        #use feature size to set image size(what shold I find the best size?)
        self.img_size = (self.patch_size*feat_size[0], self.patch_size*feat_size[1])  #set size
        # interation number
        self.num_iter = num_iter


        self.batch_size = 12 #todo: implement parallel?
        # self.reg_featureDim = 1
        #after PCA reduce to 24 dimension
        self.reg_featureDim = 24
        self.lr = lr
        self.smooth_weight = smooth_weight
        print('learning rate', self.lr)

        self.feature_height = self.img_size[0] // self.patch_size
        self.feature_width = self.img_size[1] // self.patch_size


    def extract_dinov2_feature(self, input_array):

        assert len(input_array.shape) == 3  # 2D image

        """flipping the input if needed"""
        # input_array = np.swapaxes(input_array, 0,1)

        input_rgb_array = input_array[np.newaxis, :, :, :]

        input_tensor = torch.Tensor(np.transpose(input_rgb_array, [0, 3, 1, 2]))
        input_tensor = self.transform(input_tensor)
        feature_array = self.model.forward_features(input_tensor.to(device=torch.device('cuda', self.device_id)))[
            'x_norm_patchtokens'].detach().cpu().numpy()
        del input_tensor

        return feature_array


    def case_inference(self, mov_arr, fix_arr, orig_img_shape, aff_mov,
                       mask_fixed=None, mask_moving=None, case_id='noID', disp_init=None, grid_sp_adam=1,DINOReg_useMask=True):

        assert len(mov_arr.shape) == 3

        """prepcocessing and feature extraction"""
        # mov_arr, fix_arr, slices_to_keep_indices, orig_chunked_shape, mask_fixed_arr, mask_moving_arr = self.case_preprocess(mov_arr, fix_arr, mask_fixed, mask_moving)
        mov_arr, fix_arr, slices_to_keep_indices, orig_chunked_shape, mask_fixed_arr, mask_moving_arr = self.case_preprocess(mov_arr, fix_arr)


        print('preprocessed moving and fixed image, shape', mov_arr.shape, fix_arr.shape)
        gap = 3 #3

        mov_feature = self.encode_3D_gap(mov_arr, gap=gap)
        print('encoded moving image')
        fix_feature = self.encode_3D_gap(fix_arr, gap=gap)
        print('encoded fixed image')

        feat_sliceNum = self.slice_num


        """PCA reduce dimension"""
        #only features inside the mask
        if DINOReg_useMask:
            # reshape to model output

            mask_fixed_arr = resize(mask_fixed_arr, (self.feature_height, self.feature_width, feat_sliceNum),
                                anti_aliasing=True)
            mask_moving_arr = resize(mask_moving_arr, (self.feature_height, self.feature_width, feat_sliceNum),
                                 anti_aliasing=True)
            mask_fixed_arr = np.where(mask_fixed_arr > 0.99, 1.0, 0)
            mask_moving_arr = np.where(mask_moving_arr > 0.99, 1.0, 0)
            # fixImg_1dim_threshold = nib.Nifti1Image(mask_fixed_arr, aff_mov)
            # nib.save(fixImg_1dim_threshold, os.path.join(output_dir, 'vis',case_list[i] + '_threshold.nii.gz'))

            # print('mask shape', mask_moving_arr.shape, mask_fixed_arr.shape)
            # print('feature  shape', mov_feature.shape, fix_feature.shape)
            mask_moving_arr = mask_moving_arr.flatten().astype(bool)
            mask_fixed_arr = mask_fixed_arr.flatten().astype(bool)
            mov_feature = mov_feature[mask_moving_arr, :]
            fix_feature = fix_feature[mask_fixed_arr, :]


        print('Starting PCA to reduce dimension')
        all_features = np.concatenate([mov_feature,fix_feature], axis=0)
        print('all features shape', all_features.shape, 'mask sum', mask_moving_arr.sum(), mask_fixed_arr.sum())
        pca_start_time = time.time()
        # object_pca = PCA(n_components=self.reg_featureDim) #what is SVD solver?
        # reduced_patches = object_pca.fit_transform(all_features)
        if configs['useSavedPCA']:
            reduced_patches = np.dot(all_features, PCA_matrix)
            eigenvalues = np.zeros(24)
        else:
            reduced_patches, eigenvalues = pca_lowrank_transform(all_features, self.reg_featureDim)
        print('PCA finished in {}, splitting features'.format(time.time()-pca_start_time))

        if DINOReg_useMask:
            mov_pca = np.zeros((self.feature_height * self.feature_width * feat_sliceNum, self.reg_featureDim), dtype='float32')
            fix_pca = np.zeros((self.feature_height * self.feature_width * feat_sliceNum, self.reg_featureDim), dtype='float32')
            mov_pca[mask_moving_arr, :] = reduced_patches[:mask_moving_arr.sum(), :]
            fix_pca[mask_fixed_arr, :] = reduced_patches[mask_moving_arr.sum():, :]
            mov_pca = mov_pca.reshape([self.feature_height, self.feature_width, feat_sliceNum, -1])
            fix_pca = fix_pca.reshape([self.feature_height, self.feature_width, feat_sliceNum, -1])
        else:

            mov_pca = reduced_patches[:feat_sliceNum * self.feature_height * self.feature_width, :]
            fix_pca = reduced_patches[feat_sliceNum * self.feature_height * self.feature_width:, :]
            mov_pca = mov_pca.reshape([self.feature_height, self.feature_width, feat_sliceNum, -1])
            fix_pca = fix_pca.reshape([self.feature_height, self.feature_width, feat_sliceNum, -1])

        eigenvalue_array.append(eigenvalues[:24])




        print('reshaping to original image shape')
        mov_pca_rescaled = resize(mov_pca, (orig_chunked_shape[0], orig_chunked_shape[1], orig_chunked_shape[2], self.reg_featureDim),
                                   anti_aliasing=True)
        fix_pca_rescaled = resize(fix_pca, (orig_chunked_shape[0], orig_chunked_shape[1], orig_chunked_shape[2], self.reg_featureDim),
                                   anti_aliasing=True)


        #plug in the slices to keep, the rest are 0
        mov_fullImg_pca_rescaled = np.zeros((orig_img_shape[0], orig_img_shape[1], orig_img_shape[2], self.reg_featureDim),
                                          dtype='float32')
        fix_fullImg_pca_rescaled = np.zeros((orig_img_shape[0], orig_img_shape[1], orig_img_shape[2], self.reg_featureDim),
                                          dtype='float32')

        mov_fullImg_pca_rescaled[:, :, slices_to_keep_indices, :] = mov_pca_rescaled
        fix_fullImg_pca_rescaled[:, :, slices_to_keep_indices, :] = fix_pca_rescaled

        """save copy of 1 channel feature for vis"""
        # for channel in range(3):
        #     mov_feat_1dim = mov_fullImg_pca_rescaled[:,:,:,channel:channel+3]
        #     fix_feat_1dim = fix_fullImg_pca_rescaled[:,:,:,channel:channel+3]
        #     movImg_1dim = nib.Nifti1Image(mov_feat_1dim, aff_mov)
        #     fixImg_1dim = nib.Nifti1Image(fix_feat_1dim, aff_mov)
        #     os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)
        #     nib.save(movImg_1dim, os.path.join(output_dir, 'vis', case_id + '_mov_{}.nii.gz'.format(channel)))
        #     nib.save(fixImg_1dim, os.path.join(output_dir, 'vis', case_id + '_fix_{}.nii.gz'.format(channel)))
        #
        # sys.exit()

        # mov_feat_1dim = mov_fullImg_pca_rescaled[:,:,:,:3]
        # fix_feat_1dim = fix_fullImg_pca_rescaled[:,:,:,:3]
        # movImg_1dim = nib.Nifti1Image(mov_feat_1dim, aff_mov)
        # fixImg_1dim = nib.Nifti1Image(fix_feat_1dim, aff_mov)
        # nib.save(movImg_1dim, os.path.join(output_dir, 'vis' + case_list[i] + '_mov_feat_24dim.nii.gz'))
        # nib.save(fixImg_1dim, os.path.join(output_dir, 'vis' + case_list[i] + '_fix_feat_24dim.nii.gz'))
        # sys.exit()

        if save_feature:
            os.makedirs(os.path.join(output_dir_0, 'features'), exist_ok=True)
            np.save(os.path.join(output_dir_0, 'features', case_id + '_mov_feat.npy'), mov_fullImg_pca_rescaled)
            np.save(os.path.join(output_dir_0, 'features', case_id + '_fix_feat.npy'), fix_fullImg_pca_rescaled)

        """ConvexAdam optimization"""
        print('starting ConvexAdam optimization')

        # disp = convex_adam_3d(fix_fullImg_pca_rescaled, mov_fullImg_pca_rescaled,
        # disp = convex_adam_3d_interSmooth(fix_fullImg_pca_rescaled, mov_fullImg_pca_rescaled, #default 1000 iter
        #                       loss_func = "SSD", selected_niter=self.num_iter, lr=self.lr, selected_smooth=20, ic=True, lambda_weight=self.smooth_weight, disp_init=disp_init)
                              # loss_func = "SSD", selected_niter=5000, lr=self.lr, selected_smooth=3, ic=True, lambda_weight=self.smooth_weight, disp_init=disp_init)
                              # loss_func = "SSD", selected_niter=1000, lr=1, selected_smooth=3, ic=True, lambda_weight=10, disp_init=disp_init)

        disp = convex_adam_3d_param(fix_fullImg_pca_rescaled, mov_fullImg_pca_rescaled, loss_func = "SSD", grid_sp_adam=grid_sp_adam,
                                               lambda_weight=configs['smooth_weight'], selected_niter=configs['num_iter'], lr=configs['lr'], disp_init=disp_init,
                                                iter_smooth_kernel = configs['iter_smooth_kernel'],
                                                iter_smooth_num = configs['iter_smooth_num'], end_smooth_kernel=1,final_upsample=configs['final_upsample'])
        

        """apply displacement field to moving image or landmarks"""

        """save copy of 1 channel feature for vis"""
        # mov_feat_1dim = mov_fullImg_pca_rescaled[:,:,:,:3]
        # fix_feat_1dim = fix_fullImg_pca_rescaled[:,:,:,:3]
        # movImg_1dim = nib.Nifti1Image(mov_feat_1dim, aff_mov)
        # fixImg_1dim = nib.Nifti1Image(fix_feat_1dim, aff_mov)
        # os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)
        # nib.save(movImg_1dim, os.path.join(output_dir, 'vis', case_id + '_mov_feat_gap3.nii.gz'))
        # nib.save(fixImg_1dim, os.path.join(output_dir, 'vis',case_id + '_fix_feat_gap3.nii.gz'))
        # sys.exit()

        return disp

        

    def case_preprocess(self, mov_arr, fix_arr):
        assert len(mov_arr.shape) == 3
        assert len(fix_arr.shape) == 3

        pad_indices = []
        filtered_image_data, slices_to_keep_indices = remove_uniform_intensity_slices(fix_arr)
        pad_indices.append(slices_to_keep_indices)
        fix_arr = filtered_image_data
        mov_arr = mov_arr[:, :, slices_to_keep_indices]

        orig_chunked_shape = fix_arr.shape


        #old preprop
        fix_arr = MR_normalize(fix_arr)
        mov_arr = to_lungCT_window(mov_arr, wl=50, ww=400)
        mask_fixed = np.where(fix_arr > 0.05, 1.0, 0)
        mask_moving = np.where(mov_arr > 0.005, 1.0, 0)


        filtered_z = fix_arr.shape[2]


        mask_fixed = np.zeros_like(fix_arr)
        mask_moving = np.zeros_like(mov_arr)
        for slice_idx in range(fix_arr.shape[2]):
            mask_fixed[:, :, slice_idx] = extract_lung_mask(fix_arr[:, :, slice_idx], threshold_value=0.05)
            mask_moving[:, :, slice_idx] = extract_lung_mask(mov_arr[:, :, slice_idx], threshold_value=0.005)


        #reshape to model input
        # fix_arr = resize(fix_arr, (self.img_size[0], self.img_size[1], fix_arr.shape[2]), anti_aliasing=True)
        # mov_arr = resize(mov_arr, (self.img_size[0], self.img_size[1], mov_arr.shape[2]), anti_aliasing=True)

        """save copy of mask for vis"""
        # movImg_1dim = nib.Nifti1Image(mask_moving, aff_mov)
        # fixImg_1dim = nib.Nifti1Image(mask_fixed, aff_mov)
        # os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)
        # nib.save(movImg_1dim, os.path.join(output_dir, 'vis', 'mov_mask.nii.gz'))
        # nib.save(fixImg_1dim, os.path.join(output_dir, 'vis', 'fix_mask.nii.gz'))

        # movImg_1dim = nib.Nifti1Image(mov_arr, aff_mov)
        # fixImg_1dim = nib.Nifti1Image(fix_arr, aff_mov)
        # os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)
        # nib.save(movImg_1dim, os.path.join(output_dir, 'vis', 'mov_proc.nii.gz'))
        # nib.save(fixImg_1dim, os.path.join(output_dir, 'vis', 'fix_proc.nii.gz'))
        # sys.exit()


        return mov_arr, fix_arr, slices_to_keep_indices, orig_chunked_shape , mask_fixed, mask_moving

    def load_model(self):
        """load model"""


        conf_fn = '{0:s}/dinov2/configs/eval/vitl14_reg4_pretrain'.format(sys.path[0])
        model_fn = 'models/dinov2/dinov2_vitl14_reg4_pretrain.pth'
        model_url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth'
        self.patch_size = 14
        self.embed_dim = 1024


        # Check if model file exists, download if not
        if not os.path.exists(model_fn):
            import urllib.request
            os.makedirs(os.path.dirname(model_fn), exist_ok=True)
            print(f"Downloading model from {model_url} to {model_fn}...")
            urllib.request.urlretrieve(model_url, model_fn)
            print("Download complete.")
        else:
            print("DINOv2 model found.")


        conf = load_and_merge_config(conf_fn)
        model = build_model_for_eval(conf, model_fn)
        model.to(device=torch.device('cuda', self.device_id))
        return model

    def encode_3D_gap(self, input_arr, gap=3):


        imageH, imageW, slice_num = input_arr.shape

        """old resize"""
        # feature_height = int(imageH * upsample_factor + 0.5)  // self.patch_size
        # feature_width = int(imageW * upsample_factor + 0.5) // self.patch_size
        # self.slice_num = slice_num
        # self.feature_height = feature_height
        # self.feature_width = feature_width

        """new uniform resize"""
        feature_height = self.feature_height
        feature_width = self.feature_width
        self.slice_num = slice_num


        input_arr = resize(input_arr, (feature_height*self.patch_size, feature_width*self.patch_size, slice_num), anti_aliasing=True)

        print(self.patch_size)
        print(feature_height, feature_width, slice_num)
        print('resized input shape', input_arr.shape)

        # 3D image into 2D model, stack each slices feature
        img_feature = np.zeros([feature_height * feature_width, slice_num, self.embed_dim])
        encoding_slice_idx = np.arange(0, slice_num-1, gap).tolist()
        encoding_slice_idx.append(slice_num-1)

        prev_slice = 0
        for slice_id in encoding_slice_idx:
            input_slice = input_arr[:, :, slice_id, np.newaxis]
            input_slice = np.repeat(input_slice, 3, axis=2)
            featrure = self.extract_dinov2_feature(input_slice)
            featrure = einops.rearrange(featrure, '1 n c -> n c')
            print("\rslice id:{} feature shape:{} ".format(slice_id, featrure.shape), end="")
            img_feature[:, slice_id, :] = featrure

            #interpolating the feature of the skipped slices
            if slice_id > 0 and slice_id < slice_num-1:
                for i in range(1, gap):
                    slice_id_gap = slice_id - i
                    if slice_id_gap >= 0:
                        featrure_gap = (featrure * (gap - i) + img_feature[:, prev_slice, :] * i) / gap
                        img_feature[:, slice_id_gap, :] = featrure_gap
            elif slice_id == slice_num-1:
                last_gap = slice_num - encoding_slice_idx[-2]
                for i in range(1, last_gap):
                    slice_id_gap = slice_num - i
                    featrure_gap = (featrure * (last_gap - i) + img_feature[:, prev_slice, :] * i) / last_gap
                    img_feature[:, slice_id_gap, :] = featrure_gap
            prev_slice = slice_id

        img_feature = img_feature.reshape([feature_height * feature_width * slice_num, self.embed_dim])


        return img_feature


    def extract_slice_feature(self, input_arr_orig, mask=True):

        """input single slice 2d, output the feature of that slice"""

        input_arr = resize(input_arr_orig, (self.feature_height*self.patch_size, self.feature_width*self.patch_size), anti_aliasing=True)
        if mask:
            input_arr_masksize = resize(input_arr_orig, (self.feature_height, self.feature_width), anti_aliasing=True)
            pca_mask = extract_lung_mask(input_arr_masksize).flatten().astype(bool)

        input_slice = input_arr[:, :, np.newaxis]
        input_slice = np.repeat(input_slice, 3, axis=2)
        featrure = self.extract_dinov2_feature(input_slice)

        featrure = einops.rearrange(featrure, '1 n c -> n c')

        if mask:
            return featrure, pca_mask
        return featrure, np.ones(featrure.shape[0], dtype=bool)
    

def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape

    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)

    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (
                jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) - \
             jacobian[1, 0, :, :, :] * (
                         jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :,
                                                                                                       :, :]) + \
             jacobian[2, 0, :, :, :] * (
                         jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :,
                                                                                                       :, :])

    return jacdet


def compute_95_hausdorff_distance(seg1, seg2):
    # Assuming seg1 and seg2 are binary segmentation masks
    u_indices = np.array(np.where(seg1)).T
    v_indices = np.array(np.where(seg2)).T

    # Compute all pairwise distances between the two sets of points
    distances = cdist(u_indices, v_indices, 'euclidean')

    # Flatten the distance matrix and sort the distances
    sorted_distances = np.sort(distances, axis=None)

    # Find the 95th percentile distance
    hd_95 = np.percentile(sorted_distances, 95)
    return hd_95


def compute_label_wise_95hd(seg1, seg2, labels):
    hd95_results = []
    for label in labels:
        # Isolate current label in both segmentations
        seg1_label = seg1 == label
        seg2_label = seg2 == label

        # Compute 95% HD for the current label
        hd95 = compute_95_hausdorff_distance(seg1_label, seg2_label)
        hd95_results.append(hd95)

    return hd95_results

def score_case(seg_fixed, seg_moving, disp_field, fixed_arr, case, label_list, spacing=1):

    print('disp shape in score_case',disp_field.shape)
    # disp_field = np.array([zoom(disp_field[i], 2, order=2) for i in range(3)])

    jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(0.000000001, 1000000000)
    log_jac_det = np.log(jac_det)

    # disp = einops.rearrange(disp_field, 'h d w c -> c h d w')
    # dice_coefficient = img_op.apply_deformation_and_compute_dice(seg_fixed, seg_moving, disp, fixed_arr, case, num_classes=35)
    dice_coefficient = img_op.apply_deformation_and_compute_dice(seg_fixed, seg_moving, disp, fixed_arr, case, num_classes=label_list)
    dice_coefficient = np.array(dice_coefficient)

    hd95 = img_op.warp_compute_label_wise_95hd(seg_fixed, seg_moving, [1, 2, 3, 4], disp)
    hd95= np.array(hd95)

    return {'DICE': dice_coefficient,
            'LogJacDetStd': log_jac_det[2:-2, 2:-2, 2:-2].std(),
            'HD95': hd95}


if __name__ == '__main__':

    time_start = time.time()

    save_feature = False    

    configs = {
        'smooth_weight' : 2, #50
        'lr' : 3,
        'num_iter' : 1000,
        'fm_downsample' : 1,
        'feature_size' : (112,96),
        # 'feature_size' : (80,70),
        # 'feature_size' : (150,129),
        'useSavedPCA' : False,
        'DINOReg_useMask' : True,
        'window' : True,
        'convex' : False,
        'ztrans' : False,
        'iter_smooth_num': 5,
        'iter_smooth_kernel': 7,
        'final_upsample': 1,
        'mask': 'slice fill stack'
    }


    output_dir_0 = f'output/dinoreg-{configs["smooth_weight"]}smooth-{configs["num_iter"]}iter-itersmoothK{configs["iter_smooth_kernel"]}R{configs["iter_smooth_num"]}-lr3-fmd1-fmsize112x96-noconvex' #
    # output_dir_0 = f'/fast/songx/dinoReg_benchmark/methods/l2rmrct/dinoreg-{configs["smooth_weight"]}smooth-500iter-itersmoothK5R3-lr3-fmd1-fmsize150x129-simpleThresh-noconvex' #
    
    print('output_dir_0', output_dir_0)

    dataset_dir = 'sample_dataset_dir'

    if configs['useSavedPCA']:
        PCA_matrix = np.load('/sample_dir/pca_matrix_AMOS_150x129_mask.npy')


    os.makedirs(output_dir_0, exist_ok=True)

    #save config as json
    import json
    with open(os.path.join(output_dir_0, 'configs.json'), 'w') as f:
        json.dump(configs, f)
    
    # Initialize an empty list to hold the rows
    pair_list = []
    # Read the CSV file
    with open(path.join(dataset_dir, 'pairs_Tr.csv'), 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            pair_list.append(row)


    quantify = True

    dinoReg = dinoReg(lr=configs['lr'], smooth_weight=configs['smooth_weight'], num_iter=configs['iter_smooth_num'], feat_size=configs['feature_size'])

    exp_note = 'l2rmrct_eval'

    print('exp_note', exp_note)

    output_dir = os.path.join(output_dir_0, exp_note)
    os.makedirs(output_dir, exist_ok=True)

    DICE_list = []
    LogJacDetStd_list = []
    hd_list = []
    eigenvalue_array = []

    print('exp_note', exp_note)

    # Load labels from CSV file
    csv_file = os.path.join(dataset_dir, 'structures.csv')
    df = pd.read_csv(csv_file, header=None)
    label_list = df.iloc[0, :].tolist()

    for i, pair in enumerate(pair_list):
        print('case', i)

        moving_fn = pair[0] #template is to be applied to other cases
        fixed_fn = pair[1]

        fixed_basename = os.path.basename(fixed_fn)
        fixed_basename = fixed_basename.split('.')[0]
        moving_basename = os.path.basename(moving_fn)
        moving_basename = moving_basename.split('.')[0]

        img_fixed = nib.load(path.join(dataset_dir, 'img', fixed_fn))
        img_moving = nib.load(path.join(dataset_dir, 'img', moving_fn))

        arr_fixed = img_fixed.get_fdata()
        arr_moving = img_moving.get_fdata()

        aff_mov = img_moving.affine

        seg_fixed = nib.load(path.join(dataset_dir, 'seg', fixed_fn)).get_fdata()
        seg_moving = nib.load(path.join(dataset_dir, 'seg', moving_fn)).get_fdata()


        H,W,D = arr_moving.shape
        identity = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D)).permute(0,4,1,2,3)
        disp_init = identity.numpy()

        disp_init = None
        load_note = None
        # load_note = 'cls_select'
        if load_note is not None:
            disp_init = nib.load(os.path.join(output_dir_0, load_note, i + '_disp_{}.nii.gz'.format(load_note))).get_fdata()
            disp_init = np.moveaxis(disp_init, 3, 0)[np.newaxis, :, :, :, :]

        disp = dinoReg.case_inference(arr_moving, arr_fixed, arr_moving.shape, aff_mov, case_id=fixed_basename, 
                                      disp_init=disp_init, grid_sp_adam=configs['fm_downsample'], DINOReg_useMask=configs['DINOReg_useMask'])
        # disp = np.zeros_like(arr_moving)
        # disp = np.stack([disp, disp, disp], axis=3)

        #save disp
        disp_img = nib.Nifti1Image(disp, aff_mov)
        # nib.save(disp_img, os.path.join(output_dir, '{}_disp_{}.nii.gz'.format(fixed_basename, exp_note)))
        nib.save(disp_img, os.path.join(output_dir, '{}_to_{}_disp_{}.nii.gz'.format(
            moving_basename, fixed_basename, exp_note)))

        disp = np.moveaxis(disp, 3, 0)

        #warp moving image
        D, H, W = arr_moving.shape
        identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
        warped_image = map_coordinates(arr_moving, identity + disp, order=0)
        warped_image = nib.Nifti1Image(warped_image, aff_mov)
        # nib.save(warped_image, os.path.join(output_dir, '{}_warped_{}.nii.gz'.format(fixed_basename, exp_note)))
        nib.save(warped_image, os.path.join(output_dir, '{}_to_{}_warped_{}.nii.gz'.format(
            moving_basename, fixed_basename, exp_note)))
        if quantify:
            #normalize arr_fixed
            arr_fixed_norm = to_lungCT_window(arr_fixed)

            result = score_case(seg_fixed, seg_moving, disp, arr_fixed_norm, i, label_list)

            DICE_list.append(result['DICE'])
            LogJacDetStd_list.append(result['LogJacDetStd'])
            hd_list.append(result['HD95'])
            print('DICE:', result['DICE'])
            print('LogJacDetStd', result['LogJacDetStd'])
            print('HD95', result['HD95'])

    if quantify:

        temp_dice = np.asarray(DICE_list)
        print('temp_dice shape', temp_dice.shape)
        temp_dice[0, -1] = np.nan
        temp_dice[1, -1] = np.nan
        mean_value = np.nanmean(temp_dice)


        #print mean and std
        print('DICE mean', np.nanmean(temp_dice), 'DICE std', np.nanstd(temp_dice) )
        print('LogJacDetStd mean', np.mean(np.asarray(LogJacDetStd_list)), 'LogJacDetStd std', np.std(np.asarray(LogJacDetStd_list)))

        print(DICE_list)
        print(LogJacDetStd_list)

        np.savetxt(os.path.join(output_dir, 'summary_{}.txt'.format(exp_note)), np.array([np.nanmean(temp_dice), np.nanstd(temp_dice),
                                                                                          np.mean(np.asarray(LogJacDetStd_list)),np.std(np.asarray(LogJacDetStd_list))]), fmt='%.4f')

        np.savetxt(os.path.join(output_dir, 'DICE_{}.txt'.format(exp_note)), np.asarray(DICE_list), fmt='%.4f')
        np.savetxt(os.path.join(output_dir, 'LogJacDetStd_list_{}.txt'.format(exp_note)), np.asarray(LogJacDetStd_list), fmt='%.3f')
        np.savetxt(os.path.join(output_dir, 'HD95_{}.txt'.format(exp_note)), np.asarray(hd_list), fmt='%.4f')

        # Convert list of arrays into a single 2D numpy array
        # eigenvalue_array_2d = np.vstack(eigenvalue_array)
        # np.savetxt(os.path.join(output_dir, 'eigenvalue_list_{}.txt'.format(exp_note)), eigenvalue_array_2d, fmt='%.5f')



    print('time elapsed', time.time() - time_start, 'exp_note', exp_note)
    print(output_dir_0)

