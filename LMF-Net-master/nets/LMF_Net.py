import os
import torch
import torch.nn as nn
import numpy as np
import skimage
import PIL.Image
import torch.nn.functional as f
import torchvision.transforms as transforms
from skimage import morphology
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_closing, disk, remove_small_objects
from scipy.ndimage import median_filter
from nets.CMEA import ImprovedCMEA


def extract_luminance(img):
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0

    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    return Y


def enforce_single_transition(dm, white_is_bottom=True, smooth_ks=31):
    dm = (dm > 0).astype(np.uint8)
    H, W = dm.shape
    white_cum = np.cumsum(dm, axis=0).astype(np.int32)
    total_white = white_cum[-1, :]
    black_cum = np.cumsum(1 - dm, axis=0).astype(np.int32)
    total_black = black_cum[-1, :]

    if white_is_bottom:
        cost = white_cum + (total_black[np.newaxis, :] - black_cum)
    else:
        cost = black_cum + (total_white[np.newaxis, :] - white_cum)

    y_opt = np.argmin(cost, axis=0).astype(np.int32)
    if smooth_ks is not None and smooth_ks >= 3 and smooth_ks % 2 == 1:
        y_opt = median_filter(y_opt, size=smooth_ks).astype(np.int32)

    rows = np.arange(H)[:, None]
    if white_is_bottom:
        dm_fix = (rows > y_opt[None, :]).astype(np.uint8)
    else:
        dm_fix = (rows <= y_opt[None, :]).astype(np.uint8)
    return dm_fix, y_opt


class LMFNet():

    def __init__(self):
        self.device = "cuda:0"
        self.model = LMFNetModel()

        self.mean_value = 0.521705004231865
        self.std_value = 0.2203818810958583

        self.data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([self.mean_value], [self.std_value])
        ])

        self.kernel_radius = 5
        self.area_ratio = 0.01
        self.ks = 5
        self.gf_radius = 4
        self.eps = 0.1

        self.band_width = 150
        self.uncert_tau = 0.2
        self.white_bottom = True
        self.y_smooth_ks = 31



    def fuse(self, img1, img2):
        if img1.ndim == 2:
            img1_lum, img2_lum = img1, img2
        else:
            img1_lum = extract_luminance(img1)
            img2_lum = extract_luminance(img2)

        img1_pil = PIL.Image.fromarray(img1_lum)
        img2_pil = PIL.Image.fromarray(img2_lum)
        x1 = self.data_transforms(img1_pil).unsqueeze(0).to(self.device)
        x2 = self.data_transforms(img2_pil).unsqueeze(0).to(self.device)

        dm_raw, W = self.model.forward("fuse", x1, x2, kernel_radius=self.kernel_radius)
        h, w = img1.shape[:2]


        se = skimage.morphology.disk(self.ks)
        dm = skimage.morphology.binary_opening(dm_raw, se)
        dm = morphology.remove_small_holes(dm == 0, self.area_ratio * h * w)
        dm = np.where(dm, 0, 1)
        dm = skimage.morphology.binary_closing(dm, se)
        dm = morphology.remove_small_holes(dm == 1, self.area_ratio * h * w)
        dm = np.where(dm, 1, 0).astype(np.uint8)

        dm_fix, y_line = enforce_single_transition(
            dm, white_is_bottom=self.white_bottom, smooth_ks=self.y_smooth_ks
        )

        bw = self.band_width
        upper = np.clip(y_line - bw, 0, h - 1)
        lower = np.clip(y_line + bw, 0, h - 1)

        band_core = np.zeros((h, w), np.uint8)
        for j in range(w):
            band_core[upper[j]: lower[j] + 1, j] = 1

        U = 1.0 - np.abs(W - 0.6) * 2.0
        cand = (band_core.astype(bool) & (U > self.uncert_tau)).astype(np.uint8)
        if cand.sum() == 0:
            band_Tm = band_core.copy()
        else:

            lab = label(cand, connectivity=2)
            if lab.max() >= 1:
                regions = regionprops(lab)
                largest = max(regions, key=lambda r: r.area)
                band_Tm = (lab == largest.label).astype(np.uint8)
            else:
                band_Tm = cand

        for _ in range(2):
            band_Tm = binary_dilation(band_Tm, footprint=disk(3)).astype(np.uint8)

        band_Tm = binary_closing(band_Tm, footprint=disk(5)).astype(np.uint8)


        lab2 = label(band_Tm, connectivity=2)
        if lab2.max() >= 1:
            regions2 = regionprops(lab2)
            largest2 = max(regions2, key=lambda r: r.area)
            band_Tm = (lab2 == largest2.label).astype(np.uint8)

        band_Tm = remove_small_objects(band_Tm.astype(bool), min_size=int(0.0001 * h * w)).astype(np.uint8)

        dm3 = dm_fix[..., None] if img1.ndim == 3 else dm_fix
        fused = img1 * dm3 + img2 * (1 - dm3)
        fused = np.clip(fused, 0, 255).astype(np.uint8)

        return fused, band_Tm


class LMFNetModel(nn.Module):

    def __init__(self):
        super(LMFNetModel, self).__init__()
        self.features = self.conv_block(in_channels=1, out_channels=16)
        self.conv_encode_1 = self.conv_block(16, 16)
        self.conv_encode_2 = self.conv_block(32, 16)
        self.conv_encode_3 = self.conv_block(48, 16)

        self.se_f = ImprovedCMEA(16)
        self.se_1 = ImprovedCMEA(16)
        self.se_2 = ImprovedCMEA(16)
        self.se_3 = ImprovedCMEA(16)

        self.conv_decode_1 = self.conv_block(64, 64)
        self.conv_decode_2 = self.conv_block(64, 32)
        self.conv_decode_3 = self.conv_block(32, 16)
        self.conv_decode_4 = self.conv_block(16, 1)

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    @staticmethod
    def concat(f1, f2):
        return torch.cat((f1, f2), 1)

    def forward(self, phase, img1, img2=None, kernel_radius=5):
        if phase == 'train':
            features = self.features(img1)
            se_features = self.se_f(features)

            encode_block1 = self.conv_encode_1(se_features)
            se_encode_block1 = self.se_1(encode_block1)

            se_cat1 = self.concat(se_features, se_encode_block1)
            encode_block2 = self.conv_encode_2(se_cat1)
            se_encode_block2 = self.se_2(encode_block2)

            se_cat2 = self.concat(se_cat1, se_encode_block2)
            encode_block3 = self.conv_encode_3(se_cat2)

            se_encode_block3 = self.se_3(encode_block3)
            se_cat3 = self.concat(se_cat2, se_encode_block3)

            decode_block1 = self.conv_decode_1(se_cat3)
            decode_block2 = self.conv_decode_2(decode_block1)
            decode_block3 = self.conv_decode_3(decode_block2)
            output = self.conv_decode_4(decode_block3)
            return output
        elif phase == 'fuse':
            with torch.no_grad():
                features_1 = self.features(img1)
                features_2 = self.features(img2)
                se_features_1 = self.se_f(features_1)
                se_features_2 = self.se_f(features_2)

                encode_block1_1 = self.conv_encode_1(se_features_1)
                encode_block1_2 = self.conv_encode_1(se_features_2)
                se_encode_block1_1 = self.se_1(encode_block1_1)
                se_encode_block1_2 = self.se_1(encode_block1_2)

                se_cat1_1 = self.concat(se_features_1, se_encode_block1_1)
                se_cat1_2 = self.concat(se_features_2, se_encode_block1_2)
                encode_block2_1 = self.conv_encode_2(se_cat1_1)
                encode_block2_2 = self.conv_encode_2(se_cat1_2)
                se_encode_block2_1 = self.se_2(encode_block2_1)
                se_encode_block2_2 = self.se_2(encode_block2_2)
                se_cat2_1 = self.concat(se_cat1_1, se_encode_block2_1)
                se_cat2_2 = self.concat(se_cat1_2, se_encode_block2_2)
                encode_block3_1 = self.conv_encode_3(se_cat2_1)
                encode_block3_2 = self.conv_encode_3(se_cat2_2)
                se_encode_block3_1 = self.se_3(encode_block3_1)
                se_encode_block3_2 = self.se_3(encode_block3_2)
                se_cat3_1 = self.concat(se_cat2_1, se_encode_block3_1)
                se_cat3_2 = self.concat(se_cat2_2, se_encode_block3_2)

            output, W = self.fusion_channel_sf(se_cat3_1, se_cat3_2, kernel_radius=kernel_radius)

            return output, W

    @staticmethod
    def fusion_channel_sf(f1, f2, kernel_radius=5, eps=1e-6):
        device = f1.device
        b, c, h, w = f1.shape

        laplacian_kernel = torch.FloatTensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) \
            .cuda(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)

        f1_laplacian = f.conv2d(f1, laplacian_kernel, padding=1, groups=c)
        f2_laplacian = f.conv2d(f2, laplacian_kernel, padding=1, groups=c)

        f1_second_grad = torch.pow((f1_laplacian - f1), 2)
        f2_second_grad = torch.pow((f2_laplacian - f2), 2)

        kernel_size = kernel_radius * 2 + 1
        add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float().cuda(device)
        kernel_padding = kernel_size // 2

        f1_sf = torch.sum(f.conv2d(f1_second_grad, add_kernel, padding=kernel_padding, groups=c), dim=1)
        f2_sf = torch.sum(f.conv2d(f2_second_grad, add_kernel, padding=kernel_padding, groups=c), dim=1)

        weight_zeros = torch.zeros(f1_sf.shape).cuda(device)
        weight_ones = torch.ones(f1_sf.shape).cuda(device)

        contrast_diff = torch.abs(f1_sf - f2_sf)
        contrast_diff = contrast_diff / (f1_sf + f2_sf + eps)
        contrast_diff = contrast_diff.clamp(0, 1)

        W1 = f1_sf / (f1_sf + f2_sf + eps)
        W2 = f2_sf / (f1_sf + f2_sf + eps)
        W = W1 + W2 / 2
        W = W.clamp(0, 1)

        dm_tensor = torch.where(f1_sf > f2_sf, weight_ones, weight_zeros).cuda(device)
        dm_np = dm_tensor.squeeze().cpu().numpy().astype(int)
        W = W.squeeze().cpu().numpy()

        W1 = W1.squeeze().cpu().numpy()
        contrast_diff = contrast_diff.squeeze().cpu().numpy()

        return dm_np, W1
