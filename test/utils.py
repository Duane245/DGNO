## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import cv2
import math

from skimage import metrics
from sklearn.metrics import mean_absolute_error

def MAE(img1, img2):
    mae_0=mean_absolute_error(img1[:,:,0], img2[:,:,0],
                              multioutput='uniform_average')
    mae_1=mean_absolute_error(img1[:,:,1], img2[:,:,1],
                              multioutput='uniform_average')
    mae_2=mean_absolute_error(img1[:,:,2], img2[:,:,2],
                              multioutput='uniform_average')
    return np.mean([mae_0,mae_1,mae_2])

def PSNR(img1, img2):
    mse_ = np.mean( (img1 - img2) ** 2 )
    if mse_ == 0:
        return 100
    return 10 * math.log10(1 / mse_)

import skimage
import skimage.metrics
def psnr_m(ref, dist, multi_channel=True):

    # ref_rgb = np.uint8((ref*255).round())
    # img_source = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR)

    # dist_rgb = np.uint8((dist*255).round())
    # img_target = cv2.cvtColor(dist_rgb, cv2.COLOR_RGB2BGR)

    # img_source_gray = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
    # img_target_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

    ref_gray = cv2.cvtColor((ref * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    dist_gray = cv2.cvtColor((dist * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # return skimage.metrics.peak_signal_noise_ratio(img_source_gray, img_target_gray)
    return skimage.metrics.peak_signal_noise_ratio(ref_gray, dist_gray)

def SSIM(img1, img2):
    # return metrics.structural_similarity(img1, img2, data_range=1, multichannel=True)
    return metrics.structural_similarity(img1, img2, data_range=1, channel_axis=-1)

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def load_img16(filepath):
    return cv2.cvtColor(cv2.imread(filepath, -1), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
