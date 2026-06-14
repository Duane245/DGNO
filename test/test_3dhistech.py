"""
DGNO — test on 3DHistech (RGB) cytopathology defocus deblurring.

Usage:
    python test/test_3dhistech.py \
        --config  options/DGNO_3DHistech_Face.yml \
        --weights pretrained/DGNO_3DHistech_Face.pth \
        --input_dir ./datasets/3DHistech/test \
        --device  cuda:0 [--save_images]

The test set folder must contain `blur/` and `sharp/` sub-folders of `.png` images.
"""

import os
import time
import argparse
from glob import glob

import numpy as np
import cv2
import yaml
import torch
import lpips
from natsort import natsorted

import utils
from basicsr.models.archs.dgno_eval_arch import DGNO_eval

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(path):
    """Resolve a path relative to the repo root when it is not found as given."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(REPO_ROOT, path)


def main():
    parser = argparse.ArgumentParser(description='DGNO test on 3DHistech (RGB)')
    parser.add_argument('--config', default='options/DGNO_3DHistech_Face.yml', type=str,
                        help='DGNO yml config (decides network params)')
    parser.add_argument('--weights', default='pretrained/DGNO_3DHistech_Face.pth', type=str,
                        help='Path to the checkpoint .pth')
    parser.add_argument('--input_dir', default='./datasets/3DHistech/test', type=str,
                        help='Test folder containing blur/ and sharp/ sub-folders of .png images')
    parser.add_argument('--result_dir', default='./results/DGNO_3DHistech', type=str,
                        help='Directory to save restored images')
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--save_images', action='store_true', help='Save restored images')
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- load config ----
    cfg = yaml.load(open(resolve(args.config), mode='r'), Loader=Loader)
    net_opt = dict(cfg['network_g'])
    net_opt.pop('type', None)   # `variant` field stays in net_opt and is read by DGNO_eval

    # ---- build model (DGNO_eval handles both variants via the `variant` field) ----
    model = DGNO_eval(**net_opt)
    checkpoint = torch.load(resolve(args.weights), map_location='cpu')
    model.load_state_dict(checkpoint['params'])
    model = model.to(device).eval()
    print('===> Testing with weights:', args.weights)

    alex = lpips.LPIPS(net='alex').to(device)

    if args.save_images:
        os.makedirs(args.result_dir, exist_ok=True)

    blur_files = natsorted(glob(os.path.join(args.input_dir, 'blur', '*.png')))
    sharp_files = natsorted(glob(os.path.join(args.input_dir, 'sharp', '*.png')))
    assert len(blur_files) == len(sharp_files), \
        f'blur/sharp count mismatch: {len(blur_files)} vs {len(sharp_files)}'
    print(f'===> {len(blur_files)} test images')

    psnr, ssim, mae, pips = [], [], [], []
    start = time.time()
    with torch.no_grad():
        for i, (fb, fc) in enumerate(zip(blur_files, sharp_files)):
            img_i = np.float32(utils.load_img(fb)) / 255.
            img_c = np.float32(utils.load_img(fc)) / 255.

            patch_i = torch.from_numpy(img_i).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            patch_c = torch.from_numpy(img_c).unsqueeze(0).permute(0, 3, 1, 2).to(device)

            restored = model(patch_i)
            restored = torch.clamp(restored[-1], 0, 1)
            pips.append(alex(patch_c, restored, normalize=True).item())

            restored_np = restored.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            psnr.append(utils.psnr_m(img_c, restored_np))
            ssim.append(utils.SSIM(img_c, restored_np))
            mae.append(utils.MAE(img_c, restored_np))
            print('i {:4d}  PSNR {:.4f}  SSIM {:.4f}  MAE {:.4f}  LPIPS {:.4f}'.format(
                i, psnr[i], ssim[i], mae[i], pips[i]))

            if args.save_images:
                out = np.uint8((restored_np * 255).round())
                utils.save_img(os.path.join(args.result_dir, os.path.basename(fc)), out)

    print('Elapsed: {:.2f}s'.format(time.time() - start))
    print('Overall: PSNR {:.4f}  SSIM {:.4f}  MAE {:.4f}  LPIPS {:.4f}'.format(
        np.mean(psnr), np.mean(ssim), np.mean(mae), np.mean(pips)))


if __name__ == '__main__':
    main()
