"""
DGNO — test on BBBC006 (w1 / w2), grayscale microscopy defocus deblurring.

Usage:
    python test/test_bbbc.py \
        --config  options/DGNO_BBBC006_w1_Face.yml \
        --weights pretrained/DGNO_BBBC006_w1_Face.pth \
        --input_dir ./datasets/BBBC006/test \
        --device  cuda:0 [--save_images]

The test set folder must contain `blur/` and `GT/` sub-folders of `.tif` images.
The channel (w1 / w2) is read from the yml field `datasets.val.BBBCw`.
"""

import os
import time
import argparse
from pathlib import Path

import numpy as np
import cv2
import yaml
import torch
import torch.nn as nn
import lpips

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
    parser = argparse.ArgumentParser(description='DGNO test on BBBC006 (w1/w2)')
    parser.add_argument('--config', default='options/DGNO_BBBC006_w1_Face.yml', type=str,
                        help='DGNO yml config (decides network params and BBBC channel)')
    parser.add_argument('--weights', default='pretrained/DGNO_BBBC006_w1_Face.pth', type=str,
                        help='Path to the checkpoint .pth')
    parser.add_argument('--input_dir', default='./datasets/BBBC006/test', type=str,
                        help='Test folder containing blur/ and GT/ sub-folders of .tif images')
    parser.add_argument('--result_dir', default='./results/DGNO_BBBC006', type=str,
                        help='Directory to save restored images')
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--save_images', action='store_true', help='Save restored images')
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- load config ----
    cfg = yaml.load(open(resolve(args.config), mode='r'), Loader=Loader)
    net_opt = dict(cfg['network_g'])
    net_opt.pop('type', None)   # `variant` field stays in net_opt and is read by DGNO_eval
    bbbc_w = cfg['datasets']['val'].get('BBBCw', 'w1')

    # ---- build model (DGNO_eval handles both variants via the `variant` field) ----
    model = DGNO_eval(**net_opt)
    checkpoint = torch.load(resolve(args.weights), map_location='cpu')
    model.load_state_dict(checkpoint['params'])
    model = model.to(device).eval()
    print('===> Testing with weights:', args.weights, '| BBBC channel:', bbbc_w)

    alex = lpips.LPIPS(net='alex').to(device)

    if args.save_images:
        os.makedirs(args.result_dir, exist_ok=True)

    # ---- collect tif files filtered by channel suffix (..._w1 / ..._w2) ----
    blur_dir = Path(os.path.join(args.input_dir, 'blur'))
    gt_dir = Path(os.path.join(args.input_dir, 'GT'))

    def keep(p):
        return bbbc_w == 'w1w2' or p.parts[-1].split('_')[-1][0:2] == bbbc_w

    blur_files = [str(p) for p in sorted(blur_dir.rglob('*.tif')) if keep(p)]
    gt_files = []
    for p in sorted(gt_dir.rglob('*.tif')):
        if keep(p):
            gt_files += [str(p)] * 3  # z-stack length = 3
    assert len(blur_files) == len(gt_files), \
        f'blur/GT count mismatch: {len(blur_files)} vs {len(gt_files)}'
    print(f'===> {len(blur_files)} test images')

    psnr, ssim, pips = [], [], []
    start = time.time()
    with torch.no_grad():
        for i, (fb, fc) in enumerate(zip(blur_files, gt_files)):
            img_i = cv2.imread(fb, cv2.IMREAD_UNCHANGED).astype(np.float32)
            img_i -= img_i.min()
            img_i = img_i / max(img_i.max(), 1e-8)
            img_c = cv2.imread(fc, cv2.IMREAD_UNCHANGED).astype(np.float32)
            img_c -= img_c.min()
            img_c = img_c / max(img_c.max(), 1e-8)

            patch_i = torch.FloatTensor(img_i[None, None]).to(device)
            patch_c = torch.FloatTensor(img_c[None, None]).to(device)

            restored = model(patch_i)
            restored = torch.clamp(restored[-1], 0, 1)
            pips.append(alex(patch_c, restored, normalize=True).item())

            restored_np = restored.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            gt_np = patch_c.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()

            psnr.append(utils.PSNR(gt_np, restored_np))
            ssim.append(utils.SSIM(gt_np, restored_np))
            print('i {:4d}  PSNR {:.4f}  SSIM {:.4f}  LPIPS {:.4f}'.format(
                i, psnr[i], ssim[i], pips[i]))

            if args.save_images:
                out = np.uint8((restored_np * 255).round())
                cv2.imwrite(os.path.join(args.result_dir, os.path.basename(fb)), out)

    print('Elapsed: {:.2f}s'.format(time.time() - start))
    print('Overall: PSNR {:.4f}  SSIM {:.4f}  LPIPS {:.4f}'.format(
        np.mean(psnr), np.mean(ssim), np.mean(pips)))


if __name__ == '__main__':
    main()
