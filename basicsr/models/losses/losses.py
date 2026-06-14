import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)

class FFTLoss(nn.Module):
    """L1 loss in frequency domain with FFT.

    Args:
        loss_weight (float): Loss weight for FFT loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (..., C, H, W). Predicted tensor.
            target (Tensor): of shape (..., C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (..., C, H, W). Element-wise
                weights. Default: None.
        """

        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        return self.loss_weight * l1_loss(pred_fft, target_fft, weight, reduction=self.reduction)

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # return self.loss_weight * l1_loss(
        #     pred, target, weight, reduction=self.reduction)
        if isinstance(pred, list):
            loss = 0.
            for predi in pred:
                loss += self.loss_weight * l1_loss(predi, target, weight, reduction=self.reduction)
            return loss / len(pred)
        else:
            return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss




from torchvision import models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.ab = ablation
        self.mlpA = nn.Sequential(nn.Conv2d(512, 512, 1),
                                  nn.GELU(),
                                  nn.Conv2d(512, 512, 1)).cuda()
        self.mlpP = nn.Sequential(nn.Conv2d(512, 512, 1),
                                  nn.GELU(),
                                  nn.Conv2d(512, 512, 1)).cuda()
        self.mlpN = nn.Sequential(nn.Conv2d(512, 512, 1),
                                  nn.GELU(),
                                  nn.Conv2d(512, 512, 1)).cuda()

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            if i != 4:
                a, p, n = a_vgg[i], p_vgg[i], n_vgg[i]
            else :
                a, p, n = self.mlpA(a_vgg[i]), self.mlpP(p_vgg[i]), self.mlpN(n_vgg[i])
            d_ap = self.l1(a, p.detach())
            if not self.ab:
                d_an = self.l1(a, n.detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss
    
    
    
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
    

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        # self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels = img.shape[1]
        kernel = self.kernel.expand(n_channels, -1, -1, -1)  # [C, 1, 5, 5]
        # print(kernel.shape)
        
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        
        return F.conv2d(img, kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss
    
    
    
class FreqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)

    def forward(self, pred, target):
        if isinstance(pred, list):
            loss = 0.
            for predi in pred:
                diff = torch.fft.rfft2(predi) - torch.fft.rfft2(target)
                loss_freq = torch.mean(torch.abs(diff))
                loss += self.loss_weight * (loss_freq * 0.01 + self.l1_loss(predi, target))
            return loss / len(pred)
        else:
            diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
            loss = torch.mean(torch.abs(diff))
            # print(loss)
            return self.loss_weight * (loss * 0.01 + self.l1_loss(pred, target))

from basicsr.models.losses.dual_model_mae import mae_vit_base_patch16     
from basicsr.models.losses.contrast_loss import selfPerceptualLoss
from torchvision.transforms.functional import normalize
from basicsr.utils.pos_embed import interpolate_pos_embed, interpolate_pos_encoding
import os.path

class ReconstructPerceptualLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ReconstructPerceptualLoss, self).__init__()
        self.l1_loss = L1Loss(reduction='mean')
        # self.mse_loss = MSELoss(reduction='mean')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        img_size = 256
        pretrained_ckpt = '/data/dsq/Restormer1/Defocus_Deblurring/pretrained_models/mae_pretrain_vit_base.pth'
        self.pretrain_mae = mae_vit_base_patch16(img_size=img_size).to(self.device)
        pretrained_ckpt = os.path.expanduser(pretrained_ckpt)
        # checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        checkpoint = torch.load(pretrained_ckpt, map_location=self.device)
        print("Load pre-trained checkpoint from: %s" % pretrained_ckpt)
        checkpoint_model = checkpoint['model']
        state_dict = self.pretrain_mae.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # interpolate position embedding
        interpolate_pos_embed(self.pretrain_mae, checkpoint_model)
        self.pretrain_mae.load_state_dict(checkpoint_model, strict=False)
        for _, p in self.pretrain_mae.named_parameters():
            p.requires_grad = False
        self.constrastive_loss = selfPerceptualLoss(img_size)
        # self.constrastive_loss = projectedDistributionLoss()

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]

    def forward(self, recover_img, gt):
        losses = {}
        loss_l1 = self.l1_loss(recover_img, gt)

        recover_img = normalize(recover_img, self.normalize_mean, self.normalize_std)
        gt = normalize(gt, self.normalize_mean, self.normalize_std)
        predict_embed, gt_embed, ids = self.pretrain_mae(recover_img, gt, 0.50)
        # Local MAE
        contrast_loss = self.constrastive_loss(predict_embed, gt_embed, ids) * 0.01

        # Global MAE
        # contrast_loss =0
        # for predict_e, gt_e in zip(predict_embed, gt_embed):
        #    contrast_loss += projectedDistributionLoss(predict_e, gt_e) * 1e-4

        # losses["l1"] = loss_l1
        # losses["Perceptual"] = contrast_loss
        # losses["total_loss"] = contrast_loss + loss_l1 
        
        losses = contrast_loss + loss_l1 


        return losses
    
from pytorch_wavelets import DWTForward
from torchvision import transforms
import torchvision

class FCL(nn.Module):
    """FreqContrastiveLoss"""
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.DWT = DWTForward(J=1, wave='haar', mode='reflect')
        self.cl_loss_type = 'l1'
        self.loss = torch.nn.L1Loss()
        self.reblur = torchvision.transforms.GaussianBlur(15, sigma=20.)

        self.loss_weight = loss_weight

    def forward(self, network, blur, GT, out, **kwargs):
        rebl = self.reblur(blur)
        p_list = [GT]
        n_list = [blur, rebl]

        with torch.no_grad():
            pdwt_list = []
            for p in p_list:
                pdwt = self.DWT(p)
                pdwt_list.append(pdwt)  # both high freq & low freq component
            ndwt_list = []
            for n in n_list:
                ndwt = self.DWT(n)
                ndwt_list.append(ndwt[1][0])  # high freq component
            adwt = self.DWT(out)  # both high freq & low freq component
            pos_loss = self.cl_pos(adwt, pdwt_list)
            neg_loss = self.cl_neg(adwt, ndwt_list)
        loss = self.cl_loss(pos_loss, neg_loss) * self.loss_weight
        return loss

    def cl_pos(self, a, p_list):
        pos_loss = 0
        for p in p_list:
            pos_loss += self.loss(a[0], p[0])  # low freq
            pos_loss += self.loss(a[1][0], p[1][0])  # high freq
        pos_loss /= len(p_list)
        return pos_loss

    def cl_neg(self, a, n_list):
        neg_loss = 0
        for n in n_list:
            neg_loss += self.loss(a[1][0], n)
        neg_loss /= len(n_list)
        return neg_loss

    def cl_loss(self, pos_loss, neg_loss):
        # minimize posloss, maximize negloss
        # print(pos_loss)
        # print(neg_loss)
        if self.cl_loss_type in ['l2', 'cosine']:
            cl_loss = pos_loss - neg_loss

        elif self.cl_loss_type == 'l1':
            cl_loss = pos_loss / (neg_loss + 3e-7)
            # print(cl_loss)
        else:
            raise TypeError(f'{self.args.cl_loss_type} not fount in cl_loss')

        return cl_loss
    
class FRACL(FCL):
    """FreqResidualAugContrastiveLoss"""
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.DWT = DWTForward(J=1, wave='haar', mode='reflect')
        self.cl_loss_type = 'l1'
        self.loss = torch.nn.L1Loss()
        self.reblur = transforms.GaussianBlur(15, sigma=20.)
        self.aug = transforms.Compose([
            transforms.RandomChoice([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                                     transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                                     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                                     transforms.ColorJitter(0.6, 0.6, 0.6, 0.1)],
                                    p=[0.25, 0.25, 0.25, 0.25]),
            transforms.RandomGrayscale(p=0.2)
        ])  # only use aug in CL? or use aug in trainer too?
        self.aug_len = 5

    def forward(self, network, C, GT, anchor):
        rebl = self.reblur(C)
        p_list = [GT]
        n_list = [C, rebl, *[self.aug(rebl) for _ in range(self.aug_len)]]

        with torch.no_grad():
            pdwt_list = []
            for p in p_list:
                pdwt = self.DWT(p)
                pdwt_list.append(pdwt)  # both high freq & low freq component
            ndwt_list = []
            for n in n_list:
                ndwt = self.DWT(n)
                ndwt_list.append(ndwt[1][0])  # high freq component
            adwt = self.DWT(anchor)  # both high freq & low freq component
            pos_loss = self.cl_pos(adwt, pdwt_list)
            neg_loss = self.cl_neg(adwt, ndwt_list)
            negres_loss = self.cl_res(adwt, ndwt_list)

        # print(pos_loss)
        # print(neg_loss)
        # print(negres_loss)

        # print("pos_loss:", pos_loss.shape, type(pos_loss))
        # print("neg_loss:", neg_loss.shape, type(neg_loss))
        # print("negres_loss:", negres_loss.shape, type(negres_loss))

        # print(self.loss_weight)
        # print("self.loss_weight:", type(self.loss_weight))
        loss = self.cl_loss(pos_loss, neg_loss + negres_loss) * self.loss_weight
        return loss

    def cl_res(self, a, n_list):  # not use positive sample, only push away from neg
        """
        n_list: [C, rebl, rebl_aug1, rebl_aug2, ...]
        """
        c = n_list[0]
        negres_loss = 0
        res_c_reb = []
        for n in n_list[1:]:
            res_c_reb.append(c - n)  # c - reb
        res_c_reb = torch.mean(torch.stack(res_c_reb, dim=0), dim=0).squeeze(0)
        for n in n_list[1:]:
            res_a_reb = a[1][0] - n  # a - reb
            negres_loss += self.loss(res_c_reb, res_a_reb)
        negres_loss /= len(n_list[1:])
        return negres_loss
    
import math
class DCTConvLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean',k=1.0):
        super().__init__()
        assert reduction in ['mean', 'sum', 'none']
        self.loss_weight = loss_weight
        self.reduction = reduction

        self.l1_loss = L1Loss(1.0, reduction)
        self.to_k = k

    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        # cos((x + 0.5) / N * n * \pi) which is also the form of DCT and IDCT
        # DCT: F(n) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * f(x) )
        # IDCT: f(x) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * F(n) )
        # returns: (Res_n, Res_x)
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight
    
    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        # exp(-[(n\pi/a)^2 + (m\pi/b)^2])
        # returns: (Res_h, Res_w)
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight

    def forward(self, pred, target):
        B, C, H, W = pred.shape

        loss = self.l1_loss(pred, target)

        pred = pred.permute(0, 2, 3, 1).contiguous()
        target = target.permute(0, 2, 3, 1).contiguous()

        x = pred

        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (getattr(self, "__WEIGHT_COSN__", None).device == x.device):
            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
            assert weight_cosn is not None
            assert weight_cosm is not None
            assert weight_exp is not None
        else:
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)
        

        N, M = weight_cosn.shape[0], weight_cosm.shape[0]
        
        pred_dct = F.conv1d(pred.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N, H, 1))
        pred_dct = F.conv1d(pred_dct.contiguous().view(-1, W, C), weight_cosm.contiguous().view(M, W, 1)).contiguous().view(B, N, M, -1)

        target_dct = F.conv1d(target.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N, H, 1))
        target_dct = F.conv1d(target_dct.contiguous().view(-1, W, C), weight_cosm.contiguous().view(M, W, 1)).contiguous().view(B, N, M, -1)

        weight_exp = torch.pow(weight_exp[:, :, None], self.to_k)

        # Compute frequency domain loss

        pred_dct = torch.einsum("bnmc,nmc -> bnmc", pred_dct, weight_exp) # exp decay
        target_dct = torch.einsum("bnmc,nmc -> bnmc", target_dct, weight_exp) # exp decay

        pred_dct = pred_dct.permute(0, 3, 1, 2).contiguous()
        target_dct = target_dct.permute(0, 3, 1, 2).contiguous()

        # print(pred_dct.shape)
        # print(target_dct.shape)
        loss_dct = l1_loss(pred_dct, target_dct)
        # print(loss_dct)
        
        return self.loss_weight * loss_dct + loss