# import torch
# import torch.nn as nn
# import numpy as np
# import cv2
# import torch.nn.functional as F


# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
#                  bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU() if relu else None

#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x


# class ZPool(nn.Module):
#     def forward(self, x):
#         return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


# class AttentionGate(nn.Module):
#     def __init__(self):
#         super(AttentionGate, self).__init__()
#         kernel_size = 7
#         self.compress = ZPool()
#         self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.conv(x_compress)
#         scale = torch.sigmoid_(x_out)
#         return x * scale


# class TripletAttention(nn.Module):
#     def __init__(self, no_spatial=False):
#         super(TripletAttention, self).__init__()
#         self.cw = AttentionGate()
#         self.hc = AttentionGate()
#         self.no_spatial = no_spatial
#         if not no_spatial:
#             self.hw = AttentionGate()

#     def forward(self, x):
#         if not self.no_spatial:
#             x_out = 1 / 3 * (self.hw(x) + self.cw(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1,
#                                                                                               3).contiguous() + self.hc(
#                 x.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous())
#         else:
#             x_out = 1 / 2 * (self.cw(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous() + self.hc(
#                 x.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous())
#         return x_out


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)


# class CLSTM_cell(nn.Module):
#     """Initialize a basic Conv LSTM cell.
#     Args:
#       shape: int tuple thats the height and width of the hidden states h and c()
#       filter_size: int that is the height and width of the filters
#       num_features: int thats the num of channels of the states, like hidden_size

#     """

#     def __init__(self, input_chans, num_features, filter_size):
#         super(CLSTM_cell, self).__init__()
#         self.input_chans = input_chans
#         self.filter_size = filter_size
#         self.num_features = num_features
#         self.padding = (filter_size - 1) // 2
#         self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
#                               self.padding)

#     def forward(self, input, hidden_state):
#         hidden, c = hidden_state
#         combined = torch.cat((input, hidden), 1)
#         A = self.conv(combined)
#         (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)
#         i = torch.sigmoid(ai)
#         f = torch.sigmoid(af)
#         o = torch.sigmoid(ao)
#         g = torch.tanh(ag)

#         next_c = f * c + i * g
#         next_h = o * torch.tanh(next_c)
#         return next_h, next_c

#     def init_hidden(self, batch_size, shape):
#         return (torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda(),
#                 torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda())


# class res_block(nn.Module):
#     def __init__(self, ch_in):
#         super(res_block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_in),
#             nn.ReLU(inplace=True))
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_in),
#             nn.ReLU(inplace=True))

#     def forward(self, x):
#         y = x + self.conv(x)
#         return y + self.conv1(y)


# class conv_block(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(conv_block, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )
#         self.ta = TripletAttention()
#         self.res_block = res_block(ch_out)

#     def forward(self, x):
#         return self.ta(self.res_block(self.conv(x)))


# class conv_block_i(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(conv_block_i, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
#         )
#         self.ta = TripletAttention()
#         self.res_block = res_block(ch_out)

#     def forward(self, x):
#         return self.ta(self.res_block(self.conv(x)))


# class conv_block1(nn.Module):
#     def __init__(self, ch_in, ch_out, kernelsize=3):
#         super(conv_block1, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=kernelsize, stride=1, padding=int((kernelsize - 1) / 2), bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.conv(x)


# class conv_block_d(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(conv_block_d, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )
#         self.ta = TripletAttention()
#         self.res_block = res_block(ch_out)

#     def forward(self, x):
#         return self.ta(self.res_block(self.conv(x)))


# class conv_block_u(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(conv_block_u, self).__init__()
#         self.conv = nn.Sequential(
#             nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2, padding=0, bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )
#         self.ta = TripletAttention()
#         self.res_block = res_block(ch_out)

#     def forward(self, x):
#         return self.ta(self.res_block(self.conv(x)))


# class SqueezeAttentionBlock(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super(SqueezeAttentionBlock, self).__init__()
#         self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv = conv_block1(ch_in, ch_out)
#         self.conv_atten = CLSTM_cell(ch_in, ch_out, 5)
#         self.upsample = nn.Upsample(scale_factor=2)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, hidden_state):
#         x_res = self.conv(x)
#         y = self.avg_pool(x)
#         h, c = self.conv_atten(y, hidden_state)
#         y = self.upsample(h)
#         return self.sigmoid((y * x_res) + y) * 2 - 1, h, c


# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#     return (gauss / gauss.sum()).cuda()


# def gen_gaussian_kernel(window_size, sigma):
#     _1D_window = gaussian(window_size, sigma).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = torch.autograd.Variable(_2D_window.expand(1, 1, window_size, window_size).contiguous())
#     return window


# class GaussianBlurLayer(nn.Module):
#     def __init__(self, num_kernels=21, max_kernel_size=21, mode='TG', channels=1):
#         super(GaussianBlurLayer, self).__init__()
#         self.channels = channels
#         kernel_size = 3
#         weight = torch.zeros(num_kernels + 1, 1, max_kernel_size, max_kernel_size)
#         for i in range(num_kernels):
#             pad = int((max_kernel_size - kernel_size) / 2)
#             weight[i + 1] = (F.pad(gen_gaussian_kernel(kernel_size, sigma=0.25 * (i + 1)).cuda(),
#                                    [pad, pad, pad, pad])).squeeze(0)
#             if i >= 2 and i % 2 == 0 and kernel_size < max_kernel_size:
#                 kernel_size += 2
#         pad = int((max_kernel_size - 1) / 2)
#         weight[0] = (F.pad(torch.FloatTensor([[[[1.]]]]).cuda(),
#                            [pad, pad, pad, pad])).squeeze(0)

#         kernel = np.repeat(weight, self.channels, axis=0).cuda()
        
#         if mode == 'TG':
#             self.weight = kernel
#             self.weight.requires_grad = True
#         elif mode == 'TR':
#             self.weight = nn.Parameter(data=torch.randn(num_kernels * 3, 1, max_kernel_size, max_kernel_size),
#                                        requires_grad=True)
#         else:
#             self.weight = kernel
#             self.weight.requires_grad = False
#         self.padding = int((max_kernel_size - 1) / 2)

#     def __call__(self, x):
#         # temp = self.weight.detach().unsqueeze(1).cpu().numpy()
#         # for i in range(len(temp)//3):
#         #     cv2.imwrite('kernels1/TG/' + str(i) + '.png', temp[i, 0, 0] * 255. * 1)
#         # print(x.shape)
#         x = F.conv2d(x, self.weight, padding=self.padding, groups=self.channels)
#         return x


# class SumLayer(nn.Module):
#     def __init__(self, num_kernels=21, trainable=False):
#         super(SumLayer, self).__init__()
#         # self.conv = nn.Conv2d(2 * (num_kernels + 1) * 3, 1, 1)
#         self.conv = nn.Conv2d(2 * (num_kernels + 1), 1, 1)

#     def forward(self, x):
#         return self.conv(x)


# class MultiplyLayer1(nn.Module):
#     def __init__(self):
#         super(MultiplyLayer1, self).__init__()

#     def forward(self, x, y):
#         # print(x.shape)
#         # print(y.shape)
#         # return x * torch.cat([y, y, y], dim=1)
#         return x * y


# class MultiplyLayer(nn.Module):
#     def __init__(self):
#         super(MultiplyLayer, self).__init__()
#         self.ml = MultiplyLayer1()

#     def forward(self, x, y):
#         b, c, h, w = x.shape
#         b1, c1, h1, w1 = y.shape
#         return torch.cat([self.ml(x[:, :c // 2], y[:, :c1 // 2]), self.ml(x[:, c // 2:], y[:, c1 // 2:])], dim=1)


# if __name__ == '__main__':
#     ml = MultiplyLayer().cuda()

#############
##########

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
train_size = (1,3,256,256)

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5,4,3,2,1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )
           
    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2]*self.base_size[0]//train_size[-2]
            self.kernel_size[1] = x.shape[3]*self.base_size[1]//train_size[-1]
            
            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0]*x.shape[2]//train_size[-2])
            self.max_r2 = max(1, self.rs[0]*x.shape[3]//train_size[-1])

        if self.fast_imp:   # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0]>=h and self.kernel_size[1]>=w:
                out = F.adaptive_avg_pool2d(x,1)
            else:
                r1 = [r for r in self.rs if h%r==0][0]
                r2 = [r for r in self.rs if w%r==0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:,:,::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h-1, self.kernel_size[0]//r1), min(w-1, self.kernel_size[1]//r2)
                out = (s[:,:,:-k1,:-k2]-s[:,:,:-k1,k2:]-s[:,:,k1:,:-k2]+s[:,:,k1:,k2:])/(k1*k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1,r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum(dim=-2)
            s = torch.nn.functional.pad(s, (1,0,1,0)) # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])

            s1, s2, s3, s4 = s[:,:,:-k1,:-k2],s[:,:,:-k1,k2:], s[:,:,k1:,:-k2], s[:,:,k1:,k2:]
            out = s4+s1-s2-s3
            out = out / (k1*k2)
    
        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w)//2, (w - _w + 1)//2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')
        
        return out
    
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Gap(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()

        self.fscale_d = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.fscale_h = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        # self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.gap = AvgPool2d(base_size=80)

    def forward(self, x):
        x_d = self.gap(x)
        x_h = (x - x_d) * (self.fscale_h[None, :, None, None] + 1.)
        x_d = x_d  * self.fscale_d[None, :, None, None]
        return x_d + x_h


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.filter = filter

        self.dyna = dynamic_filter(in_channel//2) if filter else nn.Identity()
        self.dyna_2 = dynamic_filter(in_channel//2, kernel_size=5) if filter else nn.Identity()

        self.localap = Patch_ap(in_channel//2, patch_size=2)
        self.global_ap = Gap(in_channel//2)


    def forward(self, x):
        out = self.conv1(x)
       
        if self.filter:

            k3, k5 = torch.chunk(out, 2, dim=1)
            out_k3 = self.dyna(k3)
            out_k5 = self.dyna_2(k5)
            out = torch.cat((out_k3, out_k5), dim=1)
            
        non_local, local = torch.chunk(out, 2, dim=1)
        non_local = self.global_ap(non_local)
        local = self.localap(local)
        out = torch.cat((non_local, local), dim=1)
        out = self.conv2(out)
        return out + x

class Unet(nn.Module):
    def __init__(self, in_channel, out_channel, num_res):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_res-1):
            self.layers.append(ResBlock(in_channel, out_channel))
        self.layers.append(ResBlock(in_channel, out_channel, filter=True))
        self.down = nn.Conv2d(in_channel, in_channel, kernel_size=2, stride=2, groups=in_channel)
        self.num_res = num_res

        self.conv = nn.Conv2d(in_channel*2, in_channel, kernel_size=1, stride=1)
    def forward(self, x):
        res = x.clone()

        for i, layer in enumerate(self.layers):
            if i == self.num_res//4:
                skip = x
                x = self.down(x)
            if i == self.num_res - self.num_res//4:
                x = F.upsample(x, res.shape[2:], mode='bilinear')
                x = self.conv(torch.cat((x, skip), dim=1))
            x = layer(x)

        return x + res

class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        # self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        # self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(kernel_size//2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.modulate = SFconv(inchannels)

    def forward(self, x):
        identity_input = x # 3,32,64,64
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)     

        n, c, h, w = x.shape  
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
       
        low_filter = self.act(low_filter)
    
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)
        out_high = identity_input - low_part
        out = self.modulate(low_part, out_high)
        return out


class SFconv(nn.Module):
    def __init__(self, features, M=2, r=2, L=32) -> None:
        super().__init__()
        
        d = max(int(features/r), L)
        self.features = features

        self.fc = nn.Conv2d(features, d, 1, 1, 0)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, 1, 1, 0)
            )
        self.softmax = nn.Softmax(dim=1)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap = AvgPool2d(base_size=80)

        self.out = nn.Conv2d(features, features, 1, 1, 0)
        
    def forward(self, low, high):
        emerge = low + high
        emerge = self.gap(emerge)

        fea_z = self.fc(emerge)

        high_att = self.fcs[0](fea_z)
        low_att = self.fcs[1](fea_z)
        
        attention_vectors = torch.cat([high_att, low_att], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        high_att, low_att = torch.chunk(attention_vectors, 2, dim=1)

        fea_high = high * high_att
        fea_low = low * low_att
        
        out = self.out(fea_high + fea_low) 
        return out

class Patch_ap(nn.Module):
    def __init__(self, inchannel, patch_size):
        super(Patch_ap, self).__init__()

        # self.ap = nn.AdaptiveAvgPool2d((1,1))
        self.ap = AvgPool2d(base_size=80)

        self.patch_size = patch_size
        self.channel = inchannel * patch_size**2
        self.h = nn.Parameter(torch.zeros(self.channel))
        self.l = nn.Parameter(torch.zeros(self.channel))

    def forward(self, x):

        patch_x = rearrange(x, 'b c (p1 w1) (p2 w2) -> b c p1 w1 p2 w2', p1=self.patch_size, p2=self.patch_size)
        patch_x = rearrange(patch_x, ' b c p1 w1 p2 w2 -> b (c p1 p2) w1 w2', p1=self.patch_size, p2=self.patch_size)

        low = self.ap(patch_x)
        high = (patch_x - low) * self.h[None, :, None, None]
        out = high + low * self.l[None, :, None, None]
        out = rearrange(out, 'b (c p1 p2) w1 w2 -> b c (p1 w1) (p2 w2)', p1=self.patch_size, p2=self.patch_size)

        return out