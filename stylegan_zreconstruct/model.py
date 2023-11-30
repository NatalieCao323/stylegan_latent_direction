import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

from math import sqrt

import random
import numpy as np

def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


## 归一化权重
class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        ## 输入神经元数目,每一层卷积核数量=Nin*Nout*K*K,
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


## 转置卷积上采样，其中权重参数自己定义
class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size ##神经元数量
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


## 卷积与下采样，其中权重参数随机定义
class FusedDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


## 像素归一化
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply

## 加权滤波函数
class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


## 归一化了权重的卷积层
class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv) ##归一化卷积层的权重

    def forward(self, input):
        return self.conv(input)


## 全连接层
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)

## 自适应的IN层
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel) ##创建IN层
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        #print("AdaIN style input="+str(style.shape)) #默认值，风格向量长度512
        ## 输入style为风格向量，长度为512；经过self.style得到输出风格矩阵，通道数等于输入通道数的2倍
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1) ##获得缩放和偏置系数，按1轴分为2块
        #print("AdaIN style output="+str(style.shape))
#等于输入通道数的2倍，in_channel*2

        out = self.norm(input) ##IN归一化
        out = gamma * out + beta

        return out

## 添加噪声，噪声权重可以学习
class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise

## 固定输入
class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out


## 风格模块层，包括两个卷积，两个AdaIN层
class StyledConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
        initial=False,
        upsample=False,
        fused=False,
    ):
        super().__init__()

        ## 第1个风格层，初始化4×4×512的特征图
        if initial:
            self.conv1 = ConstantInput(in_channel) 

        else:
            if upsample: 
                ## 对于128及以上的分辨率使用转置卷积上采样
                if fused: 
                    self.conv1 = nn.Sequential(
                        FusedUpsample(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),##滤波操作
                    )

                else:
                    ## 对于分辨小于128，使用普通的最近邻上采样
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'), 
                        EqualConv2d(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),##滤波操作
                    )

            else: ##非上采样层
                self.conv1 = EqualConv2d(
                    in_channel, out_channel, kernel_size, padding=padding
                )

        self.noise1 = equal_lr(NoiseInjection(out_channel)) ##噪声模块1
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim) ##AdaIN模块1
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise):
        out = self.conv1(input)
        out = self.noise1(out, noise) 
        out = self.lrelu1(out)
        out = self.adain1(out, style)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)
        out = self.adain2(out, style)

        return out

## 生成器主架构
class Generator(nn.Module):
    def __init__(self, code_dim, fused=True):
        super().__init__()
        ## 计数变量，用于获取w
        self.global_count = 0
        ##  9个尺度的卷积block，从4×4到64×64，使用双线性上采样；从64×64到1024×1024，使用转置卷积进行上采样
        self.progression = nn.ModuleList(
            [
                StyledConvBlock(512, 512, 3, 1, initial=True),  # 4×4
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 8×8
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 16×16
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 32×32
                StyledConvBlock(512, 256, 3, 1, upsample=True),  # 64×64
                StyledConvBlock(256, 128, 3, 1, upsample=True, fused=fused),  # 128×128
                StyledConvBlock(128, 64, 3, 1, upsample=True, fused=fused),  # 256×256
                StyledConvBlock(64, 32, 3, 1, upsample=True, fused=fused),  # 512×512
                StyledConvBlock(32, 16, 3, 1, upsample=True, fused=fused),  # 1024×1024
            ]
        )
        ## 9个尺度的1*1构成的to_rgb层，与前面对应
        self.to_rgb = nn.ModuleList(
            [
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(256, 3, 1),
                EqualConv2d(128, 3, 1),
                EqualConv2d(64, 3, 1),
                EqualConv2d(32, 3, 1),
                EqualConv2d(16, 3, 1),
            ]
        )

    def forward(self, style, noise, step=0, alpha=1, mixing_range=(-1, 1)):
        out = noise[0] ## 取噪声向量为输入

        if len(style) < 2: ## 不进行样式混合，inject_index=10
            inject_index = [len(self.progression) + 1]
            #print("len(style)<2")
        else:
            ## 生成长度等于style向量，最大不超过step的升序排列随机index，step=9，len(style)=8，比如[0, 2, 3, 4, 5, 6, 7, 8]
            inject_index = sorted(random.sample(list(range(step)), len(style) - 1))
   
        #print("inject_index="+str(inject_index)) ##default=10
        crossover = 0 ##用于mix的位置
 
        ##存储W向量
        np.save('results/w/'+str(self.global_count)+'.npy',style[0].cpu().detach().numpy())
        self.global_count = self.global_count + 1

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            #print("the resolution is="+str(4*np.power(2,i)))
            if mixing_range == (-1, 1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))
                #print("in mixing range,crossover="+str(crossover))
                style_step = style[crossover] ##获得交叉的style起始点

            else:
                ## 样式混合
                #print("not in mixing range")
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1] #取第2个样本样式
                else:
                    style_step = style[0] #取第1个样本样式

            if i > 0 and step > 0:
                out_prev = out
                
            ## 将噪声与风格向量输入风格模块
            #print("batchsize="+str(len(style_step))+",style shape="+str(style_step[0].shape))
            out = conv(out, style_step, noise[i]) 

            if i == step: ## 最后1级分辨率，输出图片
                out = to_rgb(out) ##1×1卷积

                ## 最后结果是否进行alpha融合
                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = F.interpolate(skip_rgb, scale_factor=2, mode='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out

                break

        return out

## 完整的StyleGAN生成器定义
class StyledGenerator(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        super().__init__()

        self.generator = Generator(code_dim) ##synthesis network

        ## mapping network定义，包含8个全连接层，n_mlp=8
        layers = [PixelNorm()] 
        for i in range(n_mlp): 
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        ## style即向量W
        self.style = nn.Sequential(*layers)

    def forward(
        self,
        input, ##属性向量Z
        noise=None, ##噪声向量，可选的
        step=0,
        alpha=1,
        mean_style=None,##属性向量W
        style_weight=0,
        mixing_range=(-1, 1),
    ):
        styles = [] ##风格向量W
        if type(input) not in (list, tuple):
            input = [input]

        #print("混合的样本数input size="+str(len(input))) ##input=(1,(n_sample, 512))
        for i in input:
            styles.append(self.style(i)) ##取得第i组样本的风格向量，样式混合时需要

        batch = input[0].shape[0] ## batchsize大小

        if noise is None:
            noise = []

            for i in range(step + 1): ## 0～8，共9层noise
                size = 4 * 2 ** i ## 每一层的尺度，第一层为4*4
                noise.append(torch.randn(batch, 1, size, size, device=input[0].device))

        if mean_style is not None:
            styles_norm = [] ##风格数组[1*512]

            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))

            styles = styles_norm

            print("has mean_style,the shape="+str(len(mean_style))+' '+str(mean_style[0].shape)+' the weight is'+str(1-style_weight)) #1*512
        #print("混合的样本数styles.shape="+str(len(styles))+' '+str(styles[0].shape)) #styles[0].shape=batchsize*512
        return self.generator(styles, noise, step, alpha, mixing_range=mixing_range)

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdim=True)
        return style

## 判别器
##判别器用的卷积块
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        downsample=False,
        fused=False,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
        )

        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
                    nn.LeakyReLU(0.2),
                )

            else:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                    nn.AvgPool2d(2),
                    nn.LeakyReLU(0.2),
                )

        else:
            self.conv2 = nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out

## Progressive判别器
class Discriminator(nn.Module):
    def __init__(self, fused=True, from_rgb_activate=False):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),  ## 512×512
                ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  ## 256×256
                ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  ## 128×128
                ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  ## 64×64
                ConvBlock(256, 512, 3, 1, downsample=True),  ## 32×32
                ConvBlock(512, 512, 3, 1, downsample=True),  ## 16×16
                ConvBlock(512, 512, 3, 1, downsample=True),  ## 8×8
                ConvBlock(512, 512, 3, 1, downsample=True),  ## 4×4
                ConvBlock(512, 512, 3, 1, 4, 0),
            ]
        )

        ## 从RGB图片转为概率
        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return nn.Sequential(EqualConv2d(3, out_channel, 1), nn.LeakyReLU(0.2))

            else:
                return EqualConv2d(3, out_channel, 1)

        self.from_rgb = nn.ModuleList(
            [
                make_from_rgb(16),
                make_from_rgb(32),
                make_from_rgb(64),
                make_from_rgb(128),
                make_from_rgb(256),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
            ]
        )

        self.n_layer = len(self.progression)
        self.linear = EqualLinear(512, 1)

    def forward(self, input, step=0, alpha=1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step: ##最高级，输入图片
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            ## 判别器的相邻层融合
            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out
