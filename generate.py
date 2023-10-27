import argparse
import math

import torch
from torchvision import utils

from model import StyledGenerator

## 平均风格向量获取
@torch.no_grad()
def get_mean_style(generator,device):
    mean_style = None
    for i in range(100):
        ## 从随机向量Z，经过mapping network得到W
        style = generator.mean_style(torch.randn(1024, 512).mean(0, keepdim=True).to(device))
        if mean_style is None:
            mean_style = style
        else:
            mean_style += style

    mean_style /= 100
    return mean_style

## 根据风格向量生成样本
@torch.no_grad()
def sample(generator, step, mean_style, n_sample, device):
    image = generator(
        torch.randn(n_sample, 512).to(device),
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.,
    )
    
    return image

## 样式混合
@torch.no_grad()
def style_mixing(generator, step, mean_style, n_source, n_target, device):
    ## 两个样式向量
    source_code = torch.randn(n_source, 512).to(device)
    target_code = torch.randn(n_target, 512).to(device)
    
    shape = 4 * 2 ** step ##1024分辨率
    alpha = 1

    images = [torch.ones(1, 3, shape, shape).to(device) * -1]

    ## 源域图
    source_image = generator(
        source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )
    ## 目标域图
    target_image = generator(
        target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )

    images.append(source_image)

    ## 样式混合
    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code], ##输入两组向量
            step=step,
            alpha=alpha,
            mean_style=mean_style,
            style_weight=0.7,
            mixing_range=(0, 1),
        )
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = torch.cat(images, 0)
    
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1024, help='size of the image')
    parser.add_argument('--n_row', type=int, default=2, help='number of rows of sample matrix')
    parser.add_argument('--n_col', type=int, default=4, help='number of columns of sample matrix')
    parser.add_argument('path', type=str, help='path to checkpoint file')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = StyledGenerator(512).to(device) ## Z=512维向量
    generator.load_state_dict(torch.load(args.path,map_location=device)['generator'])
    generator.eval()

    mean_style = get_mean_style(device) ##平均风格向量
    step = int(math.log(args.size, 2)) - 2 ##多少次上采样
    img = sample(generator, step, mean_style, args.n_row * args.n_col, device)
    utils.save_image(img, 'sample.png', nrow=args.n_col, normalize=True, range=(-1, 1))

    for j in range(4):
        img = style_mixing(generator, step, mean_style, args.n_col, args.n_row, device)
        utils.save_image(
            img, 'sample_mixing_'+str(j)+'.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
        )

