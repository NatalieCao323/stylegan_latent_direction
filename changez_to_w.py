#coding:utf8

# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
# =============================================================================

import argparse
import math
import os
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision import utils
from PIL import Image
import numpy as np
import glob
from model import StyledGenerator

if __name__ == "__main__":
    device = "cpu"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint"
    )
    parser.add_argument(
        "--zfiles", type=str, help="input files"
    )
    parser.add_argument(
        "--wfiles", type=str, help="output files"
    )

    args = parser.parse_args()

    netG = StyledGenerator(512)
    netG.load_state_dict(torch.load(args.ckpt,map_location=device)["g_running"], strict=False)
    netG.eval()
    netG = netG.to(device)

    npys = glob.glob(args.zfiles+'*.npy')

    ## 优化学习W向量
    for npyfile in npys:
        print("npyfile"+str(npyfile))
        latent = torch.from_numpy(np.load(npyfile))
        if len(latent.shape) == 1:
            latent = latent.unsqueeze(0)
        style = netG.style((latent).to(device)) ##将z向量变成w向量
        np.save(npyfile.replace(args.zfiles,args.wfiles),style.detach().numpy()) ##存储W方向向量


