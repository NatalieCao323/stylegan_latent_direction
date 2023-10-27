import os
import math
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import utils
import numpy as np
from generate import sample, get_mean_style
from model import StyledGenerator

SIZE = 1024
import cv2

class Predictor():
    def __init__(self,modelpath):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = StyledGenerator(512).to(self.device)
        print("Loading checkpoint")

        ## 载入模型
        weights = torch.load(modelpath,map_location=self.device)
        for key in weights.keys():
            print(key+'\n')
        #self.generator.load_state_dict(weights["g_running"])
        self.generator.load_state_dict(weights["generator"])
        self.generator.eval()

        ## 平均风格向量
        self.mean_style = get_mean_style(self.generator, self.device)
        print("mean style:"+str(self.mean_style.shape))
        #print(self.mean_style)

    def predict(self, seed, output_path):
        if seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")
        torch.manual_seed(seed) #为CPU设置种子用于生成随机数，以使得结果是确定的

        step = int(math.log(SIZE, 2)) - 2 ##step=8，包含8次上采样
        nsamples = 1
        latent_in = torch.rand(nsamples, 512)
        img = self.generator(
            latent_in.to(self.device),
            step=step,
            alpha=1,
        )

        img = F.interpolate(img,scale_factor=0.25)

        utils.save_image(img, output_path, normalize=True)
        np.save(output_path.replace('.png','.npy'),latent_in.cpu().detach().numpy())

        '''
        img = img.numpy().squeeze()
        img = np.transpose(img,(1,2,0))
        img = (img - np.min(img)) / (np.max(img)-np.min(img))
        img = (img*255.0).astype(np.uint8)
        cv2.imwrite('test.png',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))        
        print(img.shape)
        print(np.max(img))
        print(np.min(img))
        '''

if __name__ == '__main__':
    modelpath = "checkpoints/stylegan-1024px-new.model"
    predictor = Predictor(modelpath)
    for i in range(0,50000):
        predictor.predict(i,'results/'+str(i)+'.png')
    
