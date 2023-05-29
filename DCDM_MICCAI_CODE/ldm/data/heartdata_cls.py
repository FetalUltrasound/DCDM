

import os
import numpy as np
import PIL
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import pandas as pd
import cv2
import torch

class HeartData(data.Dataset):
    def __init__(self,dfp,data_len=-1, image_size=[224, 224]):
        # test_df = pd.read_csv('/home/engs2456/Documents/work/DPHIL/occ/ALOCC/data/test_df_alocc_without_situs.csv')
        self.dfp = dfp
        self.df  = pd.read_csv(dfp)
        self.data_len = data_len
        self.image_size = image_size
        self.tfs = transforms.Compose([transforms.ToPILImage(),
                transforms.Resize((self.image_size[0], self.image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.tfs_cond = transforms.Compose([transforms.ToTensor(),transforms.Resize((self.image_size[0],self.image_size[1]))])
        if self.data_len!=-1:
            self.df = self.df.sample(data_len)
            self.df = self.df.reset_index(drop=True)
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path    = self.df['Image_Path'].loc[index]
        lab     = self.df['label'].loc[index]
        # ref_img = cv2.resize(cv2.imread(f'/home/engs2456/Documents/work/DPHIL/occ/latent-diffusion/latent-diffusion/ref_dig/{lab}.jpeg',0),(self.image_size[0], self.image_size[1]))
        # print("REF Image SHAPE",ref_img.shape)
        img = cv2.imread(path)
        gray = cv2.imread(path,0)
        # lab = torch.IntTensor([lab])
        if self.tfs:
            img =  self.tfs(img)
            # ref_img = self.tfs(ref_img)

        if self.tfs_cond:
            gray = self.tfs_cond(gray)

        # print(img.shape)
        # print(lab.shape)
        # print(gray.shape)
        return {'image':img,'class_label':lab,'cond_img':gray}
    def __len__(self):
        return self.df.shape[0]



