

import os
import numpy as np
import PIL
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import pandas as pd
import cv2

class HeartData(data.Dataset):
    def __init__(self,dfp,data_len=-1, image_size=[224, 224]):
        # test_df = pd.read_csv('/home/engs2456/Documents/work/DPHIL/occ/ALOCC/data/test_df_alocc_without_situs.csv')
        self.dfp = dfp
        self.df  = pd.read_csv(dfp)
        self.data_len = data_len
        self.tfs = transforms.Compose([transforms.ToPILImage(),
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        
        if self.data_len!=-1:
            self.df = self.df.sample(data_len).reset_index(drop=True)
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.df['Image_Path'].loc[index]
        lab     = self.df['label'].loc[index]
        img = cv2.imread(path)
        if self.tfs:
            img =  self.tfs(img)
            
        return {'image':img,'class_label':lab}
    def __len__(self):
        return self.df.shape[0]



