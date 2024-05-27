import argparse
import time
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from dataloader import TestDatasetFromFolder
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader
from model import SuperResolutionModel
from tqdm import tqdm
#import pytorch_ssim
from math import log10

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = '' # Model to be loaded (filename)

data_path = '' # Dataset path
fusion_path = '' # Output path
test_set = TestDatasetFromFolder(data_path)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)

model = SuperResolutionModel(drop=0,
                             in_channels=102,
                             out_channels=102,
                             mid_channels=128,
                             factor=4,
                             num_fc=4)

model.to(device)
model.load_state_dict(torch.load('' + MODEL_NAME)) # Model path


index_test = 1
with torch.no_grad():
    for hrMS, LRHS in test_loader:
        LRHS = LRHS.type(torch.float).to(device)
        hrMS = hrMS.type(torch.float).to(device)

        torch.cuda.synchronize()
        HS_fusion = model(LRHS, hrMS)
        HS_fusion = np.array(HS_fusion.cpu())

        path = os.path.join(fusion_path, str(index_test) + '.mat')

        index_test = index_test + 1
        sio.savemat(path, {'hrHS': HS_fusion.squeeze()})

    print('test finished!')