import os
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import csv
import glob

'''
Using the Pavia Center dataset as an example
'''
class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(TrainDatasetFromFolder, self).__init__()
        self.root = dataset_dir

        self.gtHS = glob.glob(os.path.join(self.root, "train", "gtHS", '*.mat'))
        self.gtHS.sort(key=lambda x: int(x.split('.')[0].split('gtHS/')[-1]))
        self.LRHS = glob.glob(os.path.join(self.root, "train", "LRHS", '*.mat'))
        self.LRHS.sort(key=lambda x: int(x.split('.')[0].split('LRHS/')[-1]))
        self.hrMS = glob.glob(os.path.join(self.root, "train", "hrMS", '*.mat'))
        self.hrMS.sort(key=lambda x: int(x.split('.')[0].split('hrMS/')[-1]))

    def __getitem__(self, index):
        gt_HS, LR_HS, PAN = self.gtHS[index], self.LRHS[index], self.hrMS[index]
        data_ref = loadmat(os.path.join(self.root, "train", "gtHS", gt_HS))['gtHS'].reshape(102, 160, 160).astype(np.float32)
        data_LRHS = loadmat(os.path.join(self.root, "train", "LRHS", LR_HS))['LRHS'].reshape(102, 40, 40).astype(np.float32)
        data_hrMS = loadmat(os.path.join(self.root, "train", "hrMS", PAN))['hrMS'].reshape(4, 160, 160).astype(np.float32)
        return data_hrMS, data_LRHS, data_ref

    def __len__(self):
        return len(self.gtHS)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder, self).__init__()
        self.root = dataset_dir

        self.gtHS = glob.glob(os.path.join(self.root, "test", "gtHS", '*.mat'))
        self.gtHS.sort(key=lambda x: int(x.split('.')[0].split('gtHS/')[-1]))
        self.LRHS = glob.glob(os.path.join(self.root, "test", "LRHS", '*.mat'))
        self.LRHS.sort(key=lambda x: int(x.split('.')[0].split('LRHS/')[-1]))
        self.PAN = glob.glob(os.path.join(self.root, "test", "hrMS", '*.mat'))
        self.PAN.sort(key=lambda x: int(x.split('.')[0].split('hrMS/')[-1]))

    def __getitem__(self, index):

        gt_HS, LR_HS, PAN = self.gtHS[index], self.LRHS[index], self.PAN[index]
        data_ref = loadmat(os.path.join(self.root, "test", "gtHS", gt_HS))['gtHS'].reshape(102, 160, 160).astype(np.float32)
        data_LRHS = loadmat(os.path.join(self.root, "test", "LRHS", LR_HS))['LRHS'].reshape(102, 40, 40).astype(np.float32)
        data_hrMS = loadmat(os.path.join(self.root, "test", "hrMS", PAN))['hrMS'].reshape(4, 160, 160).astype(np.float32)
        return data_hrMS, data_LRHS, data_ref

    def __len__(self):
        return len(self.gtHS)

