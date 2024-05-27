import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import TrainDatasetFromFolder, TestDatasetFromFolder
from torch import nn, optim
import torch
import numpy as np
from model import SuperResolutionModel
from caculate_utils import CC_function, psnr, SAM, ssim
from scipy.io import savemat
import random
import os
from tqdm import tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_epochs = 500
learning_rate = 0.001
model_path = '' # Path to store the model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weights_init(model):
    def init_func(m):
        classname = m.__class__.__name__
        # hasattr判断对象是否具有特定的属性
        if hasattr(m, 'weight'):
            torch.nn.init.normal_(m.weight.data, mean=0, std=1)

    model.apply(init_func)

def get_lr(optim_model):
    for param_group in optim_model.param_groups:
        return param_group['lr']


if __name__ == "__main__":

    set_seed(1)

    Run_loss, Test_loss = np.array([]), np.array([])
    path = '' # Dataset Path
    train_data = TrainDatasetFromFolder(path)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4, drop_last=False)

    test_data = TestDatasetFromFolder(path)
    test_loader = DataLoader(test_data, batch_size=4, num_workers=4, shuffle=True)

    criteon = nn.L1Loss()
    best_loss = 1

    model = SuperResolutionModel(drop=0,
                                 in_channels=102,
                                 out_channels=102,
                                 mid_channels=64,
                                 factor=4,
                                 num_fc=4)
    print('# model parameters:', sum(param.numel() for param in model.parameters()))
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 150, 300], gamma=0.1, last_epoch=-1)


    # Start training
    for i in range(num_epochs):

        pn, runloss, cc, sam = 0, 0, 0, 0
        for step, (hrMS, lrHS, ref) in enumerate(train_loader):
            hrMS = hrMS.type(torch.float).cuda()
            lrHS = lrHS.type(torch.float).cuda()
            ref = ref.type(torch.float).cuda()
            model.train()
            output = model(lrHS, hrMS)
            running_loss = criteon(output, ref)
            pn += psnr(output.cpu().detach().numpy(), ref.cpu().detach().numpy())
            cc += CC_function(output.cpu().detach().numpy(), ref.cpu().detach().numpy())
            sam += SAM(output.cpu().detach().numpy(), ref.cpu().detach().numpy())
            optimizer.zero_grad()
            running_loss.backward()
            optimizer.step()
            runloss += running_loss.item()


        if i % 100 == 0:
            torch.save(model.state_dict(),os.path.join(model_path, 'epoch_%d_params.pth' % (i)))
        print('epoch', i + 1, 'train_loss', runloss / (step + 1), 'psnr', pn / (step + 1), 'CC', cc / (step + 1),
              'sam', sam / (step + 1), 'total', step + 1)

        scheduler.step()
        # Run_loss = np.append(Run_loss, runloss / (step + 1))

