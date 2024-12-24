import os
from os.path import join

import torch.backends.cudnn as cudnn

import data.sirs_dataset as datasets
from engine import Engine
from options import SIRSOptions
from tools import mutils

opt = SIRSOptions().parse()

opt.isTrain = False
cudnn.benchmark = True
opt.no_log = True
opt.display_id = 0
opt.verbose = False

test_dataset_real = datasets.RealTestDataset(opt.test_dir, size_rounded=opt.size_rounded)

test_dataloader_real = datasets.DataLoader(test_dataset_real, batch_size=1, shuffle=False,
                                           num_workers=opt.nThreads, pin_memory=True)

engine = Engine(opt)

"""Main Loop"""
result_dir = os.path.join('./checkpoints', opt.name, mutils.get_formatted_time())

res = engine.test(test_dataloader_real, savedir=join(result_dir, 'test'))
