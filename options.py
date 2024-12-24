import os
import random

import numpy as np
import torch
import argparse

from util import util


class SIRSOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize()

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--arch', type=str, default='dsit_large',
                                 help='chooses which architecture to use.')
        self.parser.add_argument('--weight_path', type=str, default=None, help='checkpoint to use.')
        self.parser.add_argument('--backbone_weight_path', type=str, default="./weights/swin_large_o365_finetune.pth",
                                 help='backbone checkpoint to use.')
        self.parser.add_argument('--name', type=str, default=None,
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--model', type=str, default='dsit_model_sirs_lrm', help='chooses which model to use.')
        self.parser.add_argument('--dataset', type=str, default='sirs_dataset', help='chooses which dataset to use.')
        self.parser.add_argument('--loss', type=str, default='losses', help='chooses which loss to use.')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--base_dir', type=str, default='~/workspace/reflection-removal')
        self.parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
        self.parser.add_argument('--resume_epoch', type=int, default=None, help='checkpoint to use. (default: latest')

        # training settings
        self.parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
        self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=None,
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--nEpochs', '-n', type=int, default=60, help='# of epochs to run')
        self.parser.add_argument('--img_size', type=int, default=384)
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--wd', type=float, default=0, help='weight decay for adam')
        self.parser.add_argument('--num_train', type=int, default=-1)

        # testing settings
        self.parser.add_argument('--test_nature', action='store_true',
                                 help='involving nature dataset for testing or not')
        self.parser.add_argument('--test_dir', type=str, default='./data/test')


        # for displaying
        self.parser.add_argument('--no-log', action='store_true', help='disable tf logger?')
        self.parser.add_argument('--no-verbose', action='store_true', help='disable verbose info?')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_id', type=int, default=0,
                                 help='window id of the web display (use 0 to disable visdom)')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                                 help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--display_freq', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--update_html_freq', type=int, default=1000,
                                 help='frequency of saving training results to html')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

        # for logging
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--eval_freq', type=int, default=1, help='frequency of evaluation')
        self.parser.add_argument('--save_freq', type=int, default=1, help='frequency of save eval samples')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1,
                                 help='frequency of saving checkpoints at the end of epochs')
        # for debugging
        self.parser.add_argument('--debug', action='store_true',
                                 help='only do one epoch and displays at each iteration')
        self.parser.add_argument('--debug_eval', action='store_true',
                                 help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--graph', action='store_true', help='print graph')
        self.parser.add_argument('--selected', type=str, nargs='+')

        # data augmentation
        self.parser.add_argument('--batchSize', '-b', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=str, default='224,336,448', help='scale images to multiple size')
        self.parser.add_argument('--fineSize', type=str, default='224,224', help='then crop to this size')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--size_rounded', action='store_true', help='if round the image size by 32x')

        # loss weights
        self.parser.add_argument('--vgg_layer', type=int, default=31, help='vgg layer of unaligned loss')
        self.parser.add_argument('--init_lr', type=float, default=1e-2, help='initial learning rate')
        self.parser.add_argument('--fixed_lr', type=float, default=0, help='initial learning rate')
        self.parser.add_argument('--lambda_vgg', type=float, default=0.1, help='weight for vgg loss')
        self.parser.add_argument('--lambda_rec', type=float, default=0.2, help='weight for reconstruction loss')

    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.isTrain = True

        if self.opt.seed == 0:
            seed = random.randrange(2 ** 12 - 1)
            self.opt.seed = seed

        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.opt.seed)
        np.random.seed(self.opt.seed)  # seed for every module
        random.seed(self.opt.seed)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        self.opt.name = self.opt.name or '_'.join([self.opt.model])
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        return self.opt
