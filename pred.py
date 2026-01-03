import os
import time
import argparse
import numpy as np
import torch.autograd
from skimage import io, exposure
from torch.nn import functional as F
from torch.utils.data import DataLoader
#################################
from datasets import RS
# from models.BiSRNet import BiSRNet as Net
import matplotlib
matplotlib.use("Agg")
from models.Network import STG as Net

DATA_NAME = 'ST'


class PredOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        working_path = os.path.dirname(os.path.abspath(__file__))
        parser.add_argument('--pred_batch_size', required=False, default=1, help='prediction batch size')
        parser.add_argument('--test_dir', required=False, default=r'',
                            help='directory to test images')
        parser.add_argument('--pred_dir', required=False, default=r'',
                            help='directory to output masks')
        parser.add_argument('--chkpt_path', required=False,
                            default=r'')
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()
        return self.opt


def main():
    begin_time = time.time()
    opt = PredOptions().parse()
    net = Net(num_classes=RS.num_classes).cuda()
    net.load_state_dict(torch.load(opt.chkpt_path))
    net.eval()

    test_set = RS.Data_test(opt.test_dir)
    test_loader = DataLoader(test_set, batch_size=opt.pred_batch_size)
    predict(net, test_set, test_loader, opt.pred_dir)
    time_use = time.time() - begin_time
    print('Total time: %.2fs' % time_use)


def predict(net, pred_set, pred_loader, pred_dir):
    pred_T2 = os.path.join(pred_dir, '2015_2016')
    pred_T3 = os.path.join(pred_dir, '2016_2018')
    pred_T4 = os.path.join(pred_dir, '2015_2018')

    if not os.path.exists(pred_T2): os.makedirs(pred_T2)
    if not os.path.exists(pred_T3): os.makedirs(pred_T3)
    if not os.path.exists(pred_T4): os.makedirs(pred_T4)
    for vi, data in enumerate(pred_loader):
        imgs_1, imgs_2, imgs_3 = data
        imgs_1 = imgs_1.cuda().float()
        imgs_2 = imgs_2.cuda().float()
        imgs_3 = imgs_3.cuda().float()
        mask_name = pred_set.get_mask_name(vi)
        with torch.no_grad():
            change, change12, change23 = net(imgs_1, imgs_2, imgs_3)
            all_change = torch.sigmoid(change).detach()
            T2 = torch.sigmoid(change12).detach()
            T3 = torch.sigmoid(change23).detach()

        all_change = torch.argmax(all_change, dim=1).cpu().squeeze().numpy()
        T2 = torch.argmax(T2, dim=1).cpu().squeeze().numpy()
        T3 = torch.argmax(T3, dim=1).cpu().squeeze().numpy()
        pred_all_path = os.path.join(pred_T4 , mask_name)
        pred_T2_path = os.path.join(pred_T2, mask_name)
        pred_T3_path = os.path.join(pred_T3, mask_name)
        io.imsave(pred_all_path, RS.Index2Color(all_change))
        io.imsave(pred_T2_path, RS.Index2Color(T2))
        io.imsave(pred_T3_path, RS.Index2Color(T3))
        print(pred_all_path)


if __name__ == '__main__':
    main()