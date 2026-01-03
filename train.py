import os
import time

import torch.nn as nn
import torch.autograd
from skimage import io
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

working_path = os.path.dirname(os.path.abspath(__file__))

from utils.loss import FocalLoss2d
from utils.utils import accuracy, AverageMeter,SCDD_eval_binary
from tqdm import tqdm

###############################################
from datasets import RS as RS
from models.Network import STG as Net

NET_NAME = 'Loss'
DATA_NAME = 'Loss'
###############################################

# Training options
###############################################
args = {
    'train_batch_size':64,
    'val_batch_size': 64,
    'lr': 0.0001,
    'epochs': 50,
    'gpu': True,
    'lr_decay_power': 0.9,
    'betas': (0.9, 0.999),
    'weight_decay': 0.05,
    'alpha': 0.99,
    'eps': 1e-8,
    'momentum': 0.9,
    'print_freq': 20,
    'predict_step': 1,
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME),
    # 'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'pretrained.pth')
}
###############################################

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
writer = SummaryWriter(args['log_dir'])

def main():
    net = Net(num_classes=RS.num_classes).cuda()

    train_set = RS.Data('train', random_flip=True)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=2, shuffle=True)
    val_set = RS.Data('test')
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=2, shuffle=False)

    criterion = FocalLoss2d(ignore_index=-1).cuda()
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'],weight_decay=args['weight_decay'], betas=args['betas'], eps=args['eps'], amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)
    train(train_loader, net, criterion,optimizer, scheduler, val_loader)
    writer.close()
    print('Training finished.')


def train(train_loader, net, criterion, optimizer, scheduler, val_loader):
    bestaccT = 0
    bestFscdV = 0.0
    bestloss = 1.0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    curr_epoch = 0
    while True:
        torch.cuda.empty_cache()
        net.train()
        # freeze_model(NCNet)
        start = time.time()
        acc_meter_all = AverageMeter()
        acc_meter_T2 = AverageMeter()
        acc_meter_T3 = AverageMeter()
        train_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(tqdm(train_loader)):
            running_iter = curr_iter + i + 1
            adjust_lr(optimizer, running_iter, all_iters)
            imgs_1, imgs_2, imgs_3, labels_12,labels_23,labels = data
            if args['gpu']:
                imgs_1 = imgs_1.cuda().float()
                imgs_2 = imgs_2.cuda().float()
                imgs_3 = imgs_3.cuda().float()
                labels12 = labels_12.cuda().long()
                labels23 = labels_23.cuda().long()
                labels = labels.cuda().long()

            optimizer.zero_grad()
            change, change12, change23  = net(imgs_1, imgs_2, imgs_3)
            loss = criterion(change, labels) + criterion(change12, labels12)+ criterion(change23, labels23)
            loss.backward()
            optimizer.step()
            labels_12 = labels12.cpu().detach().numpy()
            labels_23 = labels23.cpu().detach().numpy()
            labels_BCD = labels.cpu().detach().numpy()

            all_change = torch.sigmoid(change).detach()
            T2 = torch.sigmoid(change12).detach()
            T3 = torch.sigmoid(change23).detach()

            all_change = torch.argmax(all_change,dim=1).cpu().numpy()
            T2 = torch.argmax(T2, dim=1).cpu().numpy()
            T3 = torch.argmax(T3, dim=1).cpu().numpy()

            preds_all = all_change
            preds_T2 = T2
            preds_T3 = T3


            acc_curr_meter_all = AverageMeter()
            acc_curr_meter_T2 = AverageMeter()
            acc_curr_meter_T3 = AverageMeter()

            for (pred_all,  pred_T2, pred_T3,label_12,label_23,label_BCD) in zip( preds_all, preds_T2, preds_T3, labels_12,labels_23,labels_BCD):
                acc_all, valid_sum_all = accuracy(pred_all,  label_BCD)
                acc_T2, valid_sum_T2 = accuracy(pred_T2, label_12)
                acc_T3, valid_sum_T3 = accuracy(pred_T3, label_23)

                acc_curr_meter_all.update(acc_all)
                acc_curr_meter_T2.update(acc_T2)
                acc_curr_meter_T3.update(acc_T3)

            acc_meter_all.update(acc_curr_meter_all.avg)
            acc_meter_T2.update(acc_curr_meter_T2.avg)
            acc_meter_T3.update(acc_curr_meter_T3.avg)

            train_loss.update(loss.cpu().detach().numpy())
            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [acc_all %.2f] [change12 %.2f] [change23 %.2f]' % (curr_epoch, i + 1, len(train_loader), curr_time, acc_meter_all.val * 100,acc_meter_T2.val * 100,acc_meter_T3.val * 100))
                writer.add_scalar('train loss', train_loss.val, running_iter)
                writer.add_scalar('train accuracy all', acc_meter_all.val, running_iter)
                writer.add_scalar('train accuracy T2', acc_meter_T2.val, running_iter)
                writer.add_scalar('train accuracy T3', acc_meter_T3.val, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        mIoU_a,mIoU_1,mIoU_2 = validate(val_loader, net, criterion, curr_epoch)
        torch.save(net.state_dict(), os.path.join(args['chkpt_dir'],NET_NAME + '_%de_all%.2f_T12%.2f_T23%.2f.pth' % (curr_epoch, mIoU_a * 100, mIoU_1* 100, mIoU_2 * 100)))
        curr_epoch += 1
        # scheduler.step()
        if curr_epoch >= args['epochs']:
            return


def validate(val_loader, net, criterion, curr_epoch):

    net.eval()
    # freeze_model(NCNet)
    torch.cuda.empty_cache()
    start = time.time()
    val_loss = AverageMeter()
    preds_all_val = []
    preds_T2_val = []
    preds_T3_val = []
    labels_all_val = []
    labels_T2_val = []
    labels_T3_val = []
    for vi, data in enumerate(tqdm(val_loader)):
        imgs_1, imgs_2, imgs_3, labels_12,labels_23,labels = data
        if args['gpu']:
            imgs_1 = imgs_1.cuda().float()
            imgs_2 = imgs_2.cuda().float()
            imgs_3 = imgs_3.cuda().float()
            labels12 = labels_12.cuda().long()
            labels23 = labels_23.cuda().long()
            labels = labels.cuda().long()

        with torch.no_grad():
            change, change12, change23 = net(imgs_1, imgs_2, imgs_3)
            loss = criterion(change, labels) + criterion(change12, labels12) + criterion(change23,labels23)
        val_loss.update((loss).cpu().detach().numpy())
        labels12 = labels12.cpu().detach().numpy()
        labels23 = labels23.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        all_change = torch.sigmoid(change).detach()
        T2 = torch.sigmoid(change12).detach()
        T3 = torch.sigmoid(change23).detach()

        all_change = torch.argmax(all_change, dim=1).cpu().numpy()
        T2 = torch.argmax(T2, dim=1).cpu().numpy()
        T3 = torch.argmax(T3, dim=1).cpu().numpy()

        preds_all = all_change
        preds_T2 = T2
        preds_T3 = T3

        for (pred_all,  pred_T2, pred_T3,  label_12,label_23,label_BCD) in zip(preds_all, preds_T2, preds_T3, labels12,labels23,labels):

            preds_all_val.append(pred_all)
            preds_T2_val.append(pred_T2)
            preds_T3_val.append(pred_T3)

            labels_all_val.append(label_BCD)
            labels_T2_val.append(label_12)
            labels_T3_val.append(label_23)

        if curr_epoch % args['predict_step'] == 0 and vi == 0:
            pred_color_T2 = RS.Index2Color(preds_T2[0])
            pred_color_T3 = RS.Index2Color(preds_T3[0])
            io.imsave(os.path.join(args['pred_dir'], NET_NAME + 'T2.png'), pred_color_T2)
            io.imsave(os.path.join(args['pred_dir'], NET_NAME + 'T3.png'), pred_color_T3)
            # io.imsave(os.path.join(args['pred_dir'], NET_NAME + 'T4.png'), pred_color_T4)
            print('Prediction saved!')

    OA_a,P_a,R_a,F_a,mIoU_a = SCDD_eval_binary(preds_all_val, labels_all_val, RS.num_classes)
    OA_2, P_2, R_2, F_2, mIoU_2 = SCDD_eval_binary(preds_T2_val, labels_T2_val, RS.num_classes)
    OA_3, P_3, R_3, F_3, mIoU_3 = SCDD_eval_binary(preds_T3_val, labels_T3_val, RS.num_classes)

    curr_time = time.time() - start
    print('%.1fs M_a: %.2f M_1: %.2f M_2: %.2f' % (curr_time, mIoU_a * 100,  mIoU_2 * 100, mIoU_3 * 100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_M_a', mIoU_a, curr_epoch)
    writer.add_scalar('val_M_2', mIoU_2, curr_epoch)
    writer.add_scalar('val_M_3', mIoU_3, curr_epoch)

    return mIoU_a,mIoU_2,mIoU_3


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def adjust_lr(optimizer, curr_iter, all_iter, init_lr=args['lr']):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    main()