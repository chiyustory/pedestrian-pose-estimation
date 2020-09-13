import os
# import torch
import argparse
import pdb

attribute = {
    'HairStyle': {'ShortHair':[0,0,0], 'LongHair':[0,0,0]},
    'Sex': {'Female':[0,0,0],'Male':[0,0,0]},
    'DownClothes':{'ShortPants':[0,0,0],'LongPants':[0,0,0],'Skirts':[0,0,0]}
}

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--img_dir', type=str, default='/home/yl/Code/ped_pose_estimation/image/', help='image directory')
        self.parser.add_argument('--out_dir', type=str, default='/home/yl/Code/ped_pose_estimation/output/', help='out directory')
        self.parser.add_argument('--model_dir', type=str, default='/home/yl/Code/ped_pose_estimation/model/', help='model directory')

        self.parser.add_argument('--train_pth', type=str, default='/home/yl/Code/ped_pose_estimation/dataset/test.txt', help='train file path')
        self.parser.add_argument('--val_pth', type=str, default='/home/yl/Code/ped_pose_estimation/dataset/test.txt', help='validation file path')
        self.parser.add_argument('--test_pth', type=str, default='/home/yl/Code/ped_pose_estimation/dataset/test.txt', help='test file path')
        self.parser.add_argument('--device_ids', default=[], help='GPU ID')


        self.parser.add_argument('--input_size', default=[224, 224], help='scale image to the size prepared for croping')
        self.parser.add_argument('--heatmap_size', default=[56,56], help='scale image to the size prepared for croping')

        self.parser.add_argument('--mode', default='Test', help='run mode of training or testing. [Train | Test | train | test]')
        self.parser.add_argument('--model',default='Resnet50', help='model type. [Alexnet | LightenB | VGG16 | Resnet18 | ...]')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size of network input. Note that batch_size should only set to 1 in Test mode')
        self.parser.add_argument('--load_thread', type=int, default=1, help='how many subprocesses to use for data loading')
        self.parser.add_argument('--mean', default=(123/255,117/255,104/255), help='sequence of means for each channel used for normization')
        self.parser.add_argument('--std', default=(1,1,1), help='sequence standard deviations for each channel used for normization')
        self.parser.add_argument('--checkpoint_name', type=str, default='/home/yl/Code/ped_pose_estimation/model/epoch_40_snapshot.pth', help='path to pretrained model or model to deploy')
        # self.parser.add_argument('--checkpoint_name', type=str, default='/home/yl/Code/model/resnet50-19c8e357.pth', help='path to pretrained model or model to deploy')       
        self.parser.add_argument('--sum_epoch', type=int, default=80, help='sum epoches for training')
        self.parser.add_argument('--save_epoch_freq', type=int, default=20, help='save snapshot every $save_epoch_freq epoches training')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
        self.parser.add_argument('--optim', type=str, default='Adam', help='optimizer algrithm.')
        self.parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay.')
        self.parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_in_epoch', type=int, default=20, help='multiply by a gamma every lr_decay_in_epoch iterations')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
        self.parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay of SGD')
        self.parser.add_argument('--display_train_freq', type=int, default=1, help='print train loss and accuracy every $train_freq batches iteration')
        self.parser.add_argument('--display_validate_freq', type=int, default=1, help='test validate dateset every $validate_freq batches iteration')

        # pose estimation
        self.parser.add_argument('--sigma',default=2, help='test validate dateset every $validate_freq batches iteration')
        self.parser.add_argument('--final_conv_kernel', default=1, help='test validate dateset every $validate_freq batches iteration')
        self.parser.add_argument('--deconv_with_bias', default=False, help='test validate dateset every $validate_freq batches iteration')
        self.parser.add_argument('--num_deconv_layers', default=3, help='test validate dateset every $validate_freq batches iteration')
        self.parser.add_argument('--num_deconv_filters', default=[256,256,256], help='test validate dateset every $validate_freq batches iteration')
        self.parser.add_argument('--num_deconv_kernels', default=[4,4,4], help='test validate dateset every $validate_freq batches iteration')


    def parse(self):
        opt = self.parser.parse_args()

        if not os.path.exists(opt.out_dir):
            os.makedirs(opt.out_dir)
        if not os.path.exists(opt.model_dir):
            os.makedirs(opt.model_dir)

        return opt

if __name__ == "__main__":
    op = Options()
    op.parse()
