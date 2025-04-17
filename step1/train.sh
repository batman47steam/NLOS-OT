#!/bin/bash

CUDA_VISIBLE_DEVICES=0
python ./train.py
--datarootTarget
..\nlos-ot\B
--datarootData
..\nlos-ot\B
--datarootValTarget
..\nlos-ot\test\gt_val
--datarootValData
..\nlos-ot\test\gt_val
--learn_residual
--gpu_ids=0
--batchSize=16
--name=local_server # 保存的权重路径是由opt.checkpoints_dir和opt.name决定的，checkpoints_dir一般默认
--niter=100
--niter_decay=300
--which_model_netG
introAE
--display_port=8097
--norm
batch

# step1最重要的两个问题
# 保存的结果在哪里 ？
# 保存的是什么 ？
# 保存的路径由 opt.checkpoints(默认checkpoints)和opt.name决定
# base_model的save_network里，epoch_label_net_G_decoder.pth (epoch_label) 就是对应的epoch数
