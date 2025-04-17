#!/bin/bash
python train.py
--datarootTarget
../Simulate/2D-B1/train/gt
--datarootData
../Simulate/2D-B1/train/gt
--datarootValTarget
../Simulate/2D-B1/test/gt
--datarootValData
../Simulate/2D-B1/test/gt
--learn_residual
--gpu_ids=0
--batchSize=8
--name=local_untrain # 保存权重的路径是有opt.checkpoints_dir和opt.name决定的，checkpoints_dir一般默认
--niter=100
--niter_decay=300
--which_model_netG
introAE
--display_port=8097
--norm
batch
--input_height
48
--input_width
48
--output_height
48
--output_width
48
--channels
"64, 128, 256, 512"

# step1最重要的两个问题
# 保存的结果在哪里 ？
# 保存的是什么 ？
# 保存的路径由 opt.checkpoints(默认checkpoints)和opt.name决定
# base_model的save_network里，epoch_label_net_G_decoder.pth (epoch_label) 就是对应的epoch数

# 启动visdom
# python -m visdom.server