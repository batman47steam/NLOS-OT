#!/bin/bash

CUDA_VISIBLE_DEVICES=3
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
--name=local_server
--niter=100
--niter_decay=300
--which_model_netG
introAE
--display_port=8097
--norm
batch