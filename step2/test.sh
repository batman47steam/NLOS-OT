CUDA_VISIBLE_DEVICES=3
python ./train.py
--datarootTarget
..\nlos-ot\B
--datarootData
..\nlos-ot\C_dark_1_d100_occluder\
--datarootValTarget
..\nlos-ot\test\gt_val\
--datarootValData
..\nlos-ot\test\C_dark_1_d100_occluder_val\
--learn_residual
--gpu_ids=0
--batchSize=5
--name=local_server # 同理和checkpoint_dirs一起构成文件保存的路径
--niter=40
--niter_decay=300
--which_model_netG
introAE
--lossType=L1
--norm
batch
--which_data
stl10
--which_ep=10
--display_port=8097

# 还有个关键的问题是从哪里去加载step1中的权重
# 有点印象的是说，在step2保存权重的时候，不管加载的step1中的是哪一个epoch的
# save的时候就是统一成step2的encoder训练的epoch数目