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
--name=stl10_wall_dark1_d100_L1_occluder_new_batch_ep10
--niter=40
--niter_decay=300
--which_model_netG
SimMIM
--lossType=L1
--norm
batch
--which_data
stl10
--which_ep=10
--display_port=8097

# step2只会训练一个自己的Encoder2, 其余的Encoder和Decoder都是从step1加载的
# 问题1：是从哪里加载的step1中的权重
# 在base_model的load_ae里以相对固定的方式来加载权重
# 自己定义为了Encoder_1_+which_ep+.pth & Decoder_1_+which_ep+.pth
# 从nlos-ot的trained_weight文件夹下加载

# 训练过程中权重保存的话很简单，也是在由opt.checkpoints和opt.name决定
# '%s_net_%s.pth' % 10_net_G_decoder反正也是这种形式
# 不管你加载的step1的encoder和decoder是第几个epoch的，保存的时候统一记为step2中对应的epoch数目

# 测试的话，就是load_network来加载权重，这个是和训练时候的save是统一的，你只要保真和训练时候的路径一样就行了
# 指明--name参数