import time
import os

import cv2
import torch

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
from util.metrics import getpsnr
from util.metrics import SSIM
from util.metrics import ssim
from PIL import Image

# 对齐ssim
import os.path as osp
import numpy as np
from pytorch_msssim import ssim as ssim_standard

# 对齐lpips
import lpips

if __name__ == '__main__':
	opt = TestOptions().parse()
	opt.nThreads = 1
	opt.batchSize = 1
	opt.serial_batches = True
	opt.no_flip = True

	data_loader = CreateDataLoader(opt)
	dataset = data_loader.load_data()
	model = create_model(opt)
	visualizer = Visualizer(opt)
	# create website
	web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s_%s' % (opt.phase, opt.which_epoch,opt.snrnote))
	webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
	# test
	avgPSNR = 0.0
	avgSSIM = 0.0
	avgPSNR_i = 0.0
	avgSSIM_i = 0.0
	avgPSNR_1 = 0.0
	# new metric
	avgSSIM_me = 0.0
	avgLPIPS_alex = 0.0
	avgLPIPS_vgg = 0.0
	counter = 0

	# lpips
	lpips_score_alex = lpips.LPIPS(net='alex').cuda()
	lpips_score_vgg = lpips.LPIPS(net='vgg').cuda()

	# 加上一个图片保存的路径, 如果没有就创建
	save_dir = './result_pair'
	os.makedirs(save_dir, exist_ok=True)

	for i, data in enumerate(dataset):
		if i >= opt.how_many:
			break
		counter = i
		# pdb.set_trace()
		model.set_input(data)
		with torch.no_grad():
			model.test() # test里面会针对各个部分进行前向推理，然存储为自己的self变量

		# 模型的输出归一化到（0-1）之间以后计算ssim
		fake_screen = (model.fake_Bi + 1) / 2.0
		real_screen = (model.real_B + 1) / 2.0
		avgSSIM_me += ssim_standard(fake_screen, real_screen, data_range=1, size_average=True)

		visuals = model.get_current_visuals()
		avgPSNR += PSNR(visuals['fake_B'],visuals['real_B']) # fake_B是在step1中由AE生成的，fake_Bi是由半影带生成的
		avgPSNR_i += PSNR(visuals['fake_Bi'],visuals['real_B'])
		avgPSNR_1 += getpsnr(visuals['fake_Bi'],visuals['real_B'])
		avgSSIM += ssim(visuals['fake_B'],visuals['real_B']) # 图片的范围都是0-255
		avgSSIM_i += ssim(visuals['fake_Bi'],visuals['real_B'])

		# 把visuals里面的东西保存，visuals里面的内容已经是numpy的了，新建一个目录
		results = np.concatenate((visuals['fake_Bi'], visuals['real_B']), axis=1)
		results = cv2.cvtColor(results, cv2.COLOR_RGB2BGR)
		cv2.imwrite(osp.join(save_dir, f'test_{str(i)}.jpg'), results)


		# 从0-255的numpy再变到0-255的double的tensor
		# fake = np.transpose(visuals['fake_Bi'], (2,0,1)) # H,W,C -> C,H,W
		# real = np.transpose(visuals['real_B'], (2,0,1))
		# fake = torch.from_numpy(fake).float()
		# fake = fake.unsqueeze(0).cuda() # B,C,H,W
		# real = torch.from_numpy(real).float()
		# real = real.unsqueeze(0).cuda()
		#avgSSIM_me += ssim_standard(fake, real, data_range=255, size_average=True)

		# lpips这些是需要归一化到(-1,1)以后计算的
		# fake = fake / 255.0
		# real = real / 255.0
		# lpips_alex = lpips_score_alex((fake-0.5)/0.5, (real-0.5)/0.5)
		# lpips_alex = torch.mean(lpips_alex)
		# avgLPIPS_alex += lpips_alex
		avgLPIPS_alex = 0

		# lpips_vgg = lpips_score_vgg((fake - 0.5) / 0.5, (real - 0.5) / 0.5)
		# lpips_vgg = torch.mean(lpips_vgg)
		# avgLPIPS_vgg += lpips_vgg
		avgLPIPS_vgg = 0

		img_path = model.get_image_paths()
		print('process image... %s' % img_path)
		visualizer.save_images(webpage, visuals, img_path)


	avgPSNR /= counter
	avgSSIM /= counter
	avgPSNR_i /= counter
	avgPSNR_1 /= counter
	avgSSIM_i /= counter
	avgSSIM_me /= counter
	avgLPIPS_alex /= counter
	avgLPIPS_vgg /= counter
	txtName = "note.txt"
	filedir = os.path.join(web_dir,txtName)
	f=open(filedir, "a+")
	new_context = 'PSNR = '+  str(avgPSNR) + ';SSIM=' + str(avgSSIM) + '\n'+ ';PSNR_i=' + str(avgPSNR_i) +';PSNR_1=' + str(avgPSNR_1) + ';SSIM_i=' + str(avgSSIM_i) + '\n'
	f.write(new_context)
	print('PSNR = %f, SSIM = %f,PSNR_i = %f, PSNR_1 = %f, SSIM_i = %f' %
					  (avgPSNR, avgSSIM, avgPSNR_i,avgPSNR_1, avgSSIM_i))

	print('standard_ssim:', avgSSIM_me)
	print('lpips_alex:', avgLPIPS_alex)
	print('lpips_vgg:', avgLPIPS_vgg)

	webpage.save()
