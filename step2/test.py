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


def reorder_image(img, input_order='HWC'):
	"""Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

	if input_order not in ['HWC', 'CHW']:
		raise ValueError(
			f'Wrong input_order {input_order}. Supported input_orders are '
			"'HWC' and 'CHW'")
	if len(img.shape) == 2:
		img = img[..., None]
	if input_order == 'CHW':
		img = img.transpose(1, 2, 0)
	return img


def _ssim(img1, img2, max_value):
	"""Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

	C1 = (0.01 * max_value) ** 2
	C2 = (0.03 * max_value) ** 2

	img1 = img1.astype(np.float64)
	img2 = img2.astype(np.float64)
	kernel = cv2.getGaussianKernel(11, 1.5)
	window = np.outer(kernel, kernel.transpose())

	mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
	mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
	mu1_sq = mu1 ** 2
	mu2_sq = mu2 ** 2
	mu1_mu2 = mu1 * mu2
	sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
	sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
	sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

	ssim_map = ((2 * mu1_mu2 + C1) *
				(2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
									   (sigma1_sq + sigma2_sq + C2))
	return ssim_map.mean()


def prepare_for_ssim(img, k):
	import torch
	with torch.no_grad():
		img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
		conv = torch.nn.Conv2d(1, 1, k, stride=1, padding=k // 2, padding_mode='reflect')
		conv.weight.requires_grad = False
		conv.weight[:, :, :, :] = 1. / (k * k)

		img = conv(img)

		img = img.squeeze(0).squeeze(0)
		img = img[0::k, 0::k]
	return img.detach().cpu().numpy()


def prepare_for_ssim_rgb(img, k):
	import torch
	with torch.no_grad():
		img = torch.from_numpy(img).float()  # HxWx3

		conv = torch.nn.Conv2d(1, 1, k, stride=1, padding=k // 2, padding_mode='reflect')
		conv.weight.requires_grad = False
		conv.weight[:, :, :, :] = 1. / (k * k)

		new_img = []

		for i in range(3):
			new_img.append(conv(img[:, :, i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)[0::k, 0::k])

	return torch.stack(new_img, dim=2).detach().cpu().numpy()


def _3d_gaussian_calculator(img, conv3d):
	out = conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
	return out


def _generate_3d_gaussian_kernel():
	kernel = cv2.getGaussianKernel(11, 1.5)
	window = np.outer(kernel, kernel.transpose())
	kernel_3 = cv2.getGaussianKernel(11, 1.5)
	kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0))
	conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
	conv3d.weight.requires_grad = False
	conv3d.weight[0, 0, :, :, :] = kernel
	return conv3d


def _ssim_3d(img1, img2, max_value):
	assert len(img1.shape) == 3 and len(img2.shape) == 3
	"""Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.

    Returns:
        float: ssim result.
    """
	C1 = (0.01 * max_value) ** 2
	C2 = (0.03 * max_value) ** 2
	img1 = img1.astype(np.float64)
	img2 = img2.astype(np.float64)

	kernel = _generate_3d_gaussian_kernel().cuda()

	img1 = torch.tensor(img1).float().cuda()
	img2 = torch.tensor(img2).float().cuda()

	mu1 = _3d_gaussian_calculator(img1, kernel)
	mu2 = _3d_gaussian_calculator(img2, kernel)

	mu1_sq = mu1 ** 2
	mu2_sq = mu2 ** 2
	mu1_mu2 = mu1 * mu2
	sigma1_sq = _3d_gaussian_calculator(img1 ** 2, kernel) - mu1_sq
	sigma2_sq = _3d_gaussian_calculator(img2 ** 2, kernel) - mu2_sq
	sigma12 = _3d_gaussian_calculator(img1 * img2, kernel) - mu1_mu2

	ssim_map = ((2 * mu1_mu2 + C1) *
				(2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
									   (sigma1_sq + sigma2_sq + C2))
	return float(ssim_map.mean())


def calculate_ssim_ltm(img1,
					   img2,
					   crop_border,
					   input_order='HWC',
					   test_y_channel=False,
					   ssim3d=True):
	"""Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

	assert img1.shape == img2.shape, (
		f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
	if input_order not in ['HWC', 'CHW']:
		raise ValueError(
			f'Wrong input_order {input_order}. Supported input_orders are '
			'"HWC" and "CHW"')

	if type(img1) == torch.Tensor:
		if len(img1.shape) == 4:
			img1 = img1.squeeze(0)
		img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
	if type(img2) == torch.Tensor:
		if len(img2.shape) == 4:
			img2 = img2.squeeze(0)
		img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

	img1 = reorder_image(img1, input_order=input_order)
	img2 = reorder_image(img2, input_order=input_order)

	img1 = img1.astype(np.float64)
	img2 = img2.astype(np.float64)

	if crop_border != 0:
		img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
		img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

	def _cal_ssim(img1, img2):
		if test_y_channel:
			img1 = to_y_channel(img1)
			img2 = to_y_channel(img2)
			return _ssim_cly(img1[..., 0], img2[..., 0])

		ssims = []
		# ssims_before = []

		# skimage_before = skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True)
		# print('.._skimage',
		#       skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True))
		max_value = 1 if img1.max() <= 1 else 255
		with torch.no_grad():
			final_ssim = _ssim_3d(img1, img2, max_value) if ssim3d else _ssim(img1, img2, max_value)
			ssims.append(final_ssim)

		# for i in range(img1.shape[2]):
		#     ssims_before.append(_ssim(img1, img2))

		# print('..ssim mean , new {:.4f}  and before {:.4f} .... skimage before {:.4f}'.format(np.array(ssims).mean(), np.array(ssims_before).mean(), skimage_before))
		# ssims.append(skimage.metrics.structural_similarity(img1[..., i], img2[..., i], multichannel=False))

		return np.array(ssims).mean()

	if img1.ndim == 3 and img1.shape[2] == 6:
		l1, r1 = img1[:, :, :3], img1[:, :, 3:]
		l2, r2 = img2[:, :, :3], img2[:, :, 3:]
		return (_cal_ssim(l1, l2) + _cal_ssim(r1, r2)) / 2
	else:
		return _cal_ssim(img1, img2)


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
	avgSSIM_me_255 = 0.0
	avgSSIM_ltm = 0.0
	avgLPIPS_alex = 0.0
	avgLPIPS_vgg = 0.0
	counter = 0

	# lpips
	lpips_score_alex = lpips.LPIPS(net='alex').cuda()
	lpips_score_vgg = lpips.LPIPS(net='vgg').cuda()

	# 加上一个图片保存的路径, 如果没有就创建
	save_dir = './stl_pair'
	os.makedirs(save_dir, exist_ok=True)

	for i, data in enumerate(dataset):
		if i >= opt.how_many:
			break
		counter = i
		# pdb.set_trace()
		model.set_input(data)
		with torch.no_grad():
			model.test() # test里面会针对各个部分进行前向推理，然存储为自己的self变量

		# 模型的输出归一化到（0-1）之间以后计算ssim，这里都还是tensor，本身的输出还是-1，1的tensor
		fake_screen = (model.fake_Bi + 1) / 2.0
		real_screen = (model.real_B + 1) / 2.0
		avgSSIM_me += ssim_standard(fake_screen, real_screen, data_range=1, size_average=True)

		# lpips这些是需要归一化到(-1,1)以后计算的
		lpips_alex = lpips_score_alex(model.fake_Bi, model.real_B)
		lpips_alex = torch.mean(lpips_alex)
		avgLPIPS_alex += lpips_alex

		torch.cuda.empty_cache()


		visuals = model.get_current_visuals() # 这个步骤出来的都变成numpy了，而且是0-255之间
		#avgPSNR += PSNR(visuals['fake_B'],visuals['real_B']) # fake_B是在step1中由AE生成的，fake_Bi是由半影带生成的
		avgPSNR_i += PSNR(visuals['fake_Bi'],visuals['real_B'])
		#avgPSNR_1 += getpsnr(visuals['fake_Bi'],visuals['real_B'])
		#avgSSIM += ssim(visuals['fake_B'],visuals['real_B']) # 图片的范围都是0-255
		avgSSIM_i += ssim(visuals['fake_Bi'],visuals['real_B'])
		#print(visuals['fake_Bi'].shape, visuals['real_B'].shape)
		#avgSSIM_me_255 += ssim_standard(visuals['fake_Bi'], visuals['real_B'], data_range=255, size_average=True)
		avgSSIM_ltm += calculate_ssim_ltm(visuals['fake_Bi'], visuals['real_B'], crop_border=0)




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
		# avgSSIM_me += ssim_standard(fake, real, data_range=255, size_average=True)

		# lpips这些是需要归一化到(-1,1)以后计算的
		# fake = fake / 255.0
		# real = real / 255.0
		# lpips_alex = lpips_score_alex((fake-0.5)/0.5, (real-0.5)/0.5)
		# lpips_alex = torch.mean(lpips_alex)
		# avgLPIPS_alex += lpips_alex
		#avgLPIPS_alex = 0

		# lpips_vgg = lpips_score_vgg((fake - 0.5) / 0.5, (real - 0.5) / 0.5)
		# lpips_vgg = torch.mean(lpips_vgg)
		# avgLPIPS_vgg += lpips_vgg
		#avgLPIPS_vgg = 0

		img_path = model.get_image_paths()
		print('process image... %s' % img_path)
		visualizer.save_images(webpage, visuals, img_path)

		del visuals


	avgPSNR /= counter
	avgSSIM /= counter
	avgPSNR_i /= counter
	avgPSNR_1 /= counter
	avgSSIM_i /= counter
	avgSSIM_me /= counter
	avgSSIM_me_255 /= counter
	avgSSIM_ltm /= counter
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
	print('standard_ssim_255:', avgSSIM_me_255)
	print('ltm_ssim:', avgSSIM_ltm)
	print('ot_ssim:', avgSSIM_i)
	print('lpips_alex:', avgLPIPS_alex)
	print('lpips_vgg:', avgLPIPS_vgg)

	webpage.save()
