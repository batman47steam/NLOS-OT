import os
import cv2
import numpy as np
import torch
from pytorch_msssim import ssim as ssim_standard

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

def read_image_sequence(directory):
    image_files = sorted(os.listdir(directory))
    results = []
    gts = []

    for img_file in image_files:
        img_path = os.path.join(directory, img_file)
        img = cv2.imread(img_path)

        if img is not None:
            height, width, _ = img.shape
            num_images = width // 256

            if num_images == 2:  # 2 images concatenated
                result = img[:, :width // 2, :]
                gt = img[:, width // 2:, :]
            elif num_images == 3:  # 3 images concatenated
                result = img[:, :width // 3, :]
                gt = img[:, width // 3:2 * width // 3, :]
            else:
                continue  # Skip images that don't match expected formats

            #cv2.imshow('gt', gt)
            #cv2.waitKey(0)

            results.append(result)
            gts.append(gt)

    return results, gts


def calculate_average_ssim(results, gts):
    ssim_values = []
    ltm_values = []


    for result, gt in zip(results, gts):
        # Convert images to grayscale for SSIM calculation
        #result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        #gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

        # 都从numpy转变为tensor
        result = result.transpose(2,0,1)
        result = torch.from_numpy(result).unsqueeze(0).double()

        gt = gt.transpose(2,0,1)
        gt = torch.from_numpy(gt).unsqueeze(0).double()

        ssim = ssim_standard(result, gt, data_range=255, size_average=True)

        ssim_values.append(ssim)

        ltm = calculate_ssim_ltm(result, gt, crop_border=0)
        ltm_values.append(ltm)

    average_ssim = np.mean(ssim_values)
    average_ltm = np.mean(ltm_values)
    return average_ssim, average_ltm


# Example usage
directory = 'MNIST_pair'
results, gts = read_image_sequence(directory)
average_ssim, average_ltm = calculate_average_ssim(results, gts)

print(f"Average SSIM: {average_ssim}, Average LTM: {average_ltm}")
