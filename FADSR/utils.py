import cv2
import math
from datasets import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.utils import save_image
import copy


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ----------
# PSNR
# ----------
def calculate_psnr(cal_sr, cal_hr, train_mean, train_std):
    """
    Calculate PSNR
    :param cal_sr: Model generated high-resolution images
    :param cal_hr: Original high-resolution image
    :param train_mean: Mean value of the data set
    :param train_std: Standard deviation of the data set
    :return:
    """
    # model_hr and imgs_hr have range [0, 255]
    if not cal_sr.shape == cal_hr.shape:
        raise ValueError('Input images must have the same dimensions.')

    # The image tensor is first denormalized using the mean and standard values
    # Then the values are scaled to 0-255
    cal_sr = denormalize(cal_sr, train_mean, train_std) * 255.0

    cal_hr = denormalize(cal_hr, train_mean, train_std) * 255.0

    mse = ((cal_sr - cal_hr) ** 2).mean()
    if mse == 0:
        # positive infinity
        return float('inf')

    return 20 * math.log10(255.0 / math.sqrt(mse))


# ----------
# SSIM
# ----------
def calculate_ssim(cal_sr, cal_hr, train_mean, train_std):
    """
    Calculate SSIM
    """
    if not cal_sr.shape == cal_hr.shape:
        raise ValueError('Input images must have the same dimensions.')

    # Clear redundant dimensions (1, 3, 100, 100)->(3, 100, 100)
    cal_sr = cal_sr.squeeze()
    cal_hr = cal_hr.squeeze()

    # The image tensor is first denormalized using the mean and standard values
    # Then the values are scaled to 0-255 and then converted to Numpy
    cal_sr = (denormalize(cal_sr, train_mean, train_std) * 255.0).numpy()
    cal_hr = (denormalize(cal_hr, train_mean, train_std) * 255.0).numpy()

    if cal_sr.ndim == 2:
        return ssim(cal_sr, cal_hr)
    elif cal_sr.ndim == 3:
        ssims = []
        for i in range(3):
            ssims.append(ssim(cal_sr[i], cal_hr[i]))
        return np.array(ssims).mean()
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(imgs_sr, imgs_hr):
    """
    SSIM calculation
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    imgs_sr = imgs_sr.astype(np.float64)
    imgs_hr = imgs_hr.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(imgs_sr, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(imgs_hr, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(imgs_sr ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(imgs_hr ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(imgs_sr * imgs_hr, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# ----------
# Calculate the loss, PSNR, SSIM for each epoch
# ----------
def validationOrTest(dataset_path, hr_shape, scale, model, criterion_pixel, epoch):
    # Instance initialization
    losses = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()

    # Set the model to evaluation mode
    model.eval()

    # Load validation data or test data
    valid_mean, valid_std = MeanStd("F:/data/%s/" % dataset_path)
    dataloader = DataLoader(
        ImageDataset("F:/data/%s" % dataset_path, hr_shape, scale, valid_mean, valid_std),
        batch_size=1,
        shuffle=True,
    )

    # Tensor Type
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    # Predict results without calculating gradients
    with torch.no_grad():
        for i, imgs in enumerate(dataloader):
            # LR-HR image pairs
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))

            imgs_sr = model(imgs_lr)

            # Loss
            loss_pixel = criterion_pixel(imgs_sr, imgs_hr)
            loss_model = loss_pixel

            # Update the entire epoch for current losses
            losses.update(loss_model.item(), hr_shape[0])

            # Update PSNR
            cal_psnr_sr = copy.deepcopy(imgs_sr)
            cal_psnr_hr = copy.deepcopy(imgs_hr)
            psnr.update(calculate_psnr(cal_psnr_sr, cal_psnr_hr, valid_mean, valid_std))

            # Update SSIM
            cal_ssim_sr = copy.deepcopy(imgs_sr)
            cal_ssim_hr = copy.deepcopy(imgs_hr)
            ssim.update(calculate_ssim(cal_ssim_sr, cal_ssim_hr, valid_mean, valid_std))

            if epoch == 199 or epoch == 299:
                # Name of the image
                img_name = imgs['path'][0].rsplit("\\")[-1]

                # Interpolation
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=scale)
                # Normalize to 0-1 using mean and std
                imgs_lr_01 = denormalize(imgs_lr, valid_mean, valid_std)
                save_image(imgs_lr_01, "images_valid/Upsampling_%d_%s.tif" % (epoch, img_name),
                           normalize=False)
                # Save model-generated high-resolution image
                SR_01 = denormalize(imgs_sr, valid_mean, valid_std)
                save_image(SR_01, "images_valid/SR_%d_%s.tif" % (epoch, img_name), normalize=False)

                # Original high-resolution image
                HR_01 = denormalize(imgs_hr, valid_mean, valid_std)  # 利用mean和std归一化到0-1
                save_image(HR_01, "images_valid/HR_%d_%s.tif" % (epoch, img_name), normalize=False)

    return losses.avg, psnr.avg, ssim.avg
