from cv2 import imread # type: ignore
from pyFusion.fusion import Fusion
from pyFusion.imageQualityAssestment import IQA

import matplotlib.pyplot as plt


from skimage.metrics import structural_similarity as ssim # type: ignore
from skimage.metrics import mean_squared_error # type: ignore
from skimage.color import rgb2gray # type: ignore

from pyFusion.vgg19 import VGG19
from pyFusion.squeezeNet import Squeeze
from torch import device
from torch.cuda import is_available

import numpy as np

def entropy_2d( x , y):
    EPS = np.finfo(float).eps
    """
    Computes entropy between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    Returns
    -------
    en: floathttps://github.com/CristianCrispini/TesiLaurea
        the computed entropy measure
    """
    bins = (256, 256)



    jh,_,__ = np.histogram2d(x, y, bins=bins)[0]

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    pi = jh / sh
    EN = -1 * np.sum(pi * np.log(pi))
    return EN






# Parse arguments
images = ['images/MRI-CT/mr.png', 'images/MRI-CT/ct.png']
output = 'results/MRI-CT/'

# Read images
input_images = []
for image in images:
    input_images.append(imread(image))



# Compute fusion image

device = device("cuda" if is_available() else "cpu")
model = VGG19(device)
#model = Squeeze(device)

FU = Fusion(input_images, model)
fused_image = FU.fuse()

img1 = rgb2gray(input_images[0])
img2 = rgb2gray(input_images[1])
fused_image_g = rgb2gray(fused_image)

#iqa_model = IQA(input_images)

# ENTROPY
print(img1.shape)
print(img2.shape)
print(fused_image_g.shape)

EN_af = entropy_2d(img1, fused_image)
EN_bf = 0#iqa_model.entropy_2d(img2, fused_image)

EN_F = 0#(EN_af + EN_bf) / 2


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4),
                         sharex=True, sharey=True)
ax = axes.ravel()

mse_none = mean_squared_error(img1, fused_image)
ssim_none = ssim(img1, fused_image, data_range=fused_image.max() - fused_image.min())

mse_noise = mean_squared_error(img2, fused_image)
ssim_noise = ssim(img2, fused_image,
                  data_range=fused_image.max() - fused_image.min())

mse_const = 0
ssim_const = 1

label = 'MSE: {:.2f}, SSIM: {:.2f}, EN:{:.2f}'

ax[0].imshow(input_images[0], cmap=plt.cm.gray, vmin=0, vmax=1)
ax[0].set_xlabel(label.format(mse_none, ssim_none, EN_af))
ax[0].set_title('Original image 1')

ax[1].imshow(input_images[1], cmap=plt.cm.gray, vmin=0, vmax=1)
ax[1].set_xlabel(label.format(mse_noise, ssim_noise, EN_bf))
ax[1].set_title('Original image 2')

ax[2].imshow(fused_image_g, cmap=plt.cm.gray)
ax[2].set_xlabel(label.format(mse_const, ssim_const, EN_F))
ax[2].set_title('Fused Image')

plt.tight_layout()
plt.show()