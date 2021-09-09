
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error 
from skimage import img_as_float 
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from cv2 import cartToPolar, Sobel, CV_32F



class IQA():
    EPS = ''
    images = []
    def __init__(self, input_images):
        self.EPS = np.finfo(float).eps
        self.images = input_images

    def mutual_information_2d(self, x, y, sigma=1, normalized=True):
        """
        Computes (normalized) mutual information between two 1D variate from a
        joint histogram.
        Parameters
        ----------
        x : 1D array
            first variable
        y : 1D array
            second variable
        sigma: float
            sigma for Gaussian smoothing of the joint histogram
        Returns
        -------
        nmi: float
            the computed similariy measure
        """
        bins = (256, 256)



        jh = np.histogram2d(x, y, bins=bins)[0]

        # smooth the jh with a gaussian filter of given sigma
        gaussian_filter(jh, sigma=sigma, mode='constant',
                                        output=jh)

        # compute marginal histograms
        jh = jh + self.EPS
        sh = np.sum(jh)
        jh = jh / sh
        s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
        s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

        # Normalised Mutual Information of:
        # Studholme,  jhill & jhawkes (1998).
        # "A normalized entropy measure of 3-D medical image alignment".
        # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
        if normalized:
            mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                    / np.sum(jh * np.log(jh))) - 1
        else:
            mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
                    - np.sum(s2 * np.log(s2)))

        return mi
    
    def discrepancy(self, img1, img2):
        P, Q = img1.shape
        diff = img1 - img2
        discrepancy = np.sum(np.absolute(diff)) / P * Q

        return discrepancy

    def _sobel_edge_detection(image, verbose=False):
        #gx = convolution(image, filter)
        sx = Sobel(image, CV_32F, 1, 0)
        sy = Sobel(image, CV_32F, 0, 1)

        if verbose:
            plt.imshow(sx, cmap='gray')
            plt.title("Horizontal Edge")
            plt.show()
    
            plt.imshow(sy, cmap='gray')
            plt.title("Vertical Edge")
            plt.show()
    
        return cartToPolar(sx, sy)

    def entropy_2d( self, x , y):
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
        en: float
            the computed entropy measure
        """
        bins = (256, 256)



        jh,_,__ = np.histogram2d(x, y, bins=bins)[0]

        # compute marginal histograms
        jh = jh + self.EPS
        sh = np.sum(jh)
        pi = jh / sh
        EN = -1 * np.sum(pi * np.log(pi))
        return EN

#def entropy2dss(img1, img2, fused_img):
#   EN = _entropy_2d(img1.ravel(), fused_img.ravel())
#  EN2 = _entropy_2d(img2.ravel(), fused_img.ravel())
#  return (EN + EN2) / 2

