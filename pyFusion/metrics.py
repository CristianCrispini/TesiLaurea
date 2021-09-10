import numpy as np
from scipy.ndimage import gaussian_filter
from cv2 import cartToPolar, Sobel, CV_32F, normalize, NORM_MINMAX, filter2D

EPS = np.finfo(float).eps

def mutual_information_2d( origImage, fusedImage, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    origImage : 1D array
        first variable
    fusedImage : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    normalized : boolean
        default False
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    bins = (256, 256)

    jh = np.histogram2d(origImage, fusedImage, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    gaussian_filter(jh, sigma=sigma, mode='constant',
                                    output=jh)

    # compute marginal histograms
    jh = jh + EPS
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

def entropy_2d(origImage,fusedImage):
    """
    Computes entropy between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    origImage : 1D array
        first variable
    fusedImage : 1D array
        second variable
    Returns
    -------
    entropy: float
        the computed entropy measure
    """
    bins = (256, 256)

    jh = np.histogram2d(origImage, fusedImage, bins=bins)[0]

    # compute marginal histograms
    jh = jh + EPS
    
    sh = np.sum(jh)
    pi = jh / sh
    entropy = -1 * np.sum(pi * np.log(pi))
    return entropy

def discrepancy(img1, img2):
    P, Q = img1.shape
    diff = img1 - img2
    d = np.sum(np.absolute(diff)) / P * Q

    return d

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

def _strenght_n_orientation(image):
    #The first input is the source image, which we convert to float. 
    #The second input is the output image, but we'll set that to None as we want the function 
    # call to return that for us. 
    #The third and fourth parameters specify the minimum and maximum values 
    # you want to appear in the output, which is 0 and 1 respectively, 
    #and the last output specifies how you want to normalize the image.
    # What I described falls under the NORM_MINMAX flag.
    #image = normalize(image.astype('float'), None, 0.0, 1.0, NORM_MINMAX)  
    # Kernels for convolving over the images
    #flt1= [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    #flt2= [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    # 1) get the map Sobel operator
    #fuseX = filter2D(image, -1, flt1)
    #fuseY = filter2D(image, -1, flt2)
    #   EQUIVALENT TO:
    s_x, s_y = _sobel_edge_detection(image)
    #fusex
    # A Sobel edge operator is applied to yield the edge strength G
    g = np.sqrt(s_x**2 + s_y**2)
    # Orientation α(n,m) information for each pixel p
    alpha = np.arctan(s_y / ( s_x + EPS))
    return (g, alpha)

def xydeas_petrovic_metric(image1, image2, fusedImage):
    # Strenght and orientation for all the images
    gA, alphaA = _strenght_n_orientation(image1)
    gB, alphaB = _strenght_n_orientation(image2)
    gF, alphaF = _strenght_n_orientation(fusedImage)
