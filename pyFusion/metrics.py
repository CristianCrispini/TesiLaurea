import numpy as np
from math import exp, pi
from scipy.ndimage import gaussian_filter
from cv2 import cartToPolar, Sobel, CV_32F

EPS = np.finfo(float).eps
# xydeas_petrovic parameters
# The constants Γ, κ , σ  and Γα, κα, σα determine 
# the  exact  shape  of  the  sigmoid  functions  used  to  form  the  edge  strength  and  
# orientation  preservation  values,
GAMMA1 = 1
GAMMA2 = 1
K1 = -10
K2 = -20
DELTA1 = 0.5
DELTA2 = 0.75
L = 1
#--------------------------
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

def _perceptual_loss(gA, gF, alphaA, alphaF):
    # If g o alpha are followed by an underscore are ment to be considered written in uppercase
    # The relative strength and orientation values of g_AF(n,m) and alpha_AF(n,m) of an input 
    # image A with respect to F are formed as:
    if (sum( sum (gA )) > sum( sum( gF ))):
        g_AF = gF / ( gA + EPS)
    else:
        g_AF = gA / (gF + EPS)
    
    alpha_AF = np.abs( np.abs(alphaA - alphaF) - pi/2) / (pi/2)

    qG_AF = GAMMA1 / (1 + np.exp( K1 *(g_AF - DELTA1)))
    qalpha_AF = GAMMA2 / (1 + np.exp( K2 *(alpha_AF - DELTA2) ))
    # These are used to derive the edge strength and orientation preservation values
    # QgAF(n,m)  and  QαAF(n,m)  model  perceptual  loss  of  information  in  F,  in  terms  of  
    # how well the strength and orientation values of a pixel p(n,m) in A are 
    # represented in the fused image. 
    #
    # Edge  information preservation values are then defined as
    q_AF = qG_AF * qalpha_AF
    # with  0  ≤  Q AF(n,m)  ≤  1 .  A  value  of  0  corresponds  to  the  complete  loss  of  edge  
    # information, at location (n,m), as transferred from A into F. QAF(n,m)=1 indicates 
    # “fusion” from A to F with no loss of information. 
    return q_AF

# Gradient information preservation estimates
def xydeas_petrovic_metric(image1, image2, fusedImage):
    # EDGE Strenght and orientation for each pixels of the input images
    gA, alphaA = _strenght_n_orientation(image1)
    gB, alphaB = _strenght_n_orientation(image2)
    gF, alphaF = _strenght_n_orientation(fusedImage)
    
    q_AF = _perceptual_loss(gA, gF, alphaA, alphaF)
    q_BF = _perceptual_loss(gB, gF, alphaB, alphaF)
    #
    # In general edge preservation values which 
    # correspond to pixels with high edge strength, should influence normalised weighted  
    # performance metric QP more than 
    # those of relatively low edge strength.Thus, wA(n,m)=[gA(n,m)]^L and 
    # wB(n,m)=[gB(n,m)]^L where L is a constant.
    #
    wA = np.linalg.matrix_power(gA, L)
    wB = np.linalg.matrix_power(gB, L)
    # normalised weighted performance metric QP
    qP_ABF = sum( sum((q_AF * wA + q_BF * wB))) / sum ( sum((wA + wB)))
    return qP_ABF

def xydeas_petrovic_total_fusion_gain(image1, image2, fusedImage):
    gA, alphaA = _strenght_n_orientation(image1)
    gB, alphaB = _strenght_n_orientation(image2)
    gF, alphaF = _strenght_n_orientation(fusedImage)
    
    q_AF = _perceptual_loss(gA, gF, alphaA, alphaF)
    q_BF = _perceptual_loss(gB, gF, alphaB, alphaF)

    wA = np.linalg.matrix_power(gA, L)
    wB = np.linalg.matrix_power(gB, L)
    
    # local exclusive information in F, Q_delta
    # quantifies the total amount of local
    # exclusive information across the fused image.
  
    q_delta = np.abs(q_AF - q_BF)
  
    # For locations with strong correlation between the inputs Q_delta 
    # will be small or zero, indicating no exclusive
    # information. Conversely, in areas where one of the
    # inputs provides a meaningful feature that is not present
    # in the other this quantity will tend towards 1.

    # The common information component for all locations across the fused image
    q_common = (q_AF + q_BF  - q_delta) / 2
    # ½ is introduced as common information is contained in both Q_AF and Q_BF

    # Local estimates of exclusive information
    # components of each input
    q_delta_AF = q_AF - q_common
    # is the proportion of useful information fused in F that exists only in A
    q_delta_BF = q_BF - q_common
    # is the proportion of useful information fused in F that exists only in B

    # These quantities represent effectively, local fusion gain 
    # achieved by fusing A and B with respect to each individual {A, B}.
    
    # TOTAL FUSION GAIN
    return sum( sum((q_delta_AF * wA + q_delta_BF * wB))) / sum ( sum((wA + wB)))

def xydeas_petrovic_fusion_loss(image1, image2, fusedImage):
    # Fusion loss loss_ABF is a measure of the information lost
    # during the fusion process. This is information available
    # in the input images but not in the fused image
    gA, alphaA = _strenght_n_orientation(image1)
    gB, alphaB = _strenght_n_orientation(image2)
    gF, alphaF = _strenght_n_orientation(fusedImage)
    
    q_AF = _perceptual_loss(gA, gF, alphaA, alphaF)
    q_BF = _perceptual_loss(gB, gF, alphaB, alphaF)

    wA = np.linalg.matrix_power(gA, L)
    wB = np.linalg.matrix_power(gB, L)
    
    gA = sum( sum( gA))
    gB = sum( sum( gB))
    gF = sum( sum( gF))

    # if gradient strength in F is larger than
    # that in the inputs, F contains artifacts; conversely, a
    # weaker gradient in F indicates a loss of input
    # information.
    if ( gF < gA ) or ( gF < gB ):
        r = 1
    else:
        r = 0

    return sum( sum(r * ((1 - q_AF) * wA + (1 - q_BF) * wB))) / sum ( sum((wA + wB)))