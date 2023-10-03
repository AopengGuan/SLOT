import time
import numpy as np
from numba import cuda,float32
import torch.nn.functional as F
import torch as th
from .Time import getTime

DEFAULT_NUMPY_FLOAT_TYPE = np.float32
DEFAULT_CUDA_FLOAT_TYPE = float32

DEFAULT_TORCH_FLOAT_TYPE = th.float32
DEFAULT_TORCH_COMPLEX_TYPE = th.complex64
DEFAULT_THREADS = 16

def init_block_matching(ref_img, options, params):
    '''
    Returns the pyramid representation of ref_img, that will be used for
    future block matching

    Parameters
    ----------
    ref_img : device Array[imshape_y, imshape_x]
        Reference image J_1
    options : dict
        options.
    params : dict
        parameters.

    Returns
    -------
    referencePyramid : list [device Array]
        pyramid representation of the image

    '''
    # Initialization.
    h, w = ref_img.shape  # height and width should be identical for all images

    tileSize = params['tuning']['tileSizes'][0]

    # if needed, pad images with zeros so that getTiles contains all image pixels
    paddingPatchesHeight = (tileSize - h % (tileSize)) * (h % (tileSize) != 0)
    paddingPatchesWidth = (tileSize - w % (tileSize)) * (w % (tileSize) != 0)

    # combine the two to get the total padding
    paddingTop = 0
    paddingBottom = paddingPatchesHeight
    paddingLeft = 0
    paddingRight = paddingPatchesWidth

    # pad all images (by mirroring image edges)
    # separate reference and alternate images

    th_ref_img = th.as_tensor(ref_img, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]

    th_ref_img_padded = F.pad(th_ref_img, (paddingLeft, paddingRight, paddingTop, paddingBottom), 'circular')

    # For convenience
    currentTime, verbose = time.perf_counter(), options['verbose'] > 2
    # factors, tileSizes, distances, searchRadia and subpixels are described fine-to-coarse
    factors = params['tuning']['factors']

    # construct 4-level coarse-to fine pyramid of the reference

    referencePyramid = hdrplusPyramid(th_ref_img_padded, factors)
    if verbose:
        currentTime = getTime(currentTime, ' --- Create ref pyramid')

    return referencePyramid


def hdrplusPyramid(image, factors=[1, 2, 4, 4], kernel='gaussian'):
    '''Construct 4-level coarse-to-fine gaussian pyramid
    as described in the HDR+ paper and its supplement (Section 3.2 of the IPOL article).
    Args:
            image: input image (expected to be a grayscale image downsampled from a Bayer raw image)
            factors: [int], dowsampling factors (fine-to-coarse)
            kernel: convolution kernel to apply before downsampling (default: gaussian kernel)'''
    # Start with the finest level computed from the input
    pyramidLevels = [cuda_downsample(image, kernel, factors[0])]
    # pyramidLevels = [downsample(image, kernel, factors[0])]

    # Subsequent pyramid levels are successively created
    # with convolution by a kernel followed by downsampling
    for factor in factors[1:]:
        pyramidLevels.append(cuda_downsample(pyramidLevels[-1], kernel, factor))
        # print('the last pyramid shape:', pyramidLevels[-1].shape)

    # torch to numba, remove batch, channel dimensions
    for i, pyramidLevel in enumerate(pyramidLevels):
        pyramidLevels[i] = cuda.as_cuda_array(pyramidLevel.squeeze())

    # Reverse the pyramid to get it coarse-to-fine
    return pyramidLevels[::-1]


def cuda_downsample(th_img, kernel='gaussian', factor=2):
    '''Apply a convolution by a kernel if required, then downsample an image.
    Args:
     	image: Device Array the input image (WARNING: single channel only!)
     	kernel: None / str ('gaussian' / 'bayer') / 2d numpy array
     	factor: downsampling factor
    '''
    # print(th_img.shape)
    # Special case
    if factor == 1:
        gaussian_kernel = _gaussian_kernel1d(sigma=factor * 0.5, order=0, radius=int(4 * factor * 0.5 + 0.5))[
                          ::-1].copy()
        th_gaussian_kernel = th.as_tensor(gaussian_kernel, dtype=th.float32, device="cuda")

        # 2 times gaussian 1d is faster than gaussian 2d·
        temp = F.conv2d(th_img, th_gaussian_kernel[None, None, :, None])  # convolve y
        th_filteredImage = F.conv2d(temp, th_gaussian_kernel[None, None, None, :])  # convolve x
        return th_filteredImage
        #return th_img

    # Filter the image before downsampling it
    if kernel is None:
        raise ValueError('use Kernel')
    elif kernel == 'gaussian':
        # gaussian kernel std is proportional to downsampling factor
        # This is the default kernel of scipy gaussian_filter1d
        # Note that pytorch Convolve is actually a correlation, hence the ::-1 flip.
        # copy to avoid negative stride
        gaussian_kernel = _gaussian_kernel1d(sigma=factor * 0.5, order=0, radius=int(4 * factor * 0.5 + 0.5))[
                          ::-1].copy()
        th_gaussian_kernel = th.as_tensor(gaussian_kernel, dtype=th.float32, device="cuda")

        # 2 times gaussian 1d is faster than gaussian 2d
        temp = F.conv2d(th_img, th_gaussian_kernel[None, None, :, None])  # convolve y
        th_filteredImage = F.conv2d(temp, th_gaussian_kernel[None, None, None, :])  # convolve x
    else:
        raise ValueError("please use gaussian kernel")

    # Shape of the downsampled image
    # print('filter image shape:',np.array(th_filteredImage.shape[2:]))
    h2, w2 = np.floor(np.array(th_filteredImage.shape[2:]) / float(factor)).astype(int)
    # print(th_filteredImage.shape[2:])
    # print(h2,w2)

    return th_filteredImage[:, :, :h2 * factor:factor, :w2 * factor:factor]


def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order) / -sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x