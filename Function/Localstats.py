import numpy as np
import time
import torch as th
from numba import cuda,float32
from .Time import getTime
import math
DEFAULT_NUMPY_FLOAT_TYPE = np.float32
DEFAULT_CUDA_FLOAT_TYPE = float32

DEFAULT_TORCH_FLOAT_TYPE = th.float32
DEFAULT_TORCH_COMPLEX_TYPE = th.complex64
DEFAULT_THREADS = 16

def init_robustness(ref_img, options, params):
    """
    Initialiazes the robustness etimation procesdure by
    computing the local stats of the reference image

    Parameters
    ----------
    ref_img : device Array[imshape_y, imshape_x]
        Raw reference image J_1
    options : dict
        options.
    params : dict
        parameters.

    Returns
    -------
    ref_local_stats : device Array[guide_imshape_y, guide_imshape_x, 2, channels]
        local statistics of the reference image. guide_imshape = imshape in grey mode,
        guide_imshape = imshape//2 in bayer mode.

        ref_local_stats[:, :, 0, c] is the mean value of color channel c mu(c)
        ref_local_stats[:, :, 1, c] is the variance (sigma^2) of color channel cs

    """
    imshape_y, imshape_x = ref_img.shape

    bayer_mode = params['mode' ]=='bayer'
    verbose_3 = options['verbose'] >= 3
    r_on = params['on']



    if r_on :
        if verbose_3:
            print(" - Decimating images to RGB")
            current_time = time.perf_counter()

        # Computing guide image

        if bayer_mode:
            #guide_ref_img = compute_guide_image(ref_img, CFA_pattern)
            print('Mode set false')
        else:
            # Numba friendly code to add 1 channel
            guide_ref_img = ref_img.reshape((imshape_y, imshape_x, 1))


        if verbose_3 :
            cuda.synchronize()
            current_time = getTime(
                current_time, ' - Image decimated')

        ref_local_stats = compute_local_stats(guide_ref_img)

        if verbose_3 :
            cuda.synchronize()
            current_time = getTime(current_time, ' - Local stats estimated')

        return ref_local_stats
    else:
        return None

def compute_local_stats(guide_img):
    """
    Implementation of Algorithm 8: ComputeLocalStatistics
    Computes the mean color and variance associated for each 3 by 3 patches of
    the guide image G_n.

    Parameters
    ----------
    guide_img : device Array[guide_imshape_y, guide_imshape_x, channels]
        Guide image G_n.

    Returns
    -------
    ref_local_stats : device Array[guide_imshape_y, guide_imshape_x, 2, channels]
        Array that contains mu and sigma² for every position of the guide image.


    """
    *guide_imshape, n_channels = guide_img.shape
    if n_channels == 1:
        local_stats = cuda.device_array(guide_imshape + [2, 1], DEFAULT_NUMPY_FLOAT_TYPE) # mu, sigma
    elif n_channels == 3:
        local_stats = cuda.device_array(guide_imshape + [2, 3], DEFAULT_NUMPY_FLOAT_TYPE) # mu, sigma for rgb
    else:
        raise ValueError("Incoherent number of channel : {}".format(n_channels))

    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS, 1) # maximum, we may take less
    blockspergrid_x = math.ceil(guide_imshape[1 ] /threadsperblock[1])
    blockspergrid_y = math.ceil(guide_imshape[0 ] /threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y, n_channels)

    cuda_compute_local_stats[blockspergrid, threadsperblock](guide_img, local_stats)

    return local_stats


@cuda.jit
def cuda_compute_local_stats(guide_img, local_stats):
    guide_imshape_y, guide_imshape_x, _ = guide_img.shape

    idx, idy, channel = cuda.grid(3)
    if not(0 <= idy < guide_imshape_y and
           0 <= idx < guide_imshape_x):
        return

    local_stats_ = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    local_stats_[0] = 0
    local_stats_[1] = 0

    for i in range(-1, 2):
        for j in range(-1, 2):
            y = clamp(idy + i, 0, guide_imshape_y -1)
            x = clamp(idx + j, 0, guide_imshape_x -1)

            value = guide_img[y, x, channel]
            local_stats_[0] += value
            local_stats_[1] += value *value


    # normalizing
    channel_mean = local_stats_[0 ] /9
    local_stats[idy, idx, 0, channel] = channel_mean
    local_stats[idy, idx, 1, channel] = local_stats_[1] /9 - channel_mean *channel_mean

@cuda.jit(device=True)
def clamp(x, min_, max_):
    return min(max_, max(min_, x))