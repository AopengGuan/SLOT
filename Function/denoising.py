
from numba import cuda,float32
import numpy as np
import torch as th
import math

DEFAULT_NUMPY_FLOAT_TYPE = np.float32
DEFAULT_CUDA_FLOAT_TYPE = float32

DEFAULT_TORCH_FLOAT_TYPE = th.float32
DEFAULT_TORCH_COMPLEX_TYPE = th.complex64
DEFAULT_THREADS = 16



def frame_count_denoising_median(image, r_acc, params):
    # TODO it may be useless to bother defining this function for grey images
    denoised = cuda.device_array(image.shape, DEFAULT_NUMPY_FLOAT_TYPE)

    grey_mode = params['mode'] == 'grey'
    scale = params['scale']
    radius_max = params['radius max']
    max_frame_count = params['max frame count']

    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS, 1)
    blockspergrid_x = math.ceil(denoised.shape[1 ] /threadsperblock[1])
    blockspergrid_y = math.ceil(denoised.shape[0 ] /threadsperblock[0])
    blockspergrid_z = math.ceil(denoised.shape[2 ] /threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    cuda_frame_count_denoising_median[blockspergrid, threadsperblock](
        image, denoised, r_acc,
        scale, radius_max, max_frame_count, grey_mode)

    return denoised

@cuda.jit
def cuda_frame_count_denoising_median(noisy, denoised, r_acc,
                                      scale, radius_max, max_frame_count, grey_mode):
    x, y, c = cuda.grid(3)
    imshape_y, imshape_x, _ = noisy.shape

    if not (0 <= y < imshape_y and
            0 <= x < imshape_x):
        return

    if grey_mode:
        y_grey = int(round( y /scale))
        x_grey = int(round( x /scale))
    else:
        y_grey = int(round(( y -0.5 ) /( 2 *scale)))
        x_grey = int(round(( x -0.5 ) /( 2 *scale)))

    r = r_acc[y_grey, x_grey]
    radius = denoise_power_median(r, radius_max, max_frame_count)
    radius = min(14, radius) # for memory purpose


    buffer = cuda.local.array(16 *16, DEFAULT_CUDA_FLOAT_TYPE)
    k = 0
    for i in range(-radius, radius +1):
        for j in range(-radius, radius +1):
            x_ = x + j
            y_ = y + i
            if (0 <= y_ < imshape_y and
                    0 <= x_ < imshape_x):
                buffer[k] = noisy[y_, x_, c]
                k += 1

    bubble_sort(buffer[:k])


    denoised[y, x, c] = buffer[ k//2]


@cuda.jit(device=True)
def denoise_power_median(r_acc, radius_max, max_frame_count):
    r = min(r_acc, max_frame_count)
    return round(radius_max * (max_frame_count - r ) /max_frame_count)

@cuda.jit(device=True)
def bubble_sort(X):
    N = X.size

    for i in range( N -1):
        for j in range( N - i -1):
            if X[j] > X[ j +1]:
                X[j], X[ j +1] = X[ j +1], X[j]


def frame_count_denoising_gauss(image, r_acc, params):
    # TODO it may be useless to bother defining this function for grey images
    denoised = cuda.device_array(image.shape, DEFAULT_NUMPY_FLOAT_TYPE)

    grey_mode = params['mode'] == 'grey'
    scale = params['scale']
    sigma_max = params['sigma max']
    max_frame_count = params['max frame count']

    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS, 1)
    blockspergrid_x = math.ceil(denoised.shape[1] / threadsperblock[1])
    blockspergrid_y = math.ceil(denoised.shape[0] / threadsperblock[0])
    blockspergrid_z = math.ceil(denoised.shape[2] / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    cuda_frame_count_denoising_gauss[blockspergrid, threadsperblock](
        image, denoised, r_acc,
        scale, sigma_max, max_frame_count, grey_mode)

    return denoised


@cuda.jit
def cuda_frame_count_denoising_gauss(noisy, denoised, r_acc,
                                     scale, sigma_max, max_frame_count, grey_mode):
    x, y, c = cuda.grid(3)
    imshape_y, imshape_x, _ = noisy.shape

    if not (0 <= y < imshape_y and
            0 <= x < imshape_x):
        return

    if grey_mode:
        y_grey = int(round(y / scale))
        x_grey = int(round(x / scale))
    else:
        y_grey = int(round((y - 0.5) / (2 * scale)))
        x_grey = int(round((x - 0.5) / (2 * scale)))

    r = r_acc[y_grey, x_grey]
    sigma = denoise_power_gauss(r, sigma_max, max_frame_count)

    t = 3 * sigma

    num = 0
    den = 0
    for i in range(-t, t + 1):
        for j in range(-t, t + 1):
            x_ = x + j
            y_ = y + i
            if (0 <= y_ < imshape_y and
                    0 <= x_ < imshape_x):
                if sigma == 0:
                    w = (i == j == 0)
                else:
                    w = math.exp(-(j * j + i * i) / (2 * sigma * sigma))
                num += w * noisy[y_, x_, c]
                den += w

    denoised[y, x, c] = num / den


@cuda.jit(device=True)
def denoise_power_gauss(r_acc, sigma_max, r_max):
    r = min(r_acc, r_max)
    return sigma_max * (r_max - r) / r_max