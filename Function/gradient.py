import numpy as np
import torch as th
import time
import math
from numba import cuda,float32
from .pyramid import _gaussian_kernel1d
import torch.nn.functional as F
from .Time import getTime

DEFAULT_NUMPY_FLOAT_TYPE = np.float32
DEFAULT_CUDA_FLOAT_TYPE = float32

DEFAULT_TORCH_FLOAT_TYPE = th.float32
DEFAULT_TORCH_COMPLEX_TYPE = th.complex64
DEFAULT_THREADS = 16

def init_ICA(ref_img, options, params):
    """
    Initializes the ICa algorithm by computing the gradients of the reference
    image, and the hessian matrix.

    Parameters
    ----------
    ref_img : device Array[imshape_y, imshape_x]
        Reference image J_1
    options : dict
        verbose options.
    params : dict
        parameters.

    Returns
    -------
    cuda_gradx : Device Array
        horizontal gradients of the reference image
    cuda_grady : Device Array
        vertical gradients of the reference image
    hessian : Device Array
        hessian matrix defined for each patch of the reference image.

    """
    current_time, verbose_3 = time.perf_counter(), options['verbose'] >= 3

    sigma_blur = params['tuning']['sigma blur']
    tile_size = params['tuning']['tileSize']

    imsize_y, imsize_x = ref_img.shape

    # image is padded during BM, we need to consider that to count patches

    n_patch_y = math.ceil(imsize_y / tile_size)
    n_patch_x = math.ceil(imsize_x / tile_size)

    # Estimating gradients with Prewitt kernels
    kernely = np.array([[-1],
                        [0],
                        [1]])

    kernelx = np.array([[-1, 0, 1]])

    # translating ref_img numba pointer to pytorch
    # the type needs to be explicitely specified. Filters need to be casted to float to perform convolution
    # on float image
    th_ref_img = th.as_tensor(ref_img, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
    th_kernely = th.as_tensor(kernely, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]
    th_kernelx = th.as_tensor(kernelx, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]

    # adding 2 dummy dims for batch, channel, to use torch convolve
    if sigma_blur != 0:
        # This is the default kernel of scipy gaussian_filter1d
        # Note that pytorch Convolve is actually a correlation, hence the ::-1 flip.
        # copy to avoid negative stride (not supported by torch)
        gaussian_kernel = _gaussian_kernel1d(sigma=sigma_blur, order=0, radius=int(4 * sigma_blur + 0.5))[::-1].copy()
        th_gaussian_kernel = th.as_tensor(gaussian_kernel, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]

        # 2 times gaussian 1d is faster than gaussian 2d
        temp = F.conv2d(th_ref_img, th_gaussian_kernel[:, None], padding='same')  # convolve y
        temp = F.conv2d(temp, th_gaussian_kernel[None, :], padding='same')  # convolve x

        th_gradx = F.conv2d(temp, th_kernelx, padding='same').squeeze()  # 1 batch, 1 channel
        th_grady = F.conv2d(temp, th_kernely, padding='same').squeeze()

    else:
        th_gradx = F.conv2d(th_ref_img, th_kernelx, padding='same').squeeze()  # 1 batch, 1 channel
        th_grady = F.conv2d(th_ref_img, th_kernely, padding='same').squeeze()

    # swapping grads back to numba
    cuda_gradx = cuda.as_cuda_array(th_gradx)
    cuda_grady = cuda.as_cuda_array(th_grady)

    if verbose_3:
        cuda.synchronize()
        current_time = getTime(
            current_time, ' -- Gradients estimated')

    hessian = cuda.device_array((n_patch_y, n_patch_x, 2, 2), DEFAULT_NUMPY_FLOAT_TYPE)

    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)

    blockspergrid_x = math.ceil(n_patch_x / threadsperblock[1])
    blockspergrid_y = math.ceil(n_patch_y / threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    compute_hessian[blockspergrid, threadsperblock](cuda_gradx, cuda_grady,
                                                    tile_size, hessian)

    if verbose_3:
        cuda.synchronize()
        current_time = getTime(
            current_time, ' -- Hessian estimated')

    return cuda_gradx, cuda_grady, hessian


@cuda.jit
def compute_hessian(gradx, grady, tile_size, hessian):
    imshape = gradx.shape
    patch_idx, patch_idy = cuda.grid(2)
    n_patchy, n_patch_x, _, _ = hessian.shape

    # discarding non existing patches
    if not (patch_idy < n_patchy and
            patch_idx < n_patch_x):
        return

    patch_pos_idx = tile_size * patch_idx  # global position on the coarse grey grid. Because of extremity padding, it can be out of bound
    patch_pos_idy = tile_size * patch_idy

    local_hessian = cuda.local.array((2, 2), DEFAULT_CUDA_FLOAT_TYPE)
    local_hessian[0, 0] = 0
    local_hessian[0, 1] = 0
    local_hessian[1, 0] = 0
    local_hessian[1, 1] = 0

    for i in range(tile_size):
        for j in range(tile_size):
            pixel_global_idy = patch_pos_idy + i
            pixel_global_idx = patch_pos_idx + j

            if not (0 <= pixel_global_idy < imshape[0] and
                    0 <= pixel_global_idx < imshape[1]):
                continue

            local_gradx = gradx[pixel_global_idy, pixel_global_idx]
            local_grady = grady[pixel_global_idy, pixel_global_idx]

            local_hessian[0, 0] += local_gradx * local_gradx
            local_hessian[0, 1] += local_gradx * local_grady
            local_hessian[1, 0] += local_gradx * local_grady
            local_hessian[1, 1] += local_grady * local_grady

    hessian[patch_idy, patch_idx, 0, 0] = local_hessian[0, 0]
    hessian[patch_idy, patch_idx, 0, 1] = local_hessian[0, 1]
    hessian[patch_idy, patch_idx, 1, 0] = local_hessian[1, 0]
    hessian[patch_idy, patch_idx, 1, 1] = local_hessian[1, 1]