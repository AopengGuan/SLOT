import time
import numpy as np
from numba import cuda,float32
import math
import torch as th
from .Time import getTime

DEFAULT_NUMPY_FLOAT_TYPE = np.float32
DEFAULT_CUDA_FLOAT_TYPE = float32

DEFAULT_TORCH_FLOAT_TYPE = th.float32
DEFAULT_TORCH_COMPLEX_TYPE = th.complex64
DEFAULT_THREADS = 16



def ICA_optical_flow(cuda_im_grey, cuda_ref_grey,
                     cuda_gradx, cuda_grady,
                     hessian, cuda_pre_alignment,
                     options, params, debug=False):
    """ Computes optical flow between the ref_img and all images of comp_imgs
    based on the ICA method http://www.ipol.im/pub/art/2016/153/
    The optical flow follows a translation per patch model, such that :
    ref_img(X) ~= comp_img(X + flow(X))


    Parameters
    ----------
    cuda_img_grey : device Array[imsize_y, imsize_x]
        Image to align on grey level G_n
    cuda_ref_grey : device Array[imsize_y, imsize_x]
        Reference image on grey level G_1
    cuda_gradx : device array[imsize_y, imsize_x]
        Horizontal gradient of the reference image
    cuda_grady : device array[imsize_y, imsize_x]
        Vertical gradient of the reference image
    hessian : device_array[n_tiles_y, n_tiles_x, 2, 2]
        Hessian matrix of the reference image
    cuda_pre_alignment : device Array[n_tiles_y, n_tiles_x, 2]
        optical flow for each tile of each image, outputed by bloc matching : V_n
        pre_alignment[0] must be the horizontal flow oriented towards the right if positive.
        pre_alignment[1] must be the vertical flow oriented towards the bottom if positive.
    options : dict
        options
    params : dict
        ['tuning']['kanadeIter'] : int
            Number of iterations.
        params['tuning']['tileSize'] : int
            Size of the tiles.
        params["mode"] : {"bayer", "grey"}
            Mode of the pipeline : whether the original burst is grey or raw

        params['tuning']['sigma blur'] : float
            If non zero, applies a gaussian blur before computing gradients.

    debug : bool, optional
        If True, this function returns a list containing the flow at each iteration.
        The default is False.

    Returns
    -------
    cuda_alignment : device_array[n_tiles_y, n_tiles_x, 2]
        Updated alignment vectors V_n(p) for each tile of the image

    """
    if debug:
        debug_list = []

    n_iter = params['tuning']['kanadeIter']

    cuda_alignment = cuda_pre_alignment

    for iter_index in range(n_iter):
        ICA_optical_flow_iteration(
            cuda_ref_grey, cuda_gradx, cuda_grady, cuda_im_grey, cuda_alignment, hessian,
            options, params, iter_index)

        if debug:
            debug_list.append(cuda_alignment.copy_to_host())

    if debug:
        return debug_list
    return cuda_alignment


def ICA_optical_flow_iteration(ref_img, gradsx, gradsy, comp_img, alignment, hessian, options, params,
                               iter_index):
    """
    Computes one iteration of the Lucas-Kanade optical flow

    Parameters
    ----------
    ref_img : Array [imsize_y, imsize_x]
        Ref image (grey)
    gradx : Array [imsize_y, imsize_x]
        Horizontal gradient of the ref image
    grady : Array [imsize_y, imsize_x]
        Vertical gradient of the ref image
    comp_img : Array[imsize_y, imsize_x]
        The image to rearrange and compare to the reference (grey images)
    alignment : Array[n_tiles_y, n_tiles_x, 2]
        The inial alignment of the tiles
    options : Dict
        Options to pass
    params : Dict
        parameters
    iter_index : int
        The iteration index (for printing evolution when verbose >2,
                             and for clearing memory)

    """
    verbose_3 = options['verbose'] >= 3
    tile_size = params['tuning']['tileSize']

    n_patch_y, n_patch_x, _ = alignment.shape

    if verbose_3:
        cuda.synchronize()
        current_time = time.perf_counter()
        print(" -- Lucas-Kanade iteration {}".format(iter_index))

    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)

    blockspergrid_x = math.ceil(n_patch_x / threadsperblock[1])
    blockspergrid_y = math.ceil(n_patch_y / threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    ICA_get_new_flow[blockspergrid, threadsperblock](
        ref_img, comp_img,
        gradsx, gradsy,
        alignment, hessian, tile_size)

    if verbose_3:
        cuda.synchronize()
        current_time = getTime(
            current_time, ' --- Systems calculated and solved')


@cuda.jit
def ICA_get_new_flow(ref_img, comp_img, gradx, grady, alignment, hessian, tile_size):
    """
    The update relies on solving AX = B, a 2 by 2 system.
    A is precomputed, but B is evaluated each time.

    """
    imsize_y, imsize_x = comp_img.shape
    n_patchs_y, n_patchs_x, _ = alignment.shape
    patch_idx, patch_idy = cuda.grid(2)

    if not (0 <= patch_idy < n_patchs_y and
            0 <= patch_idx < n_patchs_x):
        return

    patch_pos_x = tile_size * patch_idx
    patch_pos_y = tile_size * patch_idy

    A = cuda.local.array((2, 2), dtype=DEFAULT_CUDA_FLOAT_TYPE)
    A[0, 0] = hessian[patch_idy, patch_idx, 0, 0]
    A[0, 1] = hessian[patch_idy, patch_idx, 0, 1]
    A[1, 0] = hessian[patch_idy, patch_idx, 1, 0]
    A[1, 1] = hessian[patch_idy, patch_idx, 1, 1]

    # By putting non solvable exit this early, the remaining calculations are
    # skipped for burned patches, which represents most of over-exposed images !
    if abs(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]) < 1e-10:  # system is Not solvable
        return

    B = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    B[0] = 0
    B[1] = 0

    local_alignment = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    local_alignment[0] = alignment[patch_idy, patch_idx, 0]
    local_alignment[1] = alignment[patch_idy, patch_idx, 1]

    buffer_val = cuda.local.array((2, 2), DEFAULT_CUDA_FLOAT_TYPE)
    pos = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)  # y, x

    for i in range(tile_size):
        for j in range(tile_size):
            pixel_global_idx = patch_pos_x + j  # global position on the coarse grey grid. Because of extremity padding, it can be out of bound
            pixel_global_idy = patch_pos_y + i

            if not (0 <= pixel_global_idx < imsize_x and
                    0 <= pixel_global_idy < imsize_y):
                continue

            local_gradx = gradx[pixel_global_idy, pixel_global_idx]
            local_grady = grady[pixel_global_idy, pixel_global_idx]

            # Warp I with W(x; p) to compute I(W(x; p))
            new_idx = local_alignment[0] + pixel_global_idx
            new_idy = local_alignment[1] + pixel_global_idy

            if not (0 <= new_idx < imsize_x - 1 and
                    0 <= new_idy < imsize_y - 1):  # -1 for bicubic interpolation
                continue

            # bicubic interpolation
            normalised_pos_x, floor_x = math.modf(new_idx)  # https://www.rollpie.com/post/252
            normalised_pos_y, floor_y = math.modf(new_idy)  # separating floor and floating part
            floor_x = int(floor_x)
            floor_y = int(floor_y)

            ceil_x = floor_x + 1
            ceil_y = floor_y + 1
            pos[0] = normalised_pos_y
            pos[1] = normalised_pos_x

            buffer_val[0, 0] = comp_img[floor_y, floor_x]
            buffer_val[0, 1] = comp_img[floor_y, ceil_x]
            buffer_val[1, 0] = comp_img[ceil_y, floor_x]
            buffer_val[1, 1] = comp_img[ceil_y, ceil_x]

            comp_val = bilinear_interpolation(buffer_val, pos)

            gradt = comp_val - ref_img[pixel_global_idy, pixel_global_idx]

            B[0] += -local_gradx * gradt
            B[1] += -local_grady * gradt

    alignment_step = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)

    # solvability is ensured by design
    solve_2x2(A, B, alignment_step)

    alignment[patch_idy, patch_idx, 0] = local_alignment[0] + alignment_step[0]
    alignment[patch_idy, patch_idx, 1] = local_alignment[1] + alignment_step[1]


@cuda.jit(device=True)
def bilinear_interpolation(values, pos):
    """


    Parameters
    ----------
    values : Array[2, 2]
        values on the 4 closest neighboors
    pos : Array[2]
        position where interpolation must be done (in [0, 1]x[0, 1]). y, x

    Returns
    -------
    val : float
        interpolated value

    """
    posy = pos[0]
    posx = pos[1]
    val = (values[0, 0] * (1 - posx) * (1 - posy) +
           values[0, 1] * (posx) * (1 - posy) +
           values[1, 0] * (1 - posx) * (posy) +
           values[1, 1] * posx * posy)
    return val


@cuda.jit(device=True)
def solve_2x2(A, B, X):
    """
    Cuda function for resolving the 2x2 system A*X = B
    by using the analytical formula

    Parameters
    ----------
    A : Array[2,2]

    B : Array[2]

    Returns
    -------
    None

    """
    det_A = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

    X[0] = (A[1, 1] * B[0] - A[0, 1] * B[1]) / det_A
    X[1] = (A[0, 0] * B[1] - A[1, 0] * B[0]) / det_A