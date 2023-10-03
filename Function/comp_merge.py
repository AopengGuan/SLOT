
import numpy as np
from numba import cuda,float32
import math
import torch as th
EPSILON_DIV = 1e-10



DEFAULT_NUMPY_FLOAT_TYPE = np.float32
DEFAULT_CUDA_FLOAT_TYPE = float32

DEFAULT_TORCH_FLOAT_TYPE = th.float32
DEFAULT_TORCH_COMPLEX_TYPE = th.complex64
DEFAULT_THREADS = 16


def merge(comp_img, alignments, covs, r, num, den,
          options, params, w_wert):
    """
    Implementation of Alg. 4: Accumulation
    Accumulates comp_img (J_n, n>1) into num and den, based on the alignment
    V_n, the covariance matrices Omega_n and the robustness mask estimated before.
    The size of the merge_result is adjustable with params['scale']


    Parameters
    ----------
    comp_imgs : device Array [imsize_y, imsize_x]
        The non-reference image to merge (J_n)
    alignments : device Array[n_tiles_y, n_tiles_x, 2]
        The final estimation of the tiles' alignment V_n(p)
    covs : device array[imsize_y//2, imsize_x//2, 2, 2]
        covariance matrices Omega_n
    r : Device_Array[imsize_y//2, imsize_x//2]
        Robustness mask r_n
    num : device Array[s*imshape_y, s*imshape_x]
        Numerator of the accumulator
    den : device Array[s*imshape_y, s*imshape_x]
        Denominator of the accumulator

    options : Dict
        Options to pass
    params : Dict
        parameters

    Returns
    -------
    None

    """
    scale = params['scale']

    bayer_mode = params['mode'] == 'bayer'
    iso_kernel = params['kernel'] == 'iso'
    tile_size = params['tuning']['tileSize']

    native_im_size = comp_img.shape
    # casting to integer to account for floating scale
    output_size = (round(scale * native_im_size[0]), round(scale * native_im_size[1]))

    # dispatching threads. 1 thread for 1 output pixel
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)  # maximum, we may take less
    blockspergrid_x = math.ceil(output_size[1] / threadsperblock[1])
    blockspergrid_y = math.ceil(output_size[0] / threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    accumulate[blockspergrid, threadsperblock](
        comp_img, alignments, covs, r,
        bayer_mode, iso_kernel, scale, tile_size,
        num, den,w_wert)


@cuda.jit
def accumulate(comp_img, alignments, covs, r,
               bayer_mode, iso_kernel, scale, tile_size,
               num, den,w_wert):
    """



    Parameters
    ----------
    comp_imgs : Array[imsize_y, imsize_x]
        The compared image
    alignements : Array[n_tiles_y, n_tiles_x, 2]
        The alignemnt vectors for each tile of the image
    covs : device array[imsize_y/2, imsize_x/2, 2, 2]
        covariance matrices sampled at the center of each bayer quad.
    r : Device_Array[imsize_y/2, imsize_x/2, 3]
            Robustness of the moving images
    bayer_mode : bool
        Whether the burst is raw or grey
    iso_kernel : bool
        Whether isotropic kernels should be used, or handhled's kernels.
    scale : float
        scaling factor
    tile_size : int
        tile size used for alignment (on the raw scale !)
    CFA_pattern : device Array[2, 2]
        CFA pattern of the burst
    output_img : Array[SCALE*imsize_y, SCALE_imsize_x]
        The empty output image

    Returns
    -------
    None.

    """

    output_pixel_idx, output_pixel_idy = cuda.grid(2)

    output_size_y, output_size_x, _ = num.shape
    input_size_y, input_size_x = comp_img.shape

    if not (0 <= output_pixel_idx < output_size_x and
            0 <= output_pixel_idy < output_size_y):
        return

    if bayer_mode:
        n_channels = 3
        acc = cuda.local.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        val = cuda.local.array(3, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    else:
        n_channels = 1
        acc = cuda.local.array(1, dtype=DEFAULT_CUDA_FLOAT_TYPE)
        val = cuda.local.array(1, dtype=DEFAULT_CUDA_FLOAT_TYPE)

    # Copying CFA locally. We will read that 9 times, so it's worth it
    #  threads could cooperate to read that
    '''
    local_CFA = cuda.local.array((2,2), uint8)
    for i in range(2):
        for j in range(2):
            local_CFA[i,j] = uint8(CFA_pattern[i,j])
            '''

    coarse_ref_sub_pos = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)  # y, x

    coarse_ref_sub_pos[0] = output_pixel_idy / scale
    coarse_ref_sub_pos[1] = output_pixel_idx / scale

    # fetch of the flow, as early as possible
    local_optical_flow = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    patch_idy = int(coarse_ref_sub_pos[0] // tile_size)
    patch_idx = int(coarse_ref_sub_pos[1] // tile_size)
    local_optical_flow[0] = alignments[patch_idy, patch_idx, 0]
    local_optical_flow[1] = alignments[patch_idy, patch_idx, 1]

    for chan in range(n_channels):
        acc[chan] = 0
        val[chan] = 0

    patch_center_pos = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)  # y, x

    # fetching robustness
    # The robustness of the center of the patch is picked through neirest neigbhoor interpolation

    if bayer_mode:
        y_r = clamp(round((coarse_ref_sub_pos[0] - 0.5) / 2), 0, r.shape[0])
        x_r = clamp(round((coarse_ref_sub_pos[1] - 0.5) / 2), 0, r.shape[1])

    else:
        y_r = clamp(round(coarse_ref_sub_pos[0]), 0, r.shape[0] - 1)
        x_r = clamp(round(coarse_ref_sub_pos[1]), 0, r.shape[1] - 1)
    local_r = r[y_r, x_r]

    patch_center_pos[1] = coarse_ref_sub_pos[1] + local_optical_flow[0]
    patch_center_pos[0] = coarse_ref_sub_pos[0] + local_optical_flow[1]

    # updating inbound condition
    if not (0 <= patch_center_pos[1] < input_size_x and
            0 <= patch_center_pos[0] < input_size_y):
        return

    # computing kernel
    if not iso_kernel:
        interpolated_cov = cuda.local.array((2, 2), dtype=DEFAULT_CUDA_FLOAT_TYPE)
        cov_i = cuda.local.array((2, 2), dtype=DEFAULT_CUDA_FLOAT_TYPE)
        # fetching the 4 closest covs
        close_covs = cuda.local.array((2, 2, 2, 2), DEFAULT_CUDA_FLOAT_TYPE)
        grey_pos = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)

        if bayer_mode:
            grey_pos[0] = (patch_center_pos[0] - 0.5) / 2  # grey grid is offseted and twice more sparse
            grey_pos[1] = (patch_center_pos[1] - 0.5) / 2

        else:
            grey_pos[0] = patch_center_pos[0]  # grey grid is exactly the coarse grid
            grey_pos[1] = patch_center_pos[1]

        # clipping the coordinates to stay in bound
        floor_x = int(max(math.floor(grey_pos[1]), 0))
        floor_y = int(max(math.floor(grey_pos[0]), 0))

        ceil_x = min(floor_x + 1, covs.shape[1] - 1)
        ceil_y = min(floor_y + 1, covs.shape[0] - 1)
        for i in range(0, 2):
            for j in range(0, 2):
                close_covs[0, 0, i, j] = covs[floor_y, floor_x,
                                              i, j]
                close_covs[0, 1, i, j] = covs[floor_y, ceil_x,
                                              i, j]
                close_covs[1, 0, i, j] = covs[ceil_y, floor_x,
                                              i, j]
                close_covs[1, 1, i, j] = covs[ceil_y, ceil_x,
                                              i, j]

        # interpolating covs at the desired spot
        interpolate_cov2(close_covs, grey_pos, interpolated_cov)

        if abs(interpolated_cov[0, 0] * interpolated_cov[1, 1] - interpolated_cov[0, 1] * interpolated_cov[
            1, 0]) > EPSILON_DIV:  # checking if cov is invertible
            invert_2x2(interpolated_cov, cov_i)


        else:  # if not invertible, identity matrix
            cov_i[0, 0] = 1
            cov_i[0, 1] = 0
            cov_i[1, 0] = 0
            cov_i[1, 1] = 1

    center_x = round(patch_center_pos[1])
    center_y = round(patch_center_pos[0])
    for i in range(-1, 2):
        for j in range(-1, 2):
            pixel_idx = center_x + j
            pixel_idy = center_y + i

            # in bound condition
            if (0 <= pixel_idx < input_size_x and
                    0 <= pixel_idy < input_size_y):

                # checking if pixel is r, g or b
                if bayer_mode:
                    print('mode is wrong, choose grey mode')
                else:
                    channel = 0

                # By fetching the value now, we can compute the kernel weight
                # while it is called from global memory
                c = comp_img[pixel_idy, pixel_idx]

                # computing distance
                dist_x = pixel_idx - patch_center_pos[1]
                dist_y = pixel_idy - patch_center_pos[0]

                ### Computing w
                if iso_kernel:
                    y = max(0, 2 * (dist_x * dist_x + dist_y * dist_y))
                else:
                    y = max(0, quad_mat_prod(cov_i, dist_x, dist_y))
                    # y can be slightly negative because of numerical precision.
                    # I clamp it to not explode the error with exp

                w = math.exp(-0.5 * y)
                w_wert[output_pixel_idy, output_pixel_idx,i+1,j+1] = w

                ############

                val[channel] += c * w * local_r
                acc[channel] += w * local_r

    # for chan in range(n_channels):
    num[output_pixel_idy, output_pixel_idx, 0] = val[0]
    den[output_pixel_idy, output_pixel_idx, 0] = acc[0]


@cuda.jit(device=True)
def interpolate_cov2(covs, center_pos, interpolated_cov):
    reframed_posx, _ = math.modf(center_pos[1])  # these positions are between 0 and 1
    reframed_posy, _ = math.modf(center_pos[0])
    # cov 00 is in (0,0) ; cov 01 in (0, 1) ; cov 01 in (1, 0), cov 11 in (1, 1)

    for i in range(2):
        for j in range(2):
            interpolated_cov[i, j] = (covs[0, 0, i, j] * (1 - reframed_posx) * (1 - reframed_posy) +
                                      covs[0, 1, i, j] * (reframed_posx) * (1 - reframed_posy) +
                                      covs[1, 0, i, j] * (1 - reframed_posx) * (reframed_posy) +
                                      covs[1, 1, i, j] * reframed_posx * reframed_posy)

@cuda.jit(device=True)
def interpolate_cov(covs, center_pos, interpolated_cov):
    reframed_posx, _ = math.modf(center_pos[1])  # these positions are between 0 and 1
    reframed_posy, _ = math.modf(center_pos[0])
    # cov 00 is in (0,0) ; cov 01 in (0, 1) ; cov 01 in (1, 0), cov 11 in (1, 1)

    for i in range(2):
        for j in range(2):
            a = covs[0, 0, i, j] + (covs[1, 0, i, j] - covs[0, 0, i, j]) * reframed_posy
            b = covs[0, 1, i, j] + (covs[1, 1, i, j] - covs[0, 1, i, j]) * reframed_posy
            interpolated_cov[i, j] = a + (b - a) * reframed_posx


@cuda.jit(device=True)
def invert_2x2(M, M_i):
    """
    inverts the 2x2 M array

    Parameters
    ----------
    M : Array[2, 2]
        Array to invert
    M_i : Array[2, 2]

    Returns
    -------
    None.

    """
    det_i = 1 / (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
    if math.isinf(det_i):
        M_i[0, 0] = 1
        M_i[0, 1] = 0
        M_i[1, 0] = 0
        M_i[1, 1] = 1
    else:
        M_i[0, 0] = M[1, 1] * det_i
        M_i[0, 1] = -M[0, 1] * det_i
        M_i[1, 0] = -M[1, 0] * det_i
        M_i[1, 1] = M[0, 0] * det_i


@cuda.jit(device=True)
def quad_mat_prod(A, X1, X2):
    """
    With X = [X1, X2], performs the quadratique form :
        X.transpose() @ A @ X

    Parameters
    ----------
    A : device Array[2, 2]
    X1 : float
    X2 : float

    Returns
    -------
    y : float

    """
    y = A[0, 0] * X1 * X1 + X1 * X2 * (A[0, 1] + A[1, 0]) + A[1, 1] * X2 * X2
    return y

@cuda.jit(device=True)
def clamp(x, min_, max_):
    return min(max_, max(min_, x))