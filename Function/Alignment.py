import time
import numpy as np
from numba import cuda,float32
import math
import torch.nn.functional as F
import torch as th
from .Time import getTime
from .pyramid import hdrplusPyramid
from collections import Counter

DEFAULT_NUMPY_FLOAT_TYPE = np.float32
DEFAULT_CUDA_FLOAT_TYPE = float32

DEFAULT_TORCH_FLOAT_TYPE = th.float32
DEFAULT_TORCH_COMPLEX_TYPE = th.complex64
DEFAULT_THREADS = 16




def align_image_block_matching(img, referencePyramid, options, params, debug=False):
    """
    Align the reference image with the img : returns a patchwise flow such that
    for patches py, px :
        img[py, px] ~= ref_img[py + alignments[py, px, 1],
                               px + alignments[py, px, 0]]

    Parameters
    ----------
    img : device Array[imshape_y, imshape_x]
        Image to be compared J_i (i>1)
    referencePyramid : list [device Array]
        Pyramid representation of the ref image J_1
    options : dict
        options.
    params : dict
        parameters.
    debug : Bool, optional
        When True, a list with the alignment at each step is returned. The default is False.

    Returns
    -------
    alignments : device Array[n_patchs_y, n_patchs_x, 2]
        Patchwise flow : V_n(p) for each patch (p)

    """
    # Initialization.
    h, w = img.shape  # height and width should be identical for all images

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

    th_img = th.as_tensor(img, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]

    img_padded = F.pad(th_img, (paddingLeft, paddingRight, paddingTop, paddingBottom), 'circular')

    # For convenience
    currentTime, verbose = time.perf_counter(), options['verbose'] > 2
    # factors, tileSizes, distances, searchRadia and subpixels are described fine-to-coarse
    factors = params['tuning']['factors']
    tileSizes = params['tuning']['tileSizes']
    distances = params['tuning']['distances']
    searchRadia = params['tuning']['searchRadia']

    upsamplingFactors = factors[1:] + [1]
    previousTileSizes = tileSizes[1:] + [None]

    # Align alternate image to the reference image

    # 4-level coarse-to fine pyramid of alternate image
    alternatePyramid = hdrplusPyramid(img_padded, factors)
    if verbose:
        cuda.synchronize()
        currentTime = getTime(currentTime, ' --- Create comp pyramid')

    # succesively align from coarsest to finest level of the pyramid
    alignments = None
    if debug:
        debug_list = []

    for lv in range(len(referencePyramid)):
        alignments = align_on_a_level(
            referencePyramid[lv],
            alternatePyramid[lv],
            options,
            upsamplingFactors[-lv - 1],
            tileSizes[-lv - 1],
            previousTileSizes[-lv - 1],
            searchRadia[-lv - 1],
            distances[-lv - 1],
            alignments
        )

        if debug:
            debug_list.append(alignments.copy_to_host())

        if verbose:
            cuda.synchronize()
            currentTime = getTime(currentTime, ' --- Align pyramid')
    if debug:
        return debug_list
    return alignments


def align_on_a_level(referencePyramidLevel, alternatePyramidLevel, options, upsamplingFactor, tileSize,
                     previousTileSize, searchRadius, distance, previousAlignments):
    """
    Alignment will always be an integer with this function, however it is
    set to DEFAULT_FLOAT_TYPE. This enables to directly use the outputed
    alignment for ICA without any casting from int to float, which would be hard
    to perform on GPU : Numba is completely powerless and cannot make the
    casting.

    """

    # For convenience
    verbose = options['verbose'] > 3
    if verbose:
        cuda.synchronize()
        currentTime = time.perf_counter()
    imshape = referencePyramidLevel.shape

    # This formula is checked : it is correct
    # Number of patches that can fit on this level
    h = imshape[0] // tileSize
    w = imshape[1] // tileSize

    # Upsample the previous alignements for initialization
    if previousAlignments is None:
        upsampledAlignments = cuda.to_device(np.zeros((h, w, 2), dtype=DEFAULT_NUMPY_FLOAT_TYPE))
        print("upsampled shape", upsampledAlignments.shape)

    else:
        # use the upsampled previous alignments as initial guesses
        upsampledAlignments = upsample_alignments(
            referencePyramidLevel,
            alternatePyramidLevel,
            previousAlignments,
            upsamplingFactor,
            tileSize,
            previousTileSize
        )

    if verbose:
        cuda.synchronize()
        currentTime = getTime(currentTime, ' ---- Upsample alignments')

    local_search(referencePyramidLevel, alternatePyramidLevel,
                 tileSize, searchRadius,
                 upsampledAlignments, distance)
    if searchRadius != 1:
        nalignment = cuda.device_array(upsampledAlignments.shape, DEFAULT_NUMPY_FLOAT_TYPE)
        x_alignments = upsampledAlignments[:, :, 0].copy_to_host()
        y_alignments = upsampledAlignments[:, :, 1].copy_to_host()
        flatten_array_x = x_alignments.flatten()
        flatten_array_y = y_alignments.flatten()
        counter_x = Counter(flatten_array_x)
        counter_y = Counter(flatten_array_y)
        most_common_element_x, most_common_count_x = counter_x.most_common(1)[0]
        most_common_element_y, most_common_count_y = counter_y.most_common(1)[0]
        nalignment[:,:,0] = most_common_element_x
        nalignment[:, :, 1] = most_common_element_y
    else:
        nalignment = upsampledAlignments
    return nalignment#upsampledAlignments


def upsample_alignments(referencePyramidLevel, alternatePyramidLevel, previousAlignments, upsamplingFactor, tileSize,
                        previousTileSize):
    '''Upsample alignements to adapt them to the next pyramid level (Section 3.2 of the IPOL article).'''
    n_tiles_y_prev, n_tiles_x_prev, _ = previousAlignments.shape
    # Different resolution upsampling factors and tile sizes lead to different vector repetitions

    # UpsampledAlignments.shape can be less than referencePyramidLevel.shape/tileSize
    # eg when previous alignments could not be computed over the whole image
    n_tiles_y_new = referencePyramidLevel.shape[0] // tileSize
    n_tiles_x_new = referencePyramidLevel.shape[1] // tileSize

    upsampledAlignments = cuda.device_array((n_tiles_y_new, n_tiles_x_new, 2), dtype=DEFAULT_NUMPY_FLOAT_TYPE)
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    blockspergrid_x = math.ceil(n_tiles_x_new / threadsperblock[1])
    blockspergrid_y = math.ceil(n_tiles_y_new / threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    cuda_upsample_alignments[blockspergrid, threadsperblock](
        referencePyramidLevel, alternatePyramidLevel,
        upsampledAlignments, previousAlignments,
        upsamplingFactor, tileSize, previousTileSize)

    return upsampledAlignments


@cuda.jit
def cuda_upsample_alignments(referencePyramidLevel, alternatePyramidLevel, upsampledAlignments, previousAlignments,
                             upsamplingFactor, tileSize, previousTileSize):
    subtile_x, subtile_y = cuda.grid(2)
    n_tiles_y_prev, n_tiles_x_prev, _ = previousAlignments.shape
    n_tiles_y_new, n_tiles_x_new, _ = upsampledAlignments.shape
    h, w = referencePyramidLevel.shape

    repeatFactor = upsamplingFactor // (tileSize // previousTileSize)
    if not (0 <= subtile_x < n_tiles_x_new and
            0 <= subtile_y < n_tiles_y_new):
        return

    # the new subtile is on the side of the image, and is not contained within a bigger old tile
    if (subtile_x >= repeatFactor * n_tiles_x_prev or
            subtile_y >= repeatFactor * n_tiles_y_prev):
        upsampledAlignments[subtile_y, subtile_x, 0] = 0
        upsampledAlignments[subtile_y, subtile_x, 1] = 0
        return

    # else
    prev_tile_x = subtile_x // repeatFactor
    prev_tile_y = subtile_y // repeatFactor

    candidate_alignment_0_shift = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    candidate_alignment_0_shift[0] = previousAlignments[prev_tile_y, prev_tile_x, 0] * upsamplingFactor
    candidate_alignment_0_shift[1] = previousAlignments[prev_tile_y, prev_tile_x, 1] * upsamplingFactor

    # position of the top left pixel in the subtile
    subtile_pos_y = subtile_y * tileSize
    subtile_pos_x = subtile_x * tileSize

    # copying ref patch into local memory, because it needs to be read 3 times
    #  this should be rewritten to allow patchs bigger than 32
    local_ref = cuda.local.array((32, 32), DEFAULT_CUDA_FLOAT_TYPE)
    for i in range(tileSize):
        for j in range(tileSize):
            idx = subtile_pos_x + j
            idy = subtile_pos_y + i
            local_ref[i, j] = referencePyramidLevel[idy, idx]

    # position of the new tile within the old tile
    ups_subtile_x = subtile_x % repeatFactor
    ups_subtile_y = subtile_y % repeatFactor

    # computing id for the 3 closest patchs
    if 2 * ups_subtile_x + 1 > repeatFactor:
        x_shift = +1
    else:
        x_shift = -1

    if 2 * ups_subtile_y + 1 > repeatFactor:
        y_shift = +1
    else:
        y_shift = -1

    # 3 Candidates alignments are fetched (by fetching them as early as possible, we may received
    # them from global memory before we even require them, as calculations are performed during this delay)
    candidate_alignment_vert_shift = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    candidate_alignment_vert_shift[0] = previousAlignments[clamp(prev_tile_y + y_shift, 0, n_tiles_y_prev - 1),
                                                           prev_tile_x,
                                                           0] * upsamplingFactor
    candidate_alignment_vert_shift[1] = previousAlignments[clamp(prev_tile_y + y_shift, 0, n_tiles_y_prev - 1),
                                                           prev_tile_x,
                                                           1] * upsamplingFactor

    candidate_alignment_horizontal_shift = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    candidate_alignment_horizontal_shift[0] = previousAlignments[prev_tile_y,
                                                                 clamp(prev_tile_x + x_shift, 0, n_tiles_x_prev - 1),
                                                                 0] * upsamplingFactor
    candidate_alignment_horizontal_shift[1] = previousAlignments[prev_tile_y,
                                                                 clamp(prev_tile_x + x_shift, 0, n_tiles_x_prev - 1),
                                                                 1] * upsamplingFactor

    # Choosing the best of the 3 alignments by minimising L1 dist
    dist = +1 / 0
    optimal_flow_x = 0
    optimal_flow_y = 0

    # 0 shift
    dist_ = 0
    for i in range(tileSize):
        for j in range(tileSize):
            new_idy = subtile_pos_y + i + int(candidate_alignment_0_shift[1])
            new_idx = subtile_pos_x + j + int(candidate_alignment_0_shift[0])
            if (0 <= new_idx < w and
                    0 <= new_idy < h):
                dist_ += abs(local_ref[i, j] - alternatePyramidLevel[new_idy, new_idx])
            else:
                dist_ = 1 / 0
    if dist_ < dist:
        dist = dist_
        optimal_flow_x = candidate_alignment_0_shift[0]
        optimal_flow_y = candidate_alignment_0_shift[1]

    # vertical shift
    dist_ = 0
    for i in range(tileSize):
        for j in range(tileSize):
            new_idy = subtile_pos_y + i + int(candidate_alignment_vert_shift[1])
            new_idx = subtile_pos_x + j + int(candidate_alignment_vert_shift[0])
            if (0 <= new_idx < w and
                    0 <= new_idy < h):
                dist_ += abs(local_ref[i, j] - alternatePyramidLevel[new_idy, new_idx])
            else:
                dist_ = 1 / 0
    if dist_ < dist:
        dist = dist_
        optimal_flow_x = candidate_alignment_vert_shift[0]
        optimal_flow_y = candidate_alignment_vert_shift[1]

    # horizontal shift
    dist_ = 0
    for i in range(tileSize):
        for j in range(tileSize):
            new_idy = subtile_pos_y + i + int(candidate_alignment_horizontal_shift[1])
            new_idx = subtile_pos_x + j + int(candidate_alignment_horizontal_shift[0])
            if (0 <= new_idx < w and
                    0 <= new_idy < h):
                dist_ += abs(local_ref[i, j] - alternatePyramidLevel[new_idy, new_idx])
            else:
                dist_ = 1 / 0
    if dist_ < dist:
        optimal_flow_x = candidate_alignment_horizontal_shift[0]
        optimal_flow_y = candidate_alignment_horizontal_shift[1]

    # applying best flow
    upsampledAlignments[subtile_y, subtile_x, 0] = optimal_flow_x
    upsampledAlignments[subtile_y, subtile_x, 1] = optimal_flow_y


def local_search(referencePyramidLevel, alternatePyramidLevel,
                 tileSize, searchRadius,
                 upsampledAlignments, distance):
    h, w, _ = upsampledAlignments.shape

    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    blockspergrid_x = math.ceil(w / threadsperblock[1])
    blockspergrid_y = math.ceil(h / threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    # 计算在上一pyramid层得出的位移矩阵条件下，comupute图像当前pyramid层与ref图像的位移偏差
    if distance == 'L1':
        cuda_L1_local_search[blockspergrid, threadsperblock](referencePyramidLevel, alternatePyramidLevel,
                                                             tileSize, searchRadius,
                                                             upsampledAlignments)

    elif distance == 'L2':
        cuda_L2_local_search[blockspergrid, threadsperblock](referencePyramidLevel, alternatePyramidLevel,
                                                             tileSize, searchRadius,
                                                             upsampledAlignments)

    else:
        raise ValueError('Unknown distance : {}'.format(distance))


@cuda.jit
def cuda_L1_local_search(referencePyramidLevel, alternatePyramidLevel,
                         tileSize, searchRadius, upsampledAlignments):
    n_patchs_y, n_patchs_x, _ = upsampledAlignments.shape
    h, w = alternatePyramidLevel.shape
    tile_x, tile_y = cuda.grid(2)
    if not (0 <= tile_y < n_patchs_y and
            0 <= tile_x < n_patchs_x):
        return

    local_flow = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    local_flow[0] = upsampledAlignments[tile_y, tile_x, 0]
    local_flow[1] = upsampledAlignments[tile_y, tile_x, 1]

    # position of the pixel in the top left corner of the patch
    patch_pos_x = tile_x * tileSize
    patch_pos_y = tile_y * tileSize

    #  this should be rewritten to allow patchs bigger than 32
    local_ref = cuda.local.array((32, 32), DEFAULT_CUDA_FLOAT_TYPE)
    for i in range(tileSize):
        for j in range(tileSize):
            idx = patch_pos_x + j
            idy = patch_pos_y + i
            local_ref[i, j] = referencePyramidLevel[idy, idx]

    min_dist = +1 / 0  # init as infty
    min_shift_y = 0
    min_shift_x = 0
    # window search
    for search_shift_y in range(-searchRadius, searchRadius + 1):
        for search_shift_x in range(-searchRadius, searchRadius + 1):
            # computing dist
            dist = 0
            # 在搜索半径内，ref图像中tileSize*tileSize范围内与compute图像这个范围内的误差小于min_dist，才认为这个特征进行了微小的移动
            for i in range(tileSize):
                for j in range(tileSize):
                    new_idx = patch_pos_x + j + int(local_flow[0]) + search_shift_x
                    new_idy = patch_pos_y + i + int(local_flow[1]) + search_shift_y

                    if (0 <= new_idx < w and
                            0 <= new_idy < h):
                        diff = local_ref[i, j] - alternatePyramidLevel[new_idy, new_idx]
                        dist += abs(diff)
                    else:
                        dist = +1 / 0

            if dist < min_dist:
                min_dist = dist
                min_shift_y = search_shift_y
                min_shift_x = search_shift_x

    upsampledAlignments[tile_y, tile_x, 0] = local_flow[0] + min_shift_x
    upsampledAlignments[tile_y, tile_x, 1] = local_flow[1] + min_shift_y


@cuda.jit
def cuda_L2_local_search(referencePyramidLevel, alternatePyramidLevel,
                         tileSize, searchRadius, upsampledAlignments):
    n_patchs_y, n_patchs_x, _ = upsampledAlignments.shape
    h, w = alternatePyramidLevel.shape
    tile_x, tile_y = cuda.grid(2)
    if not (0 <= tile_y < n_patchs_y and
            0 <= tile_x < n_patchs_x):
        return

    local_flow = cuda.local.array(2, DEFAULT_CUDA_FLOAT_TYPE)
    local_flow[0] = upsampledAlignments[tile_y, tile_x, 0]
    local_flow[1] = upsampledAlignments[tile_y, tile_x, 1]

    # position of the pixel in the top left corner of the patch
    patch_pos_x = tile_x * tileSize
    patch_pos_y = tile_y * tileSize

    # this should be rewritten to allow patchs bigger than 32
    local_ref = cuda.local.array((32, 32), DEFAULT_CUDA_FLOAT_TYPE)
    for i in range(tileSize):
        for j in range(tileSize):
            idx = patch_pos_x + j
            idy = patch_pos_y + i
            local_ref[i, j] = referencePyramidLevel[idy, idx]

    min_dist = +1 / 0  # init as infty
    min_shift_y = 0
    min_shift_x = 0
    # window search
    for search_shift_y in range(-searchRadius, searchRadius + 1):
        for search_shift_x in range(-searchRadius, searchRadius + 1):
            # computing dist
            dist = 0
            for i in range(tileSize):
                for j in range(tileSize):
                    new_idx = patch_pos_x + j + int(local_flow[0]) + search_shift_x
                    new_idy = patch_pos_y + i + int(local_flow[1]) + search_shift_y

                    if (0 <= new_idx < w and
                            0 <= new_idy < h):
                        diff = local_ref[i, j] - alternatePyramidLevel[new_idy, new_idx]
                        dist += diff * diff
                    else:
                        dist = +1 / 0

            if dist < min_dist:
                min_dist = dist
                min_shift_y = search_shift_y
                min_shift_x = search_shift_x

    upsampledAlignments[tile_y, tile_x, 0] = local_flow[0] + min_shift_x
    upsampledAlignments[tile_y, tile_x, 1] = local_flow[1] + min_shift_y

@cuda.jit(device=True)
def clamp(x, min_, max_):
    return min(max_, max(min_, x))