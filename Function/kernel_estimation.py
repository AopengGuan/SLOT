import time
import numpy as np
from numba import cuda,float32
import math
import torch.nn.functional as F
import torch as th
from .Time import getTime

DEFAULT_NUMPY_FLOAT_TYPE = np.float32
DEFAULT_CUDA_FLOAT_TYPE = float32

DEFAULT_TORCH_FLOAT_TYPE = th.float32
DEFAULT_TORCH_COMPLEX_TYPE = th.complex64
DEFAULT_THREADS = 16




def estimate_kernels(img, options, params):
    """
    Implementation of Alg. 5: ComputeKernelCovariance
    Returns the kernels covariance matrices for the frame J_n, sampled at the
    center of every bayer quad (or at the center of every grey pixel in grey
    mode).

    Parameters
    ----------
    img : device Array[imshape_y, imshape_x]
        Raw image J_n
    options : dict
        optionsqa
    params : dict
        params['mode'] : {"bayer", "grey"}
            Wether the burst is raw or grey
        params['tuning'] : dict
            parameters driving the kernel shape
        params['noise'] : dict
            cointain noise model informations

    Returns
    -------
    covs : device Array[imshape_y//2, imshape_x//2, 2, 2]
        Covariance matrices Omega_n, sampled at the center of each bayer quad.

    """
    bayer_mode = params['mode'] == 'bayer'
    verbose_3 = options['verbose'] >= 3

    k_detail = params['tuning']['k_detail']
    k_denoise = params['tuning']['k_denoise']
    l1_max = params['tuning']['l1_max']
    l1_min = params['tuning']['l1_min']
    k_stretch = params['tuning']['k_stretch']
    k_shrink = params['tuning']['k_shrink']

    alpha = params['noise']['alpha']
    beta = params['noise']['beta']

    if verbose_3:
        cuda.synchronize()
        t1 = time.perf_counter()

    # __ Decimate to grey
    if bayer_mode:
        #img_grey = compute_grey_images(img, method="decimating")
        print('false Mode')

        if verbose_3:
            cuda.synchronize()
            t1 = getTime(t1, "- Decimated Image")
    else:
        img_grey = img  # no need to copy now, they will be copied to gpu later.

    grey_imshape_y, grey_imshape_x = grey_imshape = img_grey.shape

    # __ Performing Variance Stabilization Transform

    img_grey = GAT(img_grey, alpha, beta)

    if verbose_3:
        cuda.synchronize()
        t1 = getTime(t1, "- Variance Stabilized")

    # __ Computing grads
    th_grey_img = th.as_tensor(img_grey, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")[None, None]

    grad_kernel1 = np.array([[[[-0.5, 0.5]]],

                             [[[0.5, 0.5]]]])
    grad_kernel1 = th.as_tensor(grad_kernel1, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")

    grad_kernel2 = np.array([[[[0.5],
                               [0.5]]],

                             [[[-0.5],
                               [0.5]]]])
    grad_kernel2 = th.as_tensor(grad_kernel2, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")

    tmp = F.conv2d(th_grey_img, grad_kernel1)
    th_full_grad = F.conv2d(tmp, grad_kernel2, groups=2)
    # The default padding mode reduces the shape of grey_img of 1 pixel in each
    # direction, as expected

    cuda_full_grads = cuda.as_cuda_array(th_full_grad.squeeze().transpose(0, 1).transpose(1, 2))
    # shape [y, x, 2]
    if verbose_3:
        cuda.synchronize()
        t1 = getTime(t1, "- Gradients computed")

    covs = cuda.device_array(grey_imshape + (2, 2), DEFAULT_NUMPY_FLOAT_TYPE)
    e = cuda.device_array(grey_imshape + (2,2), DEFAULT_NUMPY_FLOAT_TYPE)
    k_wert = cuda.device_array(grey_imshape + (2,), DEFAULT_NUMPY_FLOAT_TYPE)
    l_wert = cuda.device_array(grey_imshape + (2,), DEFAULT_NUMPY_FLOAT_TYPE)

    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    blockspergrid_x = math.ceil(grey_imshape_x / threadsperblock[1])
    blockspergrid_y = math.ceil(grey_imshape_y / threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    cuda_estimate_kernel[blockspergrid, threadsperblock](cuda_full_grads,
                                                         k_detail, k_denoise, l1_max, l1_min, k_stretch, k_shrink,
                                                         covs,e,k_wert,l_wert)
    if verbose_3:
        cuda.synchronize()
        t1 = getTime(t1, "- Covariances estimated")

    return covs, e, k_wert, l_wert


def GAT(image, alpha, beta):
    """
    Generalized Ascombe Transform
    noise model : std² = alpha * I + beta
    Where alpha and beta are iso dependant.

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    alpha : float
        value of alpha for the given iso
    iso : float
        ISO value
    beta : float
        Value of beta for the given iso

    Returns
    -------
    VST_image : TYPE
        input image with stabilized variance

    """
    assert len(image.shape) == 2
    imshape_y, imshape_x = image.shape

    VST_image = cuda.device_array(image.shape, DEFAULT_NUMPY_FLOAT_TYPE)

    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    blockspergrid_x = math.ceil(imshape_x / threadsperblock[1])
    blockspergrid_y = math.ceil(imshape_y / threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    cuda_GAT[blockspergrid, threadsperblock](image, VST_image,
                                             alpha, beta)

    return VST_image


@cuda.jit
def cuda_GAT(image, VST_image, alpha, beta):
    x, y = cuda.grid(2)
    imshape_y, imshape_x = image.shape

    if not (0 <= y < imshape_y and
            0 <= x < imshape_x):
        return

    # ISO should not appear here,  since alpha and beta are
    # already iso dependant.
    VST = alpha * image[y, x] + 3 / 8 * alpha * alpha + beta
    VST = max(0, VST)

    VST_image[y, x] = 2 / alpha * math.sqrt(VST)


@cuda.jit
def cuda_estimate_kernel(full_grads,
                         k_detail, k_denoise,
                         l1_max, l1_min,
                         k_stretch, k_shrink,
                         covs,e,k_wert, l_wert):
    pixel_idx, pixel_idy = cuda.grid(2)
    imshape_y, imshape_x, _, _ = covs.shape

    if not (0 <= pixel_idy < imshape_y and
            0 <= pixel_idx < imshape_x):
        return

    structure_tensor = cuda.local.array((2, 2), DEFAULT_CUDA_FLOAT_TYPE)
    structure_tensor[0, 0] = 0
    structure_tensor[0, 1] = 0
    structure_tensor[1, 0] = 0
    structure_tensor[1, 1] = 0

    for i in range(0, 1):
        for j in range(0, 1):
            x = pixel_idx - 1 + j
            y = pixel_idy - 1 + i

            if (0 <= y < full_grads.shape[0] and
                    0 <= x < full_grads.shape[1]):
                full_grad_x = full_grads[y, x, 0]
                full_grad_y = full_grads[y, x, 1]

                structure_tensor[0, 0] += full_grad_x * full_grad_x
                structure_tensor[1, 0] += full_grad_x * full_grad_y
                structure_tensor[0, 1] += full_grad_x * full_grad_y
                structure_tensor[1, 1] += full_grad_y * full_grad_y

    l = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    e1 = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    e2 = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)
    k = cuda.local.array(2, dtype=DEFAULT_CUDA_FLOAT_TYPE)

    get_eighen_elmts_2x2(structure_tensor, l, e1, e2)

    compute_k(l[0], l[1], k, k_detail, k_denoise, l1_max, l1_min, k_stretch,
              k_shrink)

    k_1_sq = k[0] * k[0]
    k_2_sq = k[1] * k[1]

    l_wert[pixel_idy, pixel_idx, 0] = l[0]
    l_wert[pixel_idy, pixel_idx, 1] = l[1]

    k_wert[pixel_idy, pixel_idx, 0] = k[0]
    k_wert[pixel_idy, pixel_idx, 1] = k[1]

    e[pixel_idy, pixel_idx, 0, 0] = e1[0]
    e[pixel_idy, pixel_idx, 0, 1] = e1[1]
    e[pixel_idy, pixel_idx, 1, 0] = e2[0]
    e[pixel_idy, pixel_idx, 1, 1] = e2[1]

    covs[pixel_idy, pixel_idx, 0, 0] = k_1_sq * e1[0] * e1[0] + k_2_sq * e2[0] * e2[0]
    covs[pixel_idy, pixel_idx, 0, 1] = k_1_sq * e1[0] * e1[1] + k_2_sq * e2[0] * e2[1]
    covs[pixel_idy, pixel_idx, 1, 0] = k_1_sq * e1[0] * e1[1] + k_2_sq * e2[0] * e2[1]
    covs[pixel_idy, pixel_idx, 1, 1] = k_1_sq * e1[1] * e1[1] + k_2_sq * e2[1] * e2[1]


@cuda.jit(device=True)
def compute_k(l1, l2, k, k_detail, k_denoise, l1_max, l1_min, k_stretch,
              k_shrink):
    """
    Computes k_1 and k_2 based on lambda1, lambda2 and the constants.

    Parameters
    ----------
    l1 : float
        lambda1 (dominant eighen value)
    l2 : float
        lambda2
    k : Array[2]
        empty vector where k_1 and k_2 will be stored
    k_detail : TYPE
        DESCRIPTION.
    k_denoise : TYPE
        DESCRIPTION.
    D_th : TYPE
        DESCRIPTION.
    D_tr : TYPE
        DESCRIPTION.
    k_stretch : TYPE
        DESCRIPTION.
    k_shrink : TYPE
        DESCRIPTION.


    """
    A = 1 + math.sqrt((l1 - l2) / (l1 + l2))
    D = clamp((l1-l1_min) / (l1_max-l1_min), 0, 1)

    # This is a very agressive way of driving anisotropy, but it works well so far.
    if A > 1.95:
        k1 = 1 / k_shrink
        k2 = k_stretch
    else:  # When A is Nan, we fall back to this condition
        k1 = 1
        k2 = 1

    k[0] = k_detail * ((1 - D) * k1 + D * k_denoise)
    k[1] = k_detail * ((1 - D) * k2 + D * k_denoise)


@cuda.jit(device=True)
def get_real_polyroots_2(a, b, c, roots):
    """
    Returns the two roots of the polynom a*X^2 + b*X + c = 0 for a, b and c
    real numbers. The function only returns real roots : make sure they exist
    before calling the function. l[0] contains the root with the biggest module
    and l[1] the smallest


    Parameters
    ----------
    a : float

    b : float

    c : float

    roots : Array[2]

    Returns
    -------
    None

    """

    # numerical instabilities can cause delta to be slightly negative despite
    # the equation admitting 2 real roots.
    delta = max(b * b - 4 * a * c, 0)

    r1 = (-b + math.sqrt(delta)) / (2 * a)
    r2 = (-b - math.sqrt(delta)) / (2 * a)
    if abs(r1) >= abs(r2):
        roots[0] = r1
        roots[1] = r2
    else:
        roots[0] = r2
        roots[1] = r1


@cuda.jit(device=True)
def get_eighen_val_2x2(M, l):
    a = 1
    b = -(M[0, 0] + M[1, 1])
    c = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    get_real_polyroots_2(a, b, c, l)


@cuda.jit(device=True)
def get_eighen_vect_2x2(M, l, e1, e2):
    """
    return the eighen vectors with norm 1 for the eighen values l
    M.e1 = l1.e1 ; M.e2 = l2.e2

    Parameters
    ----------
    M : Array[2,2]
      Real Symmetric array for which eighen values are to be determined
    l : Array[2]
    e1, e2 : Array[2]
        sorted Eigenvalues
    e1, e2 : Array[2, 2]
        Computed orthogonal and normalized eigen vectors

    Returns
    -------
    None.

    """
    # 2x2 algorithm : https://en.wikipedia.org/wiki/Eigenvalue_algorithm (9 August 2022 version)
    if M[0, 1] == 0 and M[0, 0] == M[1, 1]:
        # M is multiple of identity, picking 2 ortogonal eighen vectors.
        e1[0] = 1
        e1[1] = 0
        e2[0] = 0
        e2[1] = 1

    else:
        # averaging 2 for increased reliability
        e1[0] = M[0, 0] + M[0, 1] - l[1]
        e1[1] = M[1, 0] + M[1, 1] - l[1]

        if e1[0] == 0:
            e1[1] = 1
            e2[0] = 1;
            e2[1] = 0
        elif e1[1] == 0:
            e1[0] = 1
            e2[0] = 0
            e2[1] = 1
        else:
            norm_ = math.sqrt(e1[0] * e1[0] + e1[1] * e1[1])
            e1[0] /= norm_
            e1[1] /= norm_
            sign = math.copysign(1, e1[0])  # for whatever reason, python has no sign func
            e2[1] = abs(e1[0])
            e2[0] = -e1[1] * sign


@cuda.jit(device=True)
def get_eighen_elmts_2x2(M, l, e1, e2):
    get_eighen_val_2x2(M, l)
    get_eighen_vect_2x2(M, l, e1, e2)

@cuda.jit(device=True)
def clamp(x, min_, max_):
    return min(max_, max(min_, x))