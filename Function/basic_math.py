import numpy as np
from numba import cuda,float32
import torch as th
import math

DEFAULT_NUMPY_FLOAT_TYPE = np.float32
DEFAULT_CUDA_FLOAT_TYPE = float32

DEFAULT_TORCH_FLOAT_TYPE = th.float32
DEFAULT_TORCH_COMPLEX_TYPE = th.complex64
DEFAULT_THREADS = 16



def add(A, B):
    """
    performs A += B for 2d arrays

    Parameters
    ----------
    A : device_array[ny, nx]

    B : device_array[ny, nx]


    Returns
    -------
    None.

    """
    assert A.shape == B.shape
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    blockspergrid_x = math.ceil(A.shape[1 ] /threadsperblock[1])
    blockspergrid_y = math.ceil(A.shape[0 ] /threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    cuda_add[blockspergrid, threadsperblock](A, B)

@cuda.jit
def cuda_add(A, B):
    x, y = cuda.grid(2)
    if 0 <= x < A.shape[1] and 0 <= y < A.shape[0]:
        A[y, x] += B[y, x]


def divide(num, den):
    """
    Performs num = num/den

    Parameters
    ----------
    num : device array[ny, nx, n_channels]
        DESCRIPTION.
    den : device array[ny, nx, n_channels]
        DESCRIPTION.


    """
    assert num.shape == den.shape
    endresult = cuda.device_array(num.shape, DEFAULT_NUMPY_FLOAT_TYPE)
    n_channels = num.shape[-1]
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS, 1)
    blockspergrid_x = math.ceil(num.shape[1] / threadsperblock[1])
    blockspergrid_y = math.ceil(num.shape[0] / threadsperblock[0])
    blockspergrid_z = n_channels
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    cuda_divide[blockspergrid, threadsperblock](num, den, endresult)
    return endresult


@cuda.jit
def cuda_divide(num, den, endresult):
    x, y, c = cuda.grid(3)
    if (0 <= x < num.shape[1] and
            0 <= y < num.shape[0] and
            0 <= c < num.shape[2]):
        endresult[y, x, c] = num[y, x, c] / den[y, x, c]

def multiple(A,B):
    assert A.shape == B.shape
    C = cuda.device_array(A.shape, DEFAULT_NUMPY_FLOAT_TYPE)
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    blockspergrid_x = math.ceil(A.shape[1] / threadsperblock[1])
    blockspergrid_y = math.ceil(A.shape[0] / threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    cuda_multiple[blockspergrid, threadsperblock](A, B, C)
    return C

@cuda.jit
def cuda_multiple(A, B, C):
    x, y = cuda.grid(2)
    if 0 <= x < A.shape[1] and 0 <= y < A.shape[0]:
        C[y, x] = A[y, x] * B[y, x]

def calculate_A(l1, l2, endresult):

    assert l1.shape == l2.shape
    n_channels = l1.shape[-1]
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS, 1)
    blockspergrid_x = math.ceil(l1.shape[1] / threadsperblock[1])
    blockspergrid_y = math.ceil(l1.shape[0] / threadsperblock[0])
    blockspergrid_z = n_channels
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    cuda_calculate_A[blockspergrid, threadsperblock](l1, l2, endresult)
    #endresult中，1表示A>1.95,0表示A<=1.95
    return endresult

@cuda.jit
def cuda_calculate_A(l1, l2, endresult):
    x, y = cuda.grid(2)
    if (0 <= x < l1.shape[1] and 0 <= y < l1.shape[0]):
        A = math.sqrt((l1[y, x] - l2[y, x])/(l1[y, x] + l2[y, x]))
        if A > 0.95 and l1[y,x]>5:
            endresult[y, x] = A
        else:
            endresult[y, x] = 0

def exp(d, sigma):
    """
    Performs num = num/den

    Parameters
    ----------
    num : device array[ny, nx, n_channels]
        DESCRIPTION.
    den : device array[ny, nx, n_channels]
        DESCRIPTION.


    """
    assert d.shape == sigma.shape
    endresult = cuda.device_array(d.shape, DEFAULT_NUMPY_FLOAT_TYPE)
    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS, 1)
    blockspergrid_x = math.ceil(d.shape[1] / threadsperblock[1])
    blockspergrid_y = math.ceil(d.shape[0] / threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    cuda_exp[blockspergrid, threadsperblock](d, sigma, endresult)
    return endresult


@cuda.jit
def cuda_exp(d, sigma, endresult):
    x, y = cuda.grid(2)
    if (0 <= x < d.shape[1] and
            0 <= y < d.shape[0]):
        endresult[y, x] = math.exp(-d[y, x] / sigma[y, x])