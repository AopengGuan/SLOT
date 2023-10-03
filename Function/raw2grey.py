import os
from pathlib import Path
import numpy as np
from numba import cuda,float32
import glob
import math
import torch as th


DEFAULT_NUMPY_FLOAT_TYPE = np.float32
DEFAULT_CUDA_FLOAT_TYPE = float32

DEFAULT_TORCH_FLOAT_TYPE = th.float32
DEFAULT_TORCH_COMPLEX_TYPE = th.complex64
DEFAULT_THREADS = 16

def ref_read(foto_path, ref_id=0,white_level=1024, black_level=0,hr=False):
    if not hr:
        ref_path=Path(foto_path)
        raw_path_list = glob.glob(os.path.join(ref_path.as_posix(), '*.raw'))
        ref_img = np.fromfile(raw_path_list[ref_id], dtype=np.int16, count=-1)
        ref_img=np.reshape(ref_img, (3,580,500))
        ref_raw=ref_img[0,:,:]
        raw_comp=[]
        for index, raw_path in enumerate(raw_path_list):
            if index != ref_id:
                comp_img=np.fromfile(raw_path_list[index], dtype=np.int16, count=-1)
                comp_img= comp_img.astype(np.dtype('<i'), casting='safe')
                comp_img=np.reshape(comp_img, (3,580,500))
                comp_img=comp_img[0,:,:]
                raw_comp.append(comp_img)
        raw_comp = np.array(raw_comp)
        if np.issubdtype(type(ref_raw[0, 0]), np.integer):
            ref_raw = ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE)
            ref_raw=(ref_raw-black_level)/(white_level-black_level)
            ref_raw = np.clip(ref_raw, 0.0, 1.0)

        if np.issubdtype(type(raw_comp[0, 0, 0]), np.integer):
            raw_comp = raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE)
            # raw_comp is a (N, H,W) array
            raw_comp = raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE)
            raw_comp=(raw_comp-black_level)/(white_level-black_level)
            raw_comp = np.clip(raw_comp, 0.0, 1.0)
    elif hr:
        ref_path = Path(foto_path)
        raw_path_list = glob.glob(os.path.join(ref_path.as_posix(), '*.raw'))
        ref_img = np.fromfile(raw_path_list[ref_id], dtype=np.float32, count=-1)
        ref_img = np.reshape(ref_img, (1, 1160, 1000))
        ref_raw = ref_img[0, :, :]
        raw_comp = []
        for index, raw_path in enumerate(raw_path_list):
            if index != ref_id:
                comp_img=np.fromfile(raw_path_list[index], dtype=np.float32, count=-1)
                comp_img= comp_img.astype(np.dtype('<f'), casting='safe')
                comp_img=np.reshape(comp_img, (1,1160,1000))
                comp_img=comp_img[0,:,:]
                raw_comp.append(comp_img)
        raw_comp = np.array(raw_comp)
    return ref_raw, raw_comp

def hr_read(foto_path, ref_id=0,white_level=1024, black_level=0):
    ref_path = Path(foto_path)
    raw_path_list = glob.glob(os.path.join(ref_path.as_posix(), '*.raw'))
    hr_img = np.fromfile(raw_path_list[ref_id], dtype=np.int16, count=-1)
    hr_img = np.reshape(hr_img, (3, 4640,4000))
    hr_raw = hr_img[0, :, :]
    raw_comp = []
    for i in range(8):
        for j in range(8):
            im = np.ones([580, 500])
            for n in range(580):
                for m in range(500):
                    im[n, m] = hr_raw[8 * n + i, 8 * m + j]
            raw_comp.append(im)
    raw_comp = np.array(raw_comp)
    ref_raw = raw_comp[31,:,:]
    #if np.issubdtype(type(ref_raw[0, 0]), np.integer):
    ref_raw = ref_raw.astype(DEFAULT_NUMPY_FLOAT_TYPE)
    ref_raw=(ref_raw-black_level)/(white_level-black_level)
    ref_raw = np.clip(ref_raw, 0.0, 1.0)

    #if np.issubdtype(type(raw_comp[0, 0, 0]), np.integer):
    raw_comp = raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE)
    # raw_comp is a (N, H,W) array
    raw_comp = raw_comp.astype(DEFAULT_NUMPY_FLOAT_TYPE)
    raw_comp=(raw_comp-black_level)/(white_level-black_level)
    raw_comp = np.clip(raw_comp, 0.0, 1.0)
    return ref_raw, raw_comp


def compute_grey_images(img, method):
    """
    This function converts a raw image to a grey image, using the decimation or
    the method of Alg. 3: ComputeGrayscaleImage

    Parameters
    ----------
    img : device Array[:, :]
        Raw image J to convert to gray level.
    method : str
        FFT or decimatin.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    img_grey : device Array[:, :]
        Corresponding grey scale image G

    """
    imsize_y, imsize_x = img.shape
    if method == "FFT":
        torch_img_grey = th.as_tensor(img, dtype=DEFAULT_TORCH_FLOAT_TYPE, device="cuda")
        torch_img_grey = th.fft.fft2(torch_img_grey)
        # th FFT induces copy on the fly : this is good because we dont want to
        # modify the raw image, it is needed in the future
        # Note : the complex dtype of the fft2 is inherited from DEFAULT_TORCH_FLOAT_TYPE.
        # Therefore, for DEFAULT_TORCH_FLOAT_TYPE = float32 we directly get complex64
        torch_img_grey = th.fft.fftshift(torch_img_grey)

        torch_img_grey[:imsize_y // 2, :] = 0
        torch_img_grey[:, :imsize_x // 2] = 0
        torch_img_grey[-imsize_y // 2:, :] = 0
        torch_img_grey[:, -imsize_x // 2:] = 0

        torch_img_grey = th.fft.ifftshift(torch_img_grey)
        torch_img_grey = th.fft.ifft2(torch_img_grey)
        # Here, .real() type inherits once again from the complex type.
        # numba type is read directly from the torch tensor, so everything goes fine.
        return cuda.as_cuda_array(torch_img_grey.real)
    elif method == "decimating":
        # decimating方法得到的灰度图像大小为输入图像的一半
        grey_imshape_y, grey_imshape_x = grey_imshape = imsize_y // 2, imsize_x // 2

        img_grey = cuda.device_array(grey_imshape, DEFAULT_NUMPY_FLOAT_TYPE)

        threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
        blockspergrid_x = math.ceil(grey_imshape_x / threadsperblock[1])
        blockspergrid_y = math.ceil(grey_imshape_y / threadsperblock[0])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        cuda_decimate_to_grey[blockspergrid, threadsperblock](img, img_grey)
        return img_grey

    else:
        raise NotImplementedError('Computation of gray level on GPU is only supported for FFT')


@cuda.jit
def cuda_decimate_to_grey(img, grey_img):
    x, y = cuda.grid(2)
    grey_imshape_y, grey_imshape_x = grey_img.shape

    if (0 <= y < grey_imshape_y and
            0 <= x < grey_imshape_x):
        c = 0
        for i in range(0, 2):
            for j in range(0, 2):
                c += img[2 * y + i, 2 * x + j]
        grey_img[y, x] = c / 4


def ausrichtung(img, step):
    imshape_y, imshape_x= img.shape
    img_done = cuda.device_array([imshape_y,imshape_x], DEFAULT_NUMPY_FLOAT_TYPE)

    threadsperblock = (DEFAULT_THREADS, DEFAULT_THREADS)
    blockspergrid_x = math.ceil(imshape_x / threadsperblock[1])
    blockspergrid_y = math.ceil(imshape_y / threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    cuda_ausrichtung[blockspergrid, threadsperblock](img, img_done, step)
    return img_done

@cuda.jit
def cuda_ausrichtung(img, img_done, step):
    x, y = cuda.grid(2)
    imshape_y, imshape_x = img.shape

    if (0 <= y < imshape_y and
            0 <= x < imshape_x):
        if (y%2 == 0):
            img_done[y,x] = img[y,x]
        else:
            img_done[y,x] = img[y,x+step]

