import os
import time
import numpy as np
import torch as th
from numba import cuda,float32
import matplotlib.pyplot as plt
import math

from Function.raw2grey import ref_read, compute_grey_images, hr_read, ausrichtung
from Function.noise import estimate_Noise_curves
from Function.paramter import params_process
from Function.Time import getTime
from Function.pyramid import init_block_matching
from Function.Localstats import init_robustness
from Function.gradient import init_ICA
from Function.Alignment import align_image_block_matching
from Function.ICA import ICA_optical_flow
from Function.Robustness import compute_robustness
from Function.comp_merge import merge
from Function.kernel_estimation import estimate_kernels
from Function.ref_merge import merge_ref
from Function.basic_math import add,divide, multiple, calculate_A, exp
from Function.denoising import frame_count_denoising_gauss,frame_count_denoising_median
from Function.Postprocess import postprocess

DEFAULT_NUMPY_FLOAT_TYPE = np.float32
DEFAULT_CUDA_FLOAT_TYPE = float32

DEFAULT_TORCH_FLOAT_TYPE = th.float32
DEFAULT_TORCH_COMPLEX_TYPE = th.complex64
DEFAULT_THREADS = 16


options = {'verbose' : 1}
custom_params={
        "scale":2,
        "merging" : {
            'kernel': 'handheld'},
        'post processing' : {'on': False}
        # post processing is enabled by default,
        # but it can be turned off here
        }
n_patches = int(1e5)
n_brightness_levels = 1024
alpha = 1.80710882e-4
beta = 3.1937599182128e-6
tol = 3

currenttime, verbose_1, verbose_2 = (time.perf_counter(),
                                         options['verbose'] >= 1,
                                         options['verbose'] >= 2)
params = {}
'''
0 basic set
including
read foto
noise curve
parameter set
'''
lr_path = './fliegen_lr/'
hr_path= './fliegen_hr/'
ref_raw, raw_comp=ref_read(lr_path)
#ref_raw_test = ausrichtung(ref_raw, -1)
#ref_raw, raw_comp= hr_read(hr_path)
std_curve, diff_curve= estimate_Noise_curves(tol,alpha,beta,n_brightness_levels)
brightness = np.mean(ref_raw)
id_noise = round(1000*brightness)
std = std_curve[id_noise]
SNR = brightness/std

Ts = 10
SNR_params = {'scale': 2,  # upscaling factor ( >=1 )
              'mode': 'grey',  # 'bayer' or 'grey' (input image type)
              'grey method': 'FFT',  # method to compute grey image for alignment. Only FFT is supported !
              'debug': False,  # when True, a dict is returned with debug infos.
              'block matching': {
                  'tuning': {
                      # WARNING: these parameters are defined fine-to-coarse!
                      'factors': [1, 2, 2, 2],
                      'tileSizes': [Ts, Ts, Ts, Ts],
                      'searchRadia': [5, 5, 5, 5],
                      'distances': ['L2', 'L2', 'L2', 'L2'],
                  }},
              'kanade': {
                  'tuning': {
                      'kanadeIter': 3,  # 3
                      # gaussian blur before computing grads. If 0, no blur is applied
                      'sigma blur': 0,  # grayscale by FFT already induces low pass filtering
                  }},
              'robustness': {
                  'on': True,
                  'tuning': {
                      't': 0.12,  # 0.12
                      's1': 2,  # 2
                      's2': 12,  # 12
                      'Mt': 1,  # 0.8
                  }
              },
              'merging': {
                  'kernel': 'handheld',  # 'iso' for isotropic kernel, 'handheld' for handhel kernel
                  'tuning': {
                      'k_detail': 0.25,#+ (0.33 - 0.25) * (30 - SNR) / (30 - 6),   #[0.25, ..., 0.33]
                      'k_denoise': 3,#+ (5 - 3) * (30 - SNR) / (30 - 6),   #[3.0, ...,5.0]
                      'l1_max': 15,#+ (0.81 - 0.71) * (30 - SNR) / (30 - 6),   #[0.001, ..., 0.010]
                      'l1_min': 6.5,#+ (1.24 - 1) * (30 - SNR) / (30 - 6),   #[0.006, ..., 0.020]
                      'k_stretch': 8,  # 4
                      'k_shrink': 4,  # 2
                  }
              },

              'accumulated robustness denoiser': {
                  'median': {'on': False,  # post process median filter
                             'radius max': 3,
                             # maximum radius of the median filter. Tt cannot be more than 14. for memory purpose
                             'max frame count': 8},

                  'gauss': {'on': False,  # post process gaussian filter
                            'sigma max': 1.5,  # std of the gaussian blur applied when only 1 frame is merged
                            'max frame count': 8},  # number of merged frames above which no blur is applied

                  'merge': {'on': True,  # filter for the ref frame accumulation
                            'rad max': 2,  # max radius of the accumulation neighborhod
                            'max multiplier': 5,  # Multiplier of the covariance for single frame SR
                            'max frame count': 5}  # # number of merged frames above which no blur is applied
              },
              'post processing': {
                  'on': True,
                  'do color correction': False,
                  'do tonemapping': False,
                  'do gamma': True,
                  'do sharpening': True,
                  'do devignette': False,

                  'sharpening': {
                      'radius': 3.6,
                      'amount': 3
                  }
              }
              }

params = params_process(SNR_params, custom_params, ref_raw.shape,alpha,beta,std_curve, diff_curve)


'''
1 Vorarbeitung
including
1.1 raw to grey
1.2 Block Matching
1.3 gradient and hessian matrix
1.4 Local stats
'''
verbose = options['verbose'] >= 1
verbose_2 = options['verbose'] >= 2
verbose_3 = options['verbose'] >= 3

bayer_mode = params['mode'] == 'bayer'

debug_mode = params['debug']
debug_dict = {"robustness": [],
              "flow": []}

accumulate_r = params['accumulated robustness denoiser']['on']

#### Moving to GPU
cuda_ref_img = cuda.to_device(ref_raw.astype(np.float32))
cuda.synchronize()
if verbose:
    print("\nProcessing reference image ---------\n")
    t1 = time.perf_counter()

    #### Raw to grey
    if bayer_mode:
        grey_method = params['grey method']

        cuda_ref_grey = compute_grey_images(cuda_ref_img, grey_method)
    else:
        grey_method = params['grey method']
        cuda_ref_grey = cuda_ref_img
    if verbose_3:
        cuda.synchronize()
        getTime(t1, "- Ref grey image estimated by {}".format(grey_method))

# Block Matching
if verbose_2:
    cuda.synchronize()
    current_time = time.perf_counter()
    print('\nBeginning Block Matching initialisation')

reference_pyramid = init_block_matching(cuda_ref_grey, options, params['block matching'])

if verbose_2:
    cuda.synchronize()
    current_time = getTime(current_time, 'Block Matching initialised (Total)')

# gradient and hessian
if verbose_2:
    cuda.synchronize()
    current_time = time.perf_counter()
    print('\nBeginning ICA initialisation')

ref_gradx, ref_grady, hessian = init_ICA(cuda_ref_grey, options, params['kanade'])
if verbose_2:
    cuda.synchronize()
    current_time = getTime(current_time, 'ICA initialised (Total)')

# local stats estimation
if verbose_2:
    cuda.synchronize()
    current_time = time.perf_counter()
    print("\nEstimating ref image local stats")

ref_local_stats = init_robustness(cuda_ref_img, options, params['robustness'])

if accumulate_r:
    accumulated_r = cuda.to_device(np.zeros(ref_local_stats.shape[:2]))

if verbose_2:
    cuda.synchronize()
    current_time = getTime(current_time, 'Local stats estimated (Total)')

'''
2 Comp foto verarbeitung
2.1 Alignment
2.2 ICA loop
2.3 Robustness
2.4 Kernel estimation
2.5 merg
'''
scale = params["scale"]
native_imshape_y, native_imshape_x = cuda_ref_img.shape
output_size = (round(scale * native_imshape_y), round(scale * native_imshape_x))
num = cuda.to_device(np.zeros(output_size + (1,), dtype=DEFAULT_NUMPY_FLOAT_TYPE))
den = cuda.to_device(np.zeros(output_size + (1,), dtype=DEFAULT_NUMPY_FLOAT_TYPE))
if verbose:
    cuda.synchronize()
    getTime(t1, '\nRef Img processed (Total)')

n_images = raw_comp.shape[0]
for im_id in range(n_images):
    if verbose:
        cuda.synchronize()
        print("\nProcessing image {} ---------\n".format(im_id + 1))
        im_time = time.perf_counter()

    #### Moving to GPU
    cuda_img = cuda.to_device(raw_comp[im_id])
    if verbose_3:
        cuda.synchronize()
        current_time = getTime(im_time, 'Arrays moved to GPU')

    #### Compute Grey Images
    if bayer_mode:
        #cuda_im_grey = compute_grey_images(raw_comp[im_id], grey_method)
        print('false Mode')
        if verbose_3:
            cuda.synchronize()
            current_time = getTime(current_time, "- grey images estimated by {}".format(grey_method))
    else:
        cuda_im_grey = cuda_img

    #### Block Matching
    if verbose_2:
        cuda.synchronize()
        current_time = time.perf_counter()
        print('Beginning block matching')
    #TODO：调整Ts的大小，使alignment尽量统一
    pre_alignment = align_image_block_matching(cuda_im_grey, reference_pyramid, options, params['block matching'])

    if verbose_2:
        cuda.synchronize()
        current_time = getTime(current_time, 'Block Matching (Total)')

    #### ICA
    if verbose_2:
        cuda.synchronize()
        current_time = time.perf_counter()
        print('\nBeginning ICA alignment')

    cuda_final_alignment = ICA_optical_flow(
        cuda_im_grey, cuda_ref_grey, ref_gradx, ref_grady, hessian, pre_alignment, options, params['kanade'])

    if debug_mode:
        debug_dict["flow"].append(cuda_final_alignment.copy_to_host())

    if verbose_2:
        cuda.synchronize()
        current_time = getTime(current_time, 'Image aligned using ICA (Total)')
    #### Robustness
    if verbose_2:
        cuda.synchronize()
        current_time = time.perf_counter()
        print('\nEstimating robustness')

    r_robustness = compute_robustness(cuda_img, ref_local_stats, cuda_final_alignment,
                                         options, params['robustness'])
    cuda_robustness = r_robustness[0]
    R_wert = r_robustness[1]
    S_wert = r_robustness[2]
    d_sq_wert = r_robustness[3]
    sigma_sq_wert = r_robustness[4]
    d_p_wert = r_robustness[5]
    exp_result=exp(d_sq_wert,sigma_sq_wert)

    if accumulate_r:
        add(accumulated_r, cuda_robustness)

    if verbose_2:
        cuda.synchronize()
        current_time = getTime(current_time, 'Robustness estimated (Total)')

    #### Kernel estimation
    if verbose_2:
        cuda.synchronize()
        current_time = time.perf_counter()
        print('\nEstimating kernels')

    covs_tuple= estimate_kernels(cuda_img, options, params['merging'])
    cuda_kernels = covs_tuple[0]
    e = covs_tuple[1]
    k_wert = covs_tuple[2]
    l_wert = covs_tuple[3]

    A = cuda.to_device(np.zeros_like(l_wert[:,:,0], dtype=DEFAULT_NUMPY_FLOAT_TYPE))
    A = calculate_A(l_wert[:,:,0],l_wert[:,:,1],A)
    if verbose_2:
        cuda.synchronize()
        current_time = getTime(current_time, 'Kernels estimated (Total)')

    #### Merging
    if verbose_2:
        current_time = time.perf_counter()
        print('\nAccumulating Image')
    w_wert = cuda.to_device(np.zeros(output_size + (3,3), dtype=DEFAULT_NUMPY_FLOAT_TYPE))
    merge(cuda_img, cuda_final_alignment, cuda_kernels, cuda_robustness, num, den,
          options, params['merging'],w_wert)

    if verbose_2:
        cuda.synchronize()
        current_time = getTime(current_time, 'Image accumulated (Total)')
    if verbose:
        cuda.synchronize()
        getTime(im_time, '\nImage processed (Total)')

    if debug_mode:
        debug_dict['robustness'].append(cuda_robustness.copy_to_host())

'''
4. process with ref
4.1 merg with ref
'''
#### Ref kernel estimation
if verbose_2 :
    cuda.synchronize()
    current_time = time.perf_counter()
    print('\nEstimating kernels')

ref_covs_tuple= estimate_kernels(cuda_ref_img, options, params['merging'])
cuda_kernels_ref = ref_covs_tuple[0]
e_ref = ref_covs_tuple[1]
k_wert_ref = ref_covs_tuple[2]
l_wert_ref = ref_covs_tuple[3]

if verbose_2 :
    cuda.synchronize()
    current_time = getTime(current_time, 'Kernels estimated (Total)')

#### Merge ref
if verbose_2 :
    cuda.synchronize()
    print('\nAccumulating ref Img')

if accumulate_r:
    merge_ref(cuda_ref_img, cuda_kernels_ref,
              num, den,
              params["merging"], accumulated_r)
else:
    merge_ref(cuda_ref_img, cuda_kernels_ref,
              num, den,
              params["merging"])

if verbose_2 :
    cuda.synchronize()
    getTime(current_time, 'Ref Img accumulated (Total)')

# num is outwritten into num/den
handheld_output=divide(num, den)

if verbose_2 :
    print('\n------------------------')
    cuda.synchronize()
    current_time = getTime(current_time, 'Image normalized (Total)')

if verbose :
    print('\nTotal ellapsed time : ', time.perf_counter() - t1)

if accumulate_r :
    debug_dict['accumulated robustness'] = accumulated_r


'''
denoising
there are median denoising and gauss denoising
but they are not sure useful for grey foto
so right now not to run
'''
median_params = params['accumulated robustness denoiser']['median']
gauss_params = params['accumulated robustness denoiser']['gauss']

median = median_params['on']
gauss = gauss_params['on']
post_frame_count_denoise = (median or gauss)

params_pp = params['post processing']
post_processing_enabled = params_pp['on']

if post_frame_count_denoise or post_processing_enabled:
    if verbose_1:
        print('Beginning post processing')

if post_frame_count_denoise :
    if verbose_2:
        print('-- Robustness aware bluring')

    if median:
        handheld_output = frame_count_denoising_median(handheld_output, debug_dict['accumulated robustness'],
                                                       median_params)
    if gauss:
        handheld_output = frame_count_denoising_gauss(handheld_output, debug_dict['accumulated robustness'],
                                                      gauss_params)

'''
Post process
'''
if post_processing_enabled:
    if verbose_2:
        print('-- Post processing image')

    output_image = postprocess(ref_raw, handheld_output,
                               params_pp['do color correction'],
                               params_pp['do tonemapping'],
                               params_pp['do gamma'],
                               params_pp['do sharpening'],
                               params_pp['do devignette'],
                               params_pp['sharpening']
                               )
else:
    output_image = handheld_output

#output
'''output_img = output_image.copy_to_host().squeeze()
os.makedirs('results', exist_ok=True)
output_img.astype(np.float32)
output_img.tofile('results/2scale_fliegen'+str(ref_id)+'.raw')'''