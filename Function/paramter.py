import warnings
import numpy as np


def check_params_validity(params, imshape):
    if params["grey method"] != "FFT":
        raise NotImplementedError("Grey level images should be obtained with FFT")

    assert params['scale'] >= 1
    if params['scale'] > 3:
        warnings.warn("Warning.... The required scale is superior to 3, but the algorighm can hardly go above.")

    if (not params['robustness']['on']) and (params['accumulated robustness denoiser']['median']['on'] or
                                             params['accumulated robustness denoiser']['gauss']['on'] or
                                             params['accumulated robustness denoiser']['merge']['on']):
        warnings.warn("Warning.... Robustness based denoising is enabled, "
                      "but robustness is disabled. No further denoising will be done.")

    assert params['merging']['kernel'] in ['handheld', 'iso']
    assert params['mode'] in ["bayer", 'grey']

    if (params['accumulated robustness denoiser']['median']['on'] and
            params['accumulated robustness denoiser']['gauss']['on']):
        warnings.warn("Warning.... 2 post processing blurrings are enabled. Is it a mistake?")

    assert params['kanade']['tuning']['kanadeIter'] > 0
    assert params['kanade']['tuning']['sigma blur'] >= 0

    assert len(imshape) == 2

    Ts = params['block matching']['tuning']['tileSizes'][0]

    # Checking if block matching is possible
    padded_imshape_x = Ts * (int(np.ceil(imshape[1] / Ts)))
    padded_imshape_y = Ts * (int(np.ceil(imshape[0] / Ts)))

    lvl_imshape_y, lvl_imshape_x = padded_imshape_y, padded_imshape_x
    for lvl, (factor, ts) in enumerate(
            zip(params['block matching']['tuning']['factors'], params['block matching']['tuning']['tileSizes'])):
        lvl_imshape_y, lvl_imshape_x = np.floor(lvl_imshape_y / factor), np.floor(lvl_imshape_x / factor)

        n_tiles_y = lvl_imshape_y / ts
        n_tiles_x = lvl_imshape_x / ts

        if n_tiles_y < 1 or n_tiles_x < 1:
            raise ValueError("Image of shape {} is incompatible with the given " \
                             "block matching tile sizes and factors : at level {}, " \
                             "coarse image of shape {} cannot be divided into " \
                             "tiles of size {}.".format(
                imshape, lvl,
                (lvl_imshape_y, lvl_imshape_x),
                ts))


def merge_params(dominant, recessive):
    """
    Merges 2 sets of parameters, one being dominant (= overwrittes the recessive
                                                     when a value a specified)
    """
    recessive_ = recessive.copy()
    for dom_key in dominant.keys():
        if (dom_key in recessive_.keys()) and type(dominant[dom_key]) is dict:
            recessive_[dom_key] = merge_params(dominant[dom_key], recessive_[dom_key])
        else:
            recessive_[dom_key] = dominant[dom_key]
    return recessive_


def params_process(SNR_params, custom_params, imshape, alpha, beta, std_curve, diff_curve):
    check_params_validity(SNR_params, imshape)
    if custom_params is not None:
        params = merge_params(dominant=custom_params, recessive=SNR_params)
        check_params_validity(params, imshape)
    #### adding metadatas to dict
    if not 'noise' in params['merging'].keys():
        params['merging']['noise'] = {}

    params['merging']['noise']['alpha'] = alpha
    params['merging']['noise']['beta'] = beta

    params['robustness']['std_curve'] = std_curve
    params['robustness']['diff_curve'] = diff_curve
    # copying parameters values in sub-dictionaries
    if 'scale' not in params["merging"].keys():
        params["merging"]["scale"] = params["scale"]
    if 'scale' not in params['accumulated robustness denoiser'].keys():
        params['accumulated robustness denoiser']["scale"] = params["scale"]
    if 'tileSize' not in params["kanade"]["tuning"].keys():
        params["kanade"]["tuning"]['tileSize'] = params['block matching']['tuning']['tileSizes'][0]
    if 'tileSize' not in params["robustness"]["tuning"].keys():
        params["robustness"]["tuning"]['tileSize'] = params['kanade']['tuning']['tileSize']
    if 'tileSize' not in params["merging"]["tuning"].keys():
        params["merging"]["tuning"]['tileSize'] = params['kanade']['tuning']['tileSize']

    if 'mode' not in params["kanade"].keys():
        params["kanade"]["mode"] = params['mode']
    if 'mode' not in params["robustness"].keys():
        params["robustness"]["mode"] = params['mode']
    if 'mode' not in params["merging"].keys():
        params["merging"]["mode"] = params['mode']
    if 'mode' not in params['accumulated robustness denoiser'].keys():
        params['accumulated robustness denoiser']["mode"] = params['mode']

    # deactivating robustness accumulation if robustness is disabled
    params['accumulated robustness denoiser']['median']['on'] &= params['robustness']['on']
    params['accumulated robustness denoiser']['gauss']['on'] &= params['robustness']['on']
    params['accumulated robustness denoiser']['merge']['on'] &= params['robustness']['on']

    params['accumulated robustness denoiser']['on'] = \
        (params['accumulated robustness denoiser']['gauss']['on'] or
         params['accumulated robustness denoiser']['median']['on'] or
         params['accumulated robustness denoiser']['merge']['on'])
    # if robustness aware denoiser is in merge mode, copy in merge params
    if params['accumulated robustness denoiser']['merge']['on']:
        params['merging']['accumulated robustness denoiser'] = params['accumulated robustness denoiser']['merge']
    else:
        params['merging']['accumulated robustness denoiser'] = {'on': False}
    return params