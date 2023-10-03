
import numpy as np
from skimage import img_as_float32, filters
import cv2




def postprocess(raw, img=None, do_color_correction=True, do_tonemapping=True,
                do_gamma=True, do_sharpening=True, do_devignette=False, xyz2cam=None, sharpening_params=None):
    """
    Convert a raw image to jpg image.
    """
    if img is None:
        ## Rawpy processing - whole stack
        return img_as_float32(raw.postprocess(use_camera_wb=True))
    else:
        ## Color matrix
        '''
        if do_color_correction:
            ## First, read the color matrix from rawpy
            rgb2cam = get_color_matrix(raw, xyz2cam)
            cam2rgb = np.linalg.inv(rgb2cam)
            img = apply_ccm(img, cam2rgb)
            img = np.clip(img, 0.0, 1.0)
            '''
        ## Sharpening
        if do_sharpening:
            ## TODO: polyblur instead
            if sharpening_params is not None:
                img = filters.unsharp_mask(img, radius=sharpening_params['radius'],
                                           amount=sharpening_params['amount'],
                                           channel_axis=1, preserve_range=True)
            else:
                img = filters.unsharp_mask(img, radius=3,
                                           amount=0.5,
                                           channel_axis=1, preserve_range=True)
        ## Devignette
        if do_devignette:
            img = devignette(img)
        ## Tone mapping
        if do_tonemapping:
            img = apply_smoothstep(img)
        img = np.clip(img, 0.0, 1.0)
        ## Gamma compression
        if do_gamma:
            img = gamma_compression(img)
        img = np.clip(img, 0.0, 1.0)
        return img


def devignette(image):
    h, w, _ = image.shape
    vignette_filter = np.abs(np.linspace(-h / w * np.pi / 2, h / w * np.pi / 2, h))
    vignette_filter = np.outer(vignette_filter, np.abs(np.linspace(-np.pi / 2, np.pi / 2, w)))

    image_out = (2 - np.cos(vignette_filter) ** 4)[:, :, None] * image
    return image_out


def apply_smoothstep(image):
    """Apply global tone mapping curve."""
    # image_out = 3 * image**2 - 2 * image**3

    # tonemap = cv2.createTonemap(1.0)
    # image_out = tonemap.process(image)


    times = [1, 0.5, 2]
    images = [img_as_ubyte(np.clip(image * i, 0, 1)) for i in times]

    merge_mertens = cv2.createMergeMertens()
    image_out = merge_mertens.process(images)
    image_out = img_as_float32(image_out)

    image_out = 3 * image_out ** 2 - 2 * image_out ** 3
    return image_out


def gamma_compression(img, gamma=2.2):
    img = np.clip(img, a_min=0.0, a_max=1.0)
    return img ** (1. / gamma)