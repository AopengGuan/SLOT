import numpy as np
import multiprocessing
from multiprocessing.pool import ThreadPool
from functools import partial
from tqdm import tqdm
import sys

n_patches = int(1e5)
def estimate_Noise_curves(TOL,alpha,beta,n_brightness_levels):
    print("Estimating noise curves ...")
    tol_sq = TOL * TOL
    xmin = tol_sq / 2 * (alpha + np.sqrt(tol_sq * alpha * alpha + 4 * beta))
    xmax = (2 + tol_sq * alpha - np.sqrt((2 + tol_sq * alpha) ** 2 - 4 * (1 + tol_sq * beta))) / 2
    imin = int(np.ceil(xmin * n_brightness_levels)) + 1
    imax = int(np.floor(xmax * n_brightness_levels)) - 1

    std_curve = np.empty(n_brightness_levels + 1)
    diff_curve = np.empty(n_brightness_levels + 1)
    brigntess_normal = np.arange(n_brightness_levels + 1) / n_brightness_levels
    nl_brigntess = np.concatenate((brigntess_normal[:imin + 1], brigntess_normal[imax:]))
    if nl_brigntess.size > 500:
        return
    else:
        if multiprocessing.cpu_count() > 1:
            N_CPU = multiprocessing.cpu_count() - 1
            print("multiCPU")
            multiprocessing.freeze_support()
        else:
            N_CPU = 1
            print("one CPU")
            multiprocessing.freeze_support()
        print("Prozess start")
        pool = ThreadPool(processes=N_CPU)
        func = partial(unitary_MC, alpha, beta)

        sigma_nl = np.empty_like(nl_brigntess)
        diffs_nl = np.empty_like(nl_brigntess)
        for b_, result in enumerate(
                tqdm(pool.imap(func, list(nl_brigntess)), total=nl_brigntess.size, desc="Brightnesses")):
            diffs_nl[b_] = result[0]
            sigma_nl[b_] = result[1]
        pool.close()

        std_curve[:imin + 1], diff_curve[:imin + 1] = sigma_nl[:imin + 1], diffs_nl[:imin + 1]
        std_curve[imax:], diff_curve[imax:] = sigma_nl[imin + 1:], diffs_nl[imin + 1:]

        # padding using linear interpolation
        brightness_l = brigntess_normal[imin - 1:imax + 2]

        '''
        Interpolates the missing values for diff and and sigma, based on their
        upper and lower bounds.
        '''
        norm_b = (brightness_l - brightness_l[0]) / (brightness_l[-1] - brightness_l[0])

        sigmas_sq_lin = norm_b * (std_curve[imax] ** 2 - std_curve[imin] ** 2) + std_curve[imin] ** 2
        diffs_sq_lin = norm_b * (diff_curve[imax] ** 2 - diff_curve[imin] ** 2) + diff_curve[imin] ** 2
        sigmas_l = np.sqrt(sigmas_sq_lin[1:-1])
        diffs_l = np.sqrt(diffs_sq_lin[1:-1])
        std_curve[imin:imax + 1] = sigmas_l
        diff_curve[imin:imax + 1] = diffs_l
        return std_curve,diff_curve

def unitary_MC(alpha, beta, b):
    """
    Runs a MC scheme to estimate sigma and d for a given brightness, alpha and
    beta.

    Parameters
    ----------
    alpha : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    b : float in 0, 1
        brighntess

    Returns
    -------
    diff_mean : float
        mean difference
    std_mean : float
        mean standard deviation

    """
    # create the patch
    patch = np.ones((n_patches, 3, 3)) * b

    # add noise and clip
    patch1 = patch + np.sqrt(patch * alpha + beta) * np.random.randn(*patch.shape)
    patch1 = np.clip(patch1, 0.0, 1.0)

    patch2 = patch + np.sqrt(patch * alpha + beta) * np.random.randn(*patch.shape)
    patch2 = np.clip(patch2, 0.0, 1.0)

    # compute statistics
    std_mean = 0.5 * np.mean((np.std(patch1, axis=(1, 2)) + np.std(patch2, axis=(1, 2))))

    curr_mean1 = np.mean(patch1, axis=(1, 2))
    curr_mean2 = np.mean(patch2, axis=(1, 2))
    diff_mean = np.mean(np.abs(curr_mean1 - curr_mean2))

    return diff_mean, std_mean

def optimize_function(tols, alphas, betas, std_curve_dict, diff_curve_dict):
    n_brightness_levels = 1024
    for tol in tols:
        for alpha in alphas:
            for beta in betas:
                key = (tol, alpha, beta)
                if key not in std_curve_dict:
                    result = estimate_Noise_curves(tol, alpha, beta, n_brightness_levels)
                    if result is None:
                        print('nl_brightness size ist groesser als 500')
                    else:
                        std_curve_dict[key] = result[0]
                        diff_curve_dict[key] = result[1]
                        print('TOL: ',tol,',alpha: ',alpha,',beta: ',beta,' is done')
