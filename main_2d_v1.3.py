"""
v1.3

- Improved monopole calculation
- Added option to save/load pair counts to avoid recomputation
- Added contours to xi(σ, π) plot

v1.2

- Compute and plot monopole 

v1.0: Initial version, based on v3.2 code but focused on 2D correlation function ξ(σ, π) only.

Compute and plot the 2D correlation function ξ(σ, π) for the full sample and for
bins in dist_fil. All diagnostic plots (redshift‑magnitude, RA/Dec distributions,
redshift histograms) are also produced.

Based on original v3.2 code, stripped of 1D xi(s) calculations.
"""

import os
import shutil
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import curve_fit
from numpy.polynomial.polynomial import Polynomial
from matplotlib.colors import LogNorm
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import gaussian_kde

# ---------------------------
# PARAMETERS
# ---------------------------

# ---- Sample ----------
sample = 'nyu'
test_dilute = 1                     # fraction of galaxies to keep (1.0 = full)
sigma = 5.0
h = 0.6774
zmin, zmax = 0.07, 0.2
mag_max = -21.2
ran_method = 'random_choice'           # ['random_choice', 'piecewise', 'poly']
if ran_method == 'poly':
    deg = 5
gr_min = 0.8

# ------ 2D correlation function parameters ------
min_sep_2d = 1.0          # minimum σ, π in Mpc/h
max_sep_2d = 150.0
bin_size_2d = 2.0
pi_rebin = 2                # rebin factor for π direction

# ------ dist_fil binning ------
dist_bin_mode = "percentile_intervals"
# Options: "percentile", "fixed", "equal_width", "custom_intervals", "tails"
dist_bin_intervals = [ # used only for "custom_intervals" mode
    [(0, 2)],
    [(10, 100)],
]
dist_bin_percentile_intervals  = [ # used only for "percentile_intervals" mode
    (0, 20),      # a–bth percentile
    (70, 90)     # c–dth percentile
]
nbins_dist = 3              # used only for percentile / equal_width
dist_bin_edges = [0, 5, 30] # used only for "fixed"

# ------ Random catalog parameters ------
nside = 256
nrand_mult = 15              # Nr / Nd

# --- Method for generating RA/Dec ---
ran_radec_method = 'file'    # 'healpix' or 'file'
if nrand_mult >= 50:
    RADec_filepath = '../data/lss_randoms_combined_cut_LARGE.csv'
else:
    RADec_filepath = '../data/lss_randoms_combined_cut.csv'

# ------ Output folder --------
folderName = f'XISIGMAPI_z{zmin:.2f}-{zmax:.2f}_mag{mag_max:.1f}_gr{ (f"{gr_min:.1f}") if "gr_min" in locals() else "none" }_sigma{sigma}_nrand{nrand_mult}_RADECmethod{ran_radec_method}'
output_folder = f"../plots/{folderName}/"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# ------ Pair counts folder ----
paircounts_dir = "../data/pair_counts/"
os.makedirs(paircounts_dir, exist_ok=True)

# ---------------------------
# COSMOLOGY
# ---------------------------
cosmo = FlatLambdaCDM(H0=h * 100, Om0=0.3089)

# ---------------------------
# GENERAL HELPER FUNCTIONS
# ---------------------------

def safe_trapz(y: np.ndarray, x: np.ndarray) -> float:
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)

def spherical_to_cartesian(ra, dec, r):
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    cos_dec = np.cos(dec_rad)
    x = r * cos_dec * np.cos(ra_rad)
    y = r * cos_dec * np.sin(ra_rad)
    z = r * np.sin(dec_rad)
    return x, y, z

def ensure_dir_exists(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)

def save_figure(fig, path: str, dpi: int = 300):
    ensure_dir_exists(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def get_paircounts_filename(bin_name, params):
    """
    Generate a filename for saved pair counts based on key parameters.
    params should be a dict containing at least:
        sample, sigma, h, zmin, zmax, mag_max, gr_min,
        min_sep_2d, max_sep_2d, bin_size_2d, pi_rebin,
        nrand_mult, ran_radec_method, and the bin identifier (bin_name).
    """
    # Create a string with all relevant values
    parts = [
        f"sample={params['sample']}",
        f"sigma={params['sigma']}",
        f"h={params['h']:.4f}",
        f"z={params['zmin']:.2f}-{params['zmax']:.2f}",
        f"mag={params['mag_max']:.1f}",
        f"gr={params['gr_min']:.1f}",
        f"sep={params['min_sep_2d']}-{params['max_sep_2d']}",
        f"bin={params['bin_size_2d']}",
        f"pi_rebin={params['pi_rebin']}",
        f"nrand={params['nrand_mult']}",
        f"radec={params['ran_radec_method']}",
        f"bin={bin_name}",
        f"distbinmode={params['dist_bin_mode']}"
    ]
    # Join and replace dots to avoid filesystem issues
    fname = "_".join(parts).replace('.', 'p')
    return os.path.join(paircounts_dir, fname + ".npz")

# ---------------------------
# REDSHIFT DISTRIBUTION FITTING
# ---------------------------

def build_cdf_from_line(data, vmin, vmax, num_points=10000):
    hist, bin_edges = np.histogram(data, bins=40, range=(vmin, vmax), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def LinearFunction(x, a, b): return a * x + b
    def BreakFunction(x, a1, b1, a2, xb):
        yi = lambda x_: LinearFunction(x_, a1, b1)
        yo = lambda x_: LinearFunction(xb, a1, b1) + ((x_ - xb) * a2)
        return np.piecewise(x, [x < xb, x >= xb], [yi, yo])

    bounds = [[-np.inf, -np.inf, -np.inf, vmin], [np.inf, np.inf, np.inf, vmax]]
    popt, _ = curve_fit(BreakFunction, bin_centers, hist, bounds=bounds)
    z_vals = np.linspace(vmin, vmax, num_points)
    pdf_vals = BreakFunction(z_vals, *popt)
    pdf_vals = np.clip(pdf_vals, a_min=0.0, a_max=None)
    integ = safe_trapz(pdf_vals, z_vals)
    if integ > 0:
        pdf_vals = pdf_vals / integ
    dz = z_vals[1] - z_vals[0]
    cdf_vals = np.cumsum(pdf_vals) * dz
    if cdf_vals[-1] > 0:
        cdf_vals = cdf_vals / cdf_vals[-1]
    cdf_inv = interp1d(cdf_vals, z_vals, bounds_error=False, fill_value=(vmin, vmax))
    return cdf_inv, z_vals, pdf_vals, cdf_vals

def build_cdf_from_parabola(data, vmin, vmax, deg, num_points=10000):
    hist, bin_edges = np.histogram(data, bins=40, range=(vmin, vmax), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    poly = Polynomial.fit(bin_centers, hist, deg=deg)
    z_vals = np.linspace(vmin, vmax, num_points)
    pdf_vals = poly(z_vals)
    pdf_vals = np.clip(pdf_vals, a_min=0.0, a_max=None)
    integ = safe_trapz(pdf_vals, z_vals)
    if integ > 0:
        pdf_vals = pdf_vals / integ
    dz = z_vals[1] - z_vals[0]
    cdf_vals = np.cumsum(pdf_vals) * dz
    if cdf_vals[-1] > 0:
        cdf_vals = cdf_vals / cdf_vals[-1]
    cdf_inv = interp1d(cdf_vals, z_vals, bounds_error=False, fill_value=(vmin, vmax))
    return cdf_inv, z_vals, pdf_vals, cdf_vals

def generate_random_red(redshift, nrand, ran_method, deg=None):
    if ran_method == "poly":
        cdf_inv_z, _, _, _ = build_cdf_from_parabola(redshift, redshift.min(), redshift.max(), deg)
        u = np.random.uniform(0.0, 1.0, nrand)
        red_random = cdf_inv_z(u)
    elif ran_method == "piecewise":
        cdf_inv_z, _, _, _ = build_cdf_from_line(redshift, redshift.min(), redshift.max())
        u = np.random.uniform(0.0, 1.0, nrand)
        red_random = cdf_inv_z(u)
    elif ran_method == "random_choice":
        red_random = np.random.choice(redshift, nrand)
    else:
        raise ValueError(f"Unknown ran_method: {ran_method}")
    return red_random

# ---------------------------
# WEIGHT COMPUTATION FUNCTIONS
# ---------------------------

def compute_dec_weights(data_dec, rand_dec, alpha=1.0, method="auto",
                        kde_threshold=1_000_000, nbins=40, spline_s=0.5,
                        bw_factor=1.2, n_grid=300):
    n_rand = len(rand_dec)
    if method == "auto":
        method = "kde" if n_rand < kde_threshold else "spline"
    print(f"Using {method} method for declination weights")
    epsilon = 1e-10

    if method == "kde":
        kde_data = gaussian_kde(data_dec)
        kde_data.set_bandwidth(kde_data.factor * bw_factor)
        kde_rand = gaussian_kde(rand_dec)
        kde_rand.set_bandwidth(kde_rand.factor * bw_factor)
        dec_min = min(data_dec.min(), rand_dec.min())
        dec_max = max(data_dec.max(), rand_dec.max())
        grid = np.linspace(dec_min, dec_max, n_grid)
        density_data_grid = kde_data(grid)
        density_rand_grid = kde_rand(grid)
        ratio_grid = (density_data_grid + epsilon) / (density_rand_grid + epsilon)
        weights = np.interp(rand_dec, grid, ratio_grid)
    elif method == "spline":
        hist_data, edges = np.histogram(data_dec, bins=nbins)
        hist_rand, _ = np.histogram(rand_dec, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ratio = (hist_data + epsilon) / (hist_rand + epsilon)
        spline = UnivariateSpline(centers, ratio, s=spline_s)
        weights = spline(rand_dec)
        weights = np.clip(weights, 0.01, None)
    else:
        raise ValueError("method must be 'auto', 'kde', or 'spline'")

    weights = 1.0 + alpha * (weights - 1.0)
    weights /= np.mean(weights)
    return weights

def apply_redshift_weights_spline(bin_galaxies, rand_catalog, target_kde, dec_weights):
    """
    Compute redshift weights using a spline ratio method and assign them to
    both the galaxy and random catalogs (in‑place).
    """
    z_bin = bin_galaxies["red"].values
    n_bins_z = 40
    hist_bin, bin_edges = np.histogram(z_bin, bins=n_bins_z, density=False)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    target_counts = target_kde(bin_centers) * len(z_bin) * (bin_edges[1] - bin_edges[0])
    eps = 1e-10
    ratio = (target_counts + eps) / (hist_bin + eps)
    spline_ratio = UnivariateSpline(bin_centers, ratio, s=0.5, ext='const')
    gal_weight_raw = spline_ratio(z_bin)
    gal_weight_raw = np.clip(gal_weight_raw, 0.1, 10.0)
    gal_weight = gal_weight_raw / np.mean(gal_weight_raw)
    bin_galaxies.loc[:, "weight"] = gal_weight

    rand_z = rand_catalog["red"].values
    rand_z_weight_raw = spline_ratio(rand_z)
    rand_z_weight_raw = np.clip(rand_z_weight_raw, 0.1, 10.0)
    combined_raw = rand_z_weight_raw * dec_weights
    combined = combined_raw / np.mean(combined_raw)
    rand_catalog["weight"] = combined

# ---------------------------
# RA/DEC GENERATION FUNCTIONS
# ---------------------------

def generate_random_radec_healpix(ra, dec, nside, num_randoms):
    npix = hp.nside2npix(nside)
    mask = np.zeros(npix, dtype=int)
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    pixels = hp.ang2pix(nside, theta, phi)
    mask[pixels] = 1

    ra_random = np.random.uniform(0.0, 360.0, num_randoms)
    u = np.random.uniform(-1.0, 1.0, num_randoms)
    dec_random_rad = np.arcsin(u)
    dec_random = np.degrees(dec_random_rad)

    theta_random = np.radians(90.0 - dec_random)
    phi_random = np.radians(ra_random)
    random_pixels = hp.ang2pix(nside, theta_random, phi_random)

    valid = mask[random_pixels] == 1
    ra_random = ra_random[valid]
    dec_random = dec_random[valid]

    while ra_random.shape[0] < num_randoms:
        additional_needed = num_randoms - ra_random.shape[0]
        ra_additional = np.random.uniform(0.0, 360.0, additional_needed * 2)
        u_additional = np.random.uniform(-1.0, 1.0, additional_needed * 2)
        dec_additional_rad = np.arcsin(u_additional)
        dec_additional = np.degrees(dec_additional_rad)
        theta_additional = np.radians(90.0 - dec_additional)
        phi_additional = np.radians(ra_additional)
        additional_pixels = hp.ang2pix(nside, theta_additional, phi_additional)
        valid_additional = mask[additional_pixels] == 1
        ra_valid = ra_additional[valid_additional]
        dec_valid = dec_additional[valid_additional]
        ra_random = np.concatenate((ra_random, ra_valid))
        dec_random = np.concatenate((dec_random, dec_valid))

    if ra_random.shape[0] < num_randoms:
        raise ValueError("Not enough random points generated.")
    return ra_random[:num_randoms], dec_random[:num_randoms]

def generate_master_radec(full_cat, nrand_total, nside, ran_radec_method,
                          ra_preload=None, dec_preload=None):
    if ran_radec_method == 'file':
        if ra_preload is None or dec_preload is None:
            raise ValueError("Method 'file' requires ra_preload and dec_preload.")
        if len(ra_preload) < nrand_total:
            raise ValueError(f"RA/Dec arrays contain {len(ra_preload)} points but {nrand_total} are needed.")
        ra_master = ra_preload[:nrand_total]
        dec_master = dec_preload[:nrand_total]
    elif ran_radec_method == 'healpix':
        print("Generating master RA/Dec from Healpix mask...")
        ra_master, dec_master = generate_random_radec_healpix(
            full_cat["ra"].values, full_cat["dec"].values, nside, nrand_total
        )
    else:
        raise ValueError(f"Unknown ran_radec_method: {ran_radec_method}")
    return ra_master, dec_master

# ---------------------------
# PLOTTING FUNCTIONS
# ---------------------------

def plot_redshift_k(cat):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist2d(cat["red"], cat["mag_abs_r"], bins=40, cmap="Blues", norm=LogNorm())
    ax.axvline(zmin, color="k", linestyle=":")
    ax.axvline(zmax, color="k", linestyle=":")
    ax.axhline(mag_max, color="k", linestyle=":")
    ax.invert_yaxis()
    ax.set_xlabel("Redshift")
    ax.set_ylabel("K")
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Number of galaxies')
    filename = f"../plots/{folderName}/redshift_magnitude.png"
    save_figure(fig, filename, dpi=100)

def plot_radec_distribution(cat, randoms, subsample=None):
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    axes[0].hist(randoms["ra"], bins=40, density=True, histtype="step", color="k", lw=1.5, label="Randoms")
    axes[0].hist(cat["ra"], bins=40, density=True, histtype="stepfilled", color="C00", alpha=0.8, label="Galaxies")
    axes[0].set_xlabel("RA")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[1].hist(cat["dec"], bins=40, density=True, histtype="stepfilled", color="C00", alpha=0.8, label="Galaxies")
    axes[1].hist(randoms["dec"], bins=40, density=True, histtype="step", color="k", lw=1.5, label="Randoms")
    if "weight" in randoms.columns:
        axes[1].hist(randoms["dec"], bins=40, density=True, weights=randoms["weight"],
                     linestyle='--', color='k', histtype='step', label="Weighted Randoms")
    axes[1].set_xlabel("Dec")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    if subsample is None:
        filename = f"../plots/{folderName}/radec_distribution.png"
        axtitle = 'Total sample'
    else:
        filename = f"../plots/{folderName}/radec_distribution_bin{subsample}.png"
        axtitle = rf"r_{'fil'} \in {cat['dist_fil'].min():.1f}-{cat['dist_fil'].max():.1f} Mpc/h"
    fig.suptitle(axtitle)
    plt.tight_layout()
    save_figure(fig, filename, dpi=100)

def plot_bin_data_and_randoms(gxs, rxs, label, plotname):
    fig, axes = plt.subplots(3, 1, figsize=(7, 14))
    axes[0].scatter(rxs["ra"], rxs["dec"], s=1.5, color="k", alpha=0.5, label="Randoms")
    axes[0].scatter(gxs["ra"], gxs["dec"], s=1, color="C00", label="Galaxies")
    axes[0].set_xlabel("RA")
    axes[0].set_ylabel("DEC")
    axes[0].legend(loc='upper right')
    axes[0].set_title(label)

    axes[1].hist(gxs["red"], bins=40, density=True, histtype="stepfilled",
                 color="C00", alpha=0.8, label="Galaxies (unweighted)")
    axes[1].hist(rxs["red"], bins=40, density=True, histtype="step",
                 color="k", lw=1.5, label="Randoms")
    if "weight" in rxs.columns:
        axes[1].hist(rxs["red"], bins=40, density=True, weights=rxs["weight"],
                     histtype="step", color="k", linestyle="--", lw=1.5, label="Randoms (weighted)")
    if "weight" in gxs.columns:
        axes[1].hist(gxs["red"], bins=40, density=True, weights=gxs["weight"],
                     histtype="step", color="C00", linestyle="--", lw=2,
                     label="Galaxies (weighted)")
    axes[1].set_xlabel("Redshift")
    axes[1].set_ylabel("PDF")
    axes[1].legend()

    axes[2].hist(gxs["dist_fil"], bins=40, density=True, color="C03", alpha=0.9)
    axes[2].set_xlabel(r"$r_{\rm fil}\,[h^{-1}\mathrm{Mpc}]$")
    axes[2].set_ylabel("PDF")

    plt.tight_layout()
    save_figure(fig, plotname, dpi=200)

# ---------------------------
# CATALOG LOADING / MANIPULATION
# ---------------------------

def load_catalog(sample_name):
    if sample_name == "nyu":
        if sigma == 5.0:
            datafile = "../data/sdss_dr72safe0_sigma_5.0.csv"
        elif sigma == 3.0:
            datafile = "../data/sdss_dr72safe0_sigma_3.0.csv"
    elif sample_name == "sdss":
        datafile = "../data/sdss_zmin_0.000_zmax_0.300_sigma_5.0.csv"
    else:
        raise ValueError("Invalid sample")

    cat = pd.read_csv(datafile)
    if sample_name != "nyu":
        cat["dist_fil"] /= h
    if 'gr_min' in locals():
        cat = cat[cat["gr"] > gr_min]
    return cat

def select_sample(cat):
    cat_z = cat[(cat["red"] >= zmin) & (cat["red"] <= zmax)]
    if sample == "nyu":
        cat_z_mag = cat_z[cat_z["mag_abs_r"] < mag_max].copy()
    else:
        cat_z_mag = cat_z[cat_z["mag_abs_r"] - 5 * np.log10(h) < mag_max].copy()
    return cat_z, cat_z_mag

def split_by_dist_fil_bins(cat_z_mag):
    values = cat_z_mag["dist_fil"].values

    if dist_bin_mode == "custom_intervals":
        bins = []
        labels = []
        for i, interval_list in enumerate(dist_bin_intervals):
            mask_total = np.zeros_like(values, dtype=bool)
            label_parts = []
            for lo, hi in interval_list:
                mask = (values >= lo) & (values <= hi)
                mask_total |= mask
                label_parts.append(f"{lo}-{hi}")
            subset = cat_z_mag.loc[mask_total].copy()
            bins.append(subset)
            label = " ∪ ".join(label_parts)
            labels.append(f"$r_{{fil}} \\in [{label}]$")
        return bins, labels, None

    elif dist_bin_mode == "percentile":
        percentiles = np.linspace(0, 100, nbins_dist + 1)
        edges = np.percentile(values, percentiles)
    elif dist_bin_mode == "equal_width":
        vmin, vmax = values.min(), values.max()
        edges = np.linspace(vmin, vmax, nbins_dist + 1)
    elif dist_bin_mode == "fixed":
        edges = np.array(dist_bin_edges)
    elif dist_bin_mode == "percentile_intervals":
        if not hasattr(dist_bin_percentile_intervals, '__iter__'):
            raise ValueError("dist_bin_percentile_intervals must be a list of (low, high) tuples")
        bins = []
        labels = []
        for (lo_pct, hi_pct) in dist_bin_percentile_intervals:
            lo_val = np.percentile(values, lo_pct)
            hi_val = np.percentile(values, hi_pct)
            mask = (values >= lo_val) & (values <= hi_val)
            subset = cat_z_mag.loc[mask].copy()
            bins.append(subset)
            labels.append(f"$r_{{fil}} \\in [{lo_val:.1f}-{hi_val:.1f}]$ Mpc/h")
        return bins, labels, None
    else:
        raise ValueError(f"Unknown dist_bin_mode: {dist_bin_mode}")

    bins = []
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        bins.append(cat_z_mag.loc[mask].copy())
        labels.append(f"${lo:.1f} < r_{{fil}} \\leq {hi:.1f}$")
    return bins, labels, edges

# ---------------------------
# 2D CORRELATION FUNCTION ξ(σ, π)
# ---------------------------

# ---------------------------
# Updated xi_sigmapi_package
# ---------------------------
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import SymLogNorm

def xi_sigmapi_package(ra_data, dec_data, chi_data,
                       ra_rand, dec_rand, chi_rand,
                       pi_rebin, title=None,
                       data_weights=None, rand_weights=None,
                       output_folder=None, plotname="xi_sigma_pi.png",
                       min_sep=0.0, max_sep=50.0, bin_size=2.0,
                       paircounts_file=None, force_recompute=False):
    """
    Compute ξ(σ, π) using Corrfunc, save a plot, and return xi array + sigma bin edges.
    If paircounts_file is provided and exists, load precomputed pair counts instead of recomputing.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import multiprocessing
    from matplotlib.colors import TwoSlopeNorm
    try:
        from Corrfunc.theory.DDrppi import DDrppi
    except ImportError:
        raise ImportError("Corrfunc is not installed. Please run: conda install -c conda-forge corrfunc")

    # ------------------------------------------------------------
    # Try to load precomputed pair counts
    # ------------------------------------------------------------
    if paircounts_file and os.path.exists(paircounts_file) and not force_recompute:
        print(f"Loading precomputed pair counts from {paircounts_file}")
        data = np.load(paircounts_file)
        rp_bins = data['rp_bins']
        pi_rebin = data['pi_rebin']
        max_pimax = data['max_pimax']
        H_dd_rebinned = data['H_dd_rebinned']
        H_dr_rebinned = data['H_dr_rebinned']
        H_rr_rebinned = data['H_rr_rebinned']
        WD = data['WD']
        WR = data['WR']
        WD2 = data['WD2']
        WR2 = data['WR2']
        # Compute norms
        norm_DD = WD*WD - WD2
        norm_RR = WR*WR - WR2
        norm_DR = WD*WR
        DD = H_dd_rebinned / norm_DD
        RR = H_rr_rebinned / norm_RR
        DR = H_dr_rebinned / norm_DR
        with np.errstate(divide='ignore', invalid='ignore'):
            xi = (DD - 2*DR + RR) / RR
            xi[RR == 0] = 0
        # Proceed to plotting (skip recomputation)
    else:
        # ------------------------------------------------------------
        # Compute from scratch
        # ------------------------------------------------------------
        def sph2cart(ra, dec, chi):
            x = chi * np.cos(np.radians(dec)) * np.cos(np.radians(ra))
            y = chi * np.cos(np.radians(dec)) * np.sin(np.radians(ra))
            z = chi * np.sin(np.radians(dec))
            return x, y, z

        x_data, y_data, z_data = sph2cart(ra_data, dec_data, chi_data)
        x_rand, y_rand, z_rand = sph2cart(ra_rand, dec_rand, chi_rand)

        if data_weights is None:
            data_weights = np.ones(len(x_data))
        if rand_weights is None:
            rand_weights = np.ones(len(x_rand))

        nbins = int((max_sep - min_sep) / bin_size)
        rp_bins = np.linspace(min_sep, max_sep, nbins + 1)
        pimax = int(max_sep)
        nthreads = multiprocessing.cpu_count()

        dd_counts = DDrppi(1, nthreads, pimax, rp_bins,
                           x_data, y_data, z_data,
                           weights1=data_weights,
                           weight_type='pair_product',
                           periodic=False, verbose=False)
        H_dd = (dd_counts['npairs'] * dd_counts['weightavg']).reshape(nbins, pimax)

        rr_counts = DDrppi(1, nthreads, pimax, rp_bins,
                           x_rand, y_rand, z_rand,
                           weights1=rand_weights,
                           weight_type='pair_product',
                           periodic=False, verbose=False)
        H_rr = (rr_counts['npairs'] * rr_counts['weightavg']).reshape(nbins, pimax)

        dr_counts = DDrppi(0, nthreads, pimax, rp_bins,
                           x_data, y_data, z_data,
                           X2=x_rand, Y2=y_rand, Z2=z_rand,
                           weights1=data_weights,
                           weights2=rand_weights,
                           weight_type='pair_product',
                           periodic=False, verbose=False)
        H_dr = (dr_counts['npairs'] * dr_counts['weightavg']).reshape(nbins, pimax)

        max_pimax = (pimax // pi_rebin) * pi_rebin
        H_dd_rebinned = H_dd[:, :max_pimax].reshape(nbins, max_pimax // pi_rebin, pi_rebin).sum(axis=2)
        H_dr_rebinned = H_dr[:, :max_pimax].reshape(nbins, max_pimax // pi_rebin, pi_rebin).sum(axis=2)
        H_rr_rebinned = H_rr[:, :max_pimax].reshape(nbins, max_pimax // pi_rebin, pi_rebin).sum(axis=2)

        print("DD sum:", np.sum(H_dd_rebinned))
        print("DR sum:", np.sum(H_dr_rebinned))
        print("RR sum:", np.sum(H_rr_rebinned))

        WD = np.sum(data_weights)
        WR = np.sum(rand_weights)
        WD2 = np.sum(data_weights**2)
        WR2 = np.sum(rand_weights**2)

        norm_DD = WD*WD - WD2
        norm_RR = WR*WR - WR2
        norm_DR = WD*WR

        DD = H_dd_rebinned / norm_DD
        RR = H_rr_rebinned / norm_RR
        DR = H_dr_rebinned / norm_DR

        with np.errstate(divide='ignore', invalid='ignore'):
            xi = (DD - 2*DR + RR) / RR
            xi[RR == 0] = 0

        # Save pair counts if requested
        if paircounts_file:
            print(f"Saving pair counts to {paircounts_file}")
            os.makedirs(os.path.dirname(paircounts_file), exist_ok=True)
            np.savez(paircounts_file,
                     rp_bins=rp_bins,
                     pi_rebin=pi_rebin,
                     max_pimax=max_pimax,
                     H_dd_rebinned=H_dd_rebinned,
                     H_dr_rebinned=H_dr_rebinned,
                     H_rr_rebinned=H_rr_rebinned,
                     WD=WD, WR=WR, WD2=WD2, WR2=WR2)

    # ------------------------------------------------------------
    # Plotting (common for both loaded and computed)
    # ------------------------------------------------------------
    pi_edges_rebinned = np.arange(0, max_pimax + pi_rebin, pi_rebin)
    sigma_edges = rp_bins
    pi_edges = pi_edges_rebinned
    X, Y = np.meshgrid(sigma_edges, pi_edges)
    C = xi.T

    # vmin = np.percentile(xi, 1)
    # vmax = np.percentile(xi, 99)
    # if vmin >= 0:
    #     vmin = -vmax / 2
    # if vmax <= 0:
    #     vmax = -vmin / 2
    # norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)


    # Choose a linear threshold around zero (e.g., 0.01)
    linthresh = 0.001
    vmin = np.percentile(xi, 1)
    vmax = np.percentile(xi, 99)
    # Ensure vmin < 0 and vmax > 0 (or adjust)
    if vmin >= 0:
        vmin = -vmax / 2
    if vmax <= 0:
        vmax = -vmin / 2

    norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=vmin, vmax=vmax)
    # A diverging colormap works well with SymLogNorm
    #cmap = 'RdBu_r'

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(X, Y, C, shading='flat', cmap='plasma', norm=norm)
    ax.set_xlabel(r'$\sigma$ [$h^{-1}$ Mpc]')
    ax.set_ylabel(r'$\pi$ [$h^{-1}$ Mpc]')
    ax.set_title(title if title else r'$\xi(\sigma, \pi)$')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$\xi(\sigma,\pi)$')

    # Contours
    sigma_centers = 0.5 * (sigma_edges[:-1] + sigma_edges[1:])
    pi_centers = 0.5 * (pi_edges[:-1] + pi_edges[1:])
    Xc, Yc = np.meshgrid(sigma_centers, pi_centers)   # shape (n_pi, n_sigma)

    # Mask for BAO region (both σ and π between 50 and 150)
    mask_region = (Xc >= 50) & (Xc <= 150) & (Yc >= 50) & (Yc <= 150)
    values_region = C[mask_region]

    # Compute percentile levels
    levels = np.percentile(values_region, [50, 70, 90])

    # Plot contours
    ax.contour(Xc, Yc, C, levels=levels, colors='k', linewidths=1.5)

    ax.set_xlim(left=min_sep)
    ax.set_ylim(bottom=min_sep)

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        outpath = os.path.join(output_folder, plotname)
        fig.savefig(outpath, dpi=200, bbox_inches='tight')
        print(f"Saved ξ(σ, π) plot to {outpath}")
    else:
        plt.show()
    plt.close(fig)

    return xi, rp_bins

# ---------------------------
# Monopole functions
# ---------------------------
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def compute_monopole(xi, sigma_edges, pi_edges, s_bins=None):
    """
    Compute monopole ξ0(s) from ξ(σ,π) on a regular grid.

    Parameters
    ----------
    xi : 2D array, shape (n_sigma, n_pi)
        ξ(σ,π) values at the centers of the cells.
    sigma_edges : 1D array, length n_sigma+1
        Edges of the σ bins.
    pi_edges : 1D array, length n_pi+1
        Edges of the π bins.
    s_bins : 1D array, optional
        Edges of the output s bins. If None, uses σ edges up to π_max.

    Returns
    -------
    s_centers : 1D array
        Centers of the s bins.
    xi0 : 1D array
        Monopole ξ0(s) at each s center.
    """
    sigma_centers = 0.5 * (sigma_edges[:-1] + sigma_edges[1:])
    pi_centers = 0.5 * (pi_edges[:-1] + pi_edges[1:])

    # Interpolator for ξ(σ,π). Values outside the grid are set to 0.
    interp = RegularGridInterpolator(
        (sigma_centers, pi_centers), xi,
        bounds_error=False, fill_value=0.0
    )

    # Define output s bins
    if s_bins is None:
        max_s = pi_edges[-1]          # cannot integrate beyond the maximum π
        s_bins = sigma_edges[sigma_edges <= max_s]
        if s_bins[-1] < max_s:
            s_bins = np.append(s_bins, max_s)
    s_centers = 0.5 * (s_bins[:-1] + s_bins[1:])

    xi0 = np.zeros_like(s_centers)

    for i, s in enumerate(s_centers):
        π_max = min(s, pi_edges[-1])
        if π_max <= 0:
            xi0[i] = 0.0
            continue

        # Fine grid for the integration over π
        n_int = 100
        π_vals = np.linspace(0, π_max, n_int)
        σ_vals = np.sqrt(s**2 - π_vals**2)

        points = np.column_stack((σ_vals, π_vals))
        ξ_vals = interp(points)

        # Integral over π → monopole
        integral = np.trapz(ξ_vals, π_vals)
        xi0[i] = integral / s

    return s_centers, xi0

def plot_monopoles_combined(monopoles_list, labels, output_folder=None, filename='xi0_combined.png'):
    """
    Plot multiple monopoles ξ0(s) in a single figure.
    
    Parameters
    ----------
    monopoles_list : list of tuples
        Each tuple: (s_centers, xi0) for one sample/bin.
    labels : list of str
        Labels corresponding to each monopole.
    """
    import matplotlib.pyplot as plt
    import os

    fig, ax = plt.subplots(figsize=(8,6))
    
    colors = plt.cm.tab10.colors  # up to 10 colors
    for i, ((s, xi0), label) in enumerate(zip(monopoles_list, labels)):
        ax.plot(s, xi0*s**2, marker='o', linestyle='-', color=colors[i % len(colors)], label=label)
    
    #ax.axhline(0, color='k', linestyle='--')
    ax.set_xlabel(r'$s\,[h^{-1}\mathrm{Mpc}]$')
    ax.set_ylabel(r'$s^{2}\xi_0(s)$')
    ax.set_title('Monopoles ξ₀(s)')
    ax.axvline(102, color='k', linestyle=':', label='BAO scale')
    ax.legend(loc='lower left')
    plt.tight_layout()
    
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        outpath = os.path.join(output_folder, filename)
        fig.savefig(outpath, dpi=200, bbox_inches='tight')
        print(f"Saved combined monopole plot to {outpath}")
    else:
        plt.show()
    plt.close(fig)

# ---------------------------
# MAIN PROCEDURE
# ---------------------------

def main():
    # Print parameters
    print(f"""
    Running with parameters:
    - Sample: {sample}
    - Sigma: {sigma}
    - h: {h}
    - Redshift range: {zmin} to {zmax}
    - Magnitude cut: {mag_max}
    - gr cut: {f"{gr_min:.1f}" if 'gr_min' in locals() else 'none'}
    - RA/Dec method: {ran_radec_method}
    - Redshift generation: {ran_method}{f' (degree {deg})' if ran_method == 'poly' else ''}
    - nside: {nside}
    - nrand_mult: {nrand_mult}
    - dist_bin_mode: {dist_bin_mode}
    - dist_bin_intervals: {dist_bin_intervals if dist_bin_mode == 'custom_intervals' else 'N/A'}
    - dist_percentile_intervals: {dist_bin_percentile_intervals if dist_bin_mode == 'percentile_intervals' else 'N/A'}
    """)

    # Load catalog and apply cuts
    cat_full = load_catalog(sample)
    cat_z, cat_z_mag = select_sample(cat_full)

    if test_dilute != 1.0:
        print(f"Diluting sample by factor {test_dilute}...")
        cat_z_mag = cat_z_mag.sample(frac=test_dilute, random_state=42).copy()
        print(f"Sample diluted: {len(cat_z_mag)} galaxies")

    # Comoving distance
    cat_z_mag.loc[:, "r"] = cosmo.comoving_distance(cat_z_mag["red"].values).value * h

    # Plot K vs redshift
    plot_redshift_k(cat_z)

    # Preload RA/Dec if method='file'
    if ran_radec_method == 'file':
        print("Reading RA/Dec file...")
        radec_data = pd.read_csv(RADec_filepath)
        ra_random_file = radec_data["ra"].values
        dec_random_file = radec_data["dec"].values
        print(f"Loaded {len(ra_random_file)} random RA/Dec points")
    else:
        ra_random_file = dec_random_file = None

    # Split galaxies into dist_fil bins
    bins, labels, _ = split_by_dist_fil_bins(cat_z_mag)
    # Compute mean filament distance for each bin
    # median_distfil = [bin_df['dist_fil'].median() for bin_df in bins]

    # ------------------------------------------------------------
    # NEW: Compute required random counts and generate master RA/Dec
    # ------------------------------------------------------------
    nrand_full = nrand_mult * len(cat_z_mag)                     # for full sample
    nrand_bins = [nrand_mult * len(b) for b in bins]             # for each bin
    total_rand = nrand_full + sum(nrand_bins)                    # total needed

    if ran_radec_method == 'file':
        print(f"Main sample size: {len(cat_z_mag)}")
        print(f"Random sample size: {len(ra_random_file)}")
        print(f"Randoms to data ratio: {len(ra_random_file) / len(cat_z_mag):.2f}")
        print(f"Required randoms: {total_rand}")

    print(f"Generating master RA/Dec array of length {total_rand}...")
    master_ra, master_dec = generate_master_radec(
        full_cat=cat_z_mag,
        nrand_total=total_rand,
        nside=nside,
        ran_radec_method=ran_radec_method,
        ra_preload=ra_random_file if ran_radec_method == 'file' else None,
        dec_preload=dec_random_file if ran_radec_method == 'file' else None
    )
    print("Master array generated.")

    # --- Full sample random catalog (first slice) ---
    ra_rand_full = master_ra[:nrand_full]
    dec_rand_full = master_dec[:nrand_full]
    ptr = nrand_full   # pointer for next slice

    red_full = generate_random_red(cat_z_mag["red"].values, nrand_full, ran_method,
                                   deg if ran_method == "poly" else None)
    random_full = pd.DataFrame({"ra": ra_rand_full, "dec": dec_rand_full, "red": red_full})
    random_full["r"] = cosmo.comoving_distance(random_full["red"].values).value * h

    # Declination weights for full sample
    rand_weights_full = compute_dec_weights(cat_z_mag["dec"].values, random_full["dec"].values,
                                            nbins=40, method="kde", alpha=1)
    random_full["weight"] = rand_weights_full

    # Target redshift KDE from full sample (used for homogenisation)
    target_kde = gaussian_kde(cat_z_mag["red"].values)
    target_kde.set_bandwidth(target_kde.factor * 1.2)

    # --- Generate random catalogs for each bin (next slices) ---
    randoms_bins_list = []
    for i, bin_df in enumerate(bins):
        nrand_bin = nrand_bins[i]
        ra_bin = master_ra[ptr:ptr + nrand_bin]
        dec_bin = master_dec[ptr:ptr + nrand_bin]
        ptr += nrand_bin

        red_bin = generate_random_red(bin_df["red"].values, nrand_bin, ran_method,
                                      deg if ran_method == "poly" else None)
        rand_bin = pd.DataFrame({"ra": ra_bin, "dec": dec_bin, "red": red_bin})
        rand_bin["r"] = cosmo.comoving_distance(rand_bin["red"].values).value * h

        # Declination weights
        dec_weights = compute_dec_weights(bin_df["dec"].values, rand_bin["dec"].values,
                                          nbins=40, method="kde", alpha=1)

        # Redshift homogenisation (assigns weights to both galaxies and randoms)
        apply_redshift_weights_spline(bin_df, rand_bin, target_kde, dec_weights)

        randoms_bins_list.append(rand_bin)

    # Print length of data and random catalogs for verification
    print(f"\nFull sample: {len(cat_z_mag)} galaxies, {len(random_full)} randoms")
    for i, (bin_df, rand_bin) in enumerate(zip(bins, randoms_bins_list)):
        print(f"Bin {i}: {len(bin_df)} galaxies, {len(rand_bin)} randoms")
    
    # ------------------------------------------------------------
    # Diagnostic plots (unchanged)
    # ------------------------------------------------------------
    plot_radec_distribution(cat_z_mag, random_full)
    plot_bin_data_and_randoms(cat_z_mag, random_full, label="Full Sample",
                              plotname=f"../plots/{folderName}/bin_full_data_randoms.png")
    for i, (gxs, rxs, lab) in enumerate(zip(bins, randoms_bins_list, labels)):
        plot_bin_data_and_randoms(gxs, rxs, label=lab,
                                  plotname=f"../plots/{folderName}/bin_{i}_data_randoms.png")
        plot_radec_distribution(gxs, rxs, subsample=i)

    # ------------------------------------------------------------
    # Compute ξ(σ, π) for full sample and each bin
    # ------------------------------------------------------------
    print("\nComputing ξ(σ, π) for full sample...")
    # Full sample
    params = {
        'sample': sample,
        'sigma': sigma,
        'h': h,
        'zmin': zmin,
        'zmax': zmax,
        'mag_max': mag_max,
        'gr_min': gr_min,
        'min_sep_2d': min_sep_2d,
        'max_sep_2d': max_sep_2d,
        'bin_size_2d': bin_size_2d,
        'pi_rebin': pi_rebin,
        'nrand_mult': nrand_mult,
        'ran_radec_method': ran_radec_method,
        'dist_bin_mode': dist_bin_mode
    }
    paircounts_file_full = get_paircounts_filename("full", params)

    xi_sigma_pi_full, rp_bins = xi_sigmapi_package(
        cat_z_mag["ra"].values, cat_z_mag["dec"].values, cat_z_mag["r"].values,
        random_full["ra"].values, random_full["dec"].values, random_full["r"].values,
        pi_rebin, title=rf"$\xi(\sigma, \pi)$ Full Sample",
        data_weights=None,
        rand_weights=random_full["weight"].values,
        output_folder=output_folder,
        plotname="xi_sigma_pi_full.png",
        min_sep=min_sep_2d, max_sep=max_sep_2d, bin_size=bin_size_2d,
        paircounts_file=paircounts_file_full,
        force_recompute=False   # set to True to override
    )

    monopoles_list = []

    # Full sample
    # Build π edges from the rebinned bin size
    n_pi = xi_sigma_pi_full.shape[1]
    pi_edges = np.arange(n_pi + 1) * pi_rebin          # step = pi_rebin
    s_centers, xi0_full = compute_monopole(
        xi_sigma_pi_full, rp_bins, pi_edges
    )
    monopoles_list.append((s_centers, xi0_full))
    labels_list = ['Full Sample']

    # Each dist_fil bin
    for i, (gxs, rxs, lab) in enumerate(zip(bins, randoms_bins_list, labels)):
        params_bin = params.copy()
        paircounts_file_bin = get_paircounts_filename(f"bin{i}", params_bin)

        xi_sigma_pi_bin, rp_bins_bin = xi_sigmapi_package(
            gxs["ra"].values, gxs["dec"].values, gxs["r"].values,
            rxs["ra"].values, rxs["dec"].values, rxs["r"].values,
            pi_rebin, title=rf"$\xi(\sigma, \pi)$ {lab}",
            data_weights=gxs["weight"].values,
            rand_weights=rxs["weight"].values,
            output_folder=output_folder,
            plotname=f"xi_sigma_pi_bin{i}.png",
            min_sep=min_sep_2d, max_sep=max_sep_2d, bin_size=bin_size_2d,
            paircounts_file=paircounts_file_bin,
            force_recompute=False
        )
        # Build π edges from the rebinned bin size
        # Compute monopole for this bin
        n_pi_bin = xi_sigma_pi_bin.shape[1]
        pi_edges_bin = np.arange(n_pi_bin + 1) * pi_rebin
        s_centers_bin, xi0_bin = compute_monopole(
            xi_sigma_pi_bin, rp_bins_bin, pi_edges_bin
        )
        monopoles_list.append((s_centers_bin, xi0_bin))
        labels_list.append(f"{lab}") #(mean={median_distfil[i]:.1f})")

    # Plot all together
    plot_monopoles_combined(monopoles_list, labels_list,
                            output_folder=output_folder,
                            filename='xi0_combined.png')

    print("All done.")
if __name__ == "__main__":
    main()