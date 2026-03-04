"""
v3.1

- Cleanup with DeepSeek
- Removed Angular Cut option
- Removed common_RADec option

v3

- Redshift homogenisation: each bin's galaxies are weighted so that the weighted redshift distribution
  matches that of the full sample. The random catalog weights combine the declination correction with
  the same redshift factor to keep the Landy–Szalay estimator unbiased.
- Galaxy weights are passed to TreeCorr via the `data_weights` argument.
- Added `compute_z_weights` function (based on KDE) and integrated it into the main loop.
- Deleted deprecated 'beta_mask' method for RA/Dec generation

v2.2.2

- New method to generate random RA/Dec: fit Beta distributions to the data (RA and Dec independently),
  then generate points via inverse transform sampling and apply a Healpix mask derived from an external
  random catalog file (to match the survey footprint).
- The method is selected via the parameter `ran_radec_method` (options: 'healpix', 'file', 'beta_mask').
- Existing methods ('healpix' from data mask, direct file read) are kept for backward compatibility.
- Created output folder, instead of name modifiers

- When common_RADec = True, a single master RA/Dec array is generated from the full galaxy sample
  (using the chosen ran_radec_method). For each dist_fil bin, we take a contiguous slice of this
  master array of length nrand_mult * len(bin) – i.e., each bin shares the same underlying angular
  distribution as the full sample.
- Redshifts are always generated per bin using the bin's own redshift distribution.
- Removed unused functions (calculate_crossxi, old plot_xi, commented helper).

- Implemented weights for randoms based on declination distribution matching (to correct for any residual Dec-dependent selection effects).

v2.2.1

- Option to cut data and randoms around an angular circle.
  This is to avoid edge effects.
- Option to choose dist_fil bins by percentile, fixed edges, or equal width.

v2.2

- Compute random RA and Dec for each bin (instead of using a master RA and Dec)
- Implemented possibility of reading RA/Dec from file (if both common_RADec and read_RADec are True)
  or generating them on the fly (if common_RADec is True but read_RADec is False).

v2.1

- Computes correlations for bins in dist_fil (instead of just filament vs non-filament).
"""

import os
import shutil
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import healpy as hp
import treecorr
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

# ---- Sample----------
sample = 'nyu'
sigma = 5.0
h = 0.6774  # Hubble constant
zmin, zmax = 0.07, 0.2  # Redshift range
mag_max = -21.2
ran_method = 'random_choice'  # ['random_choice', 'piecewise', 'poly']
if ran_method == 'poly':
    deg = 5  # degree of polynomial for redshift distribution fit 
gr_min = 0.8

# ------ dist_fil binning ------
dist_bin_mode = "custom_intervals"
# Options:
#   "percentile"  → automatic equal-count bins
#   "fixed"       → user-defined bin edges
#   "equal_width" → uniform width bins between min/max
#   "custom_intervals" → user-defined arbitrary intervals (can be non‑contiguous and overlapping)

# Used only if dist_bin_mode == "custom_intervals"
dist_bin_intervals = [
    [(0, 3)],
    [(15, 40)],
]

# Used if dist_bin_mode is "percentile" or "equal_width"
nbins_dist = 3   

# Only used if dist_bin_mode == "fixed"
dist_bin_edges = [0, 5, 30]  # example edges in h^-1 Mpc

# ------ Random catalog parameters ------
nside = 256 # Healpix nside
nrand_mult = 500  # Nr/Nd

# --- Method for generating RA/Dec ---
# Options:
#   'healpix'  : generate from Healpix mask of the data (original method)
#   'file'     : read RA/Dec from an external file (requires read_RADec=True and RADec_filepath)
ran_radec_method = 'file'   # <-- set to desired method

# Parameters for method='file' (kept for compatibility)
if nrand_mult > 50:
    RADec_filepath = '../data/lss_randoms_combined_cut_LARGE.csv'
else:
    RADec_filepath = '../data/lss_randoms_combined_cut.csv'

# options:
# - '../data/random_catalog_beta_N1500000_nside128.csv'  (pre-generated RA/Dec using Beta+mask method, for consistency)
# - '../data/lss_randoms_combined_cut.csv' (downloaded from NYU website)

# ------ Output folder --------
folderName = f'z{zmin:.2f}-{zmax:.2f}_mag{mag_max:.1f}_gr{gr_min:.1f}_sigma{sigma}_nrand{nrand_mult}_RADECmethod{ran_radec_method}'

# Create output folder – if it exists, delete it and recreate a clean one
output_folder = f"../plots/{folderName}/"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# ------ Correlation function parameters ------
minsep = 30.
maxsep = 150.0
bin_width = 3.5 #Mpc
nbins = int((maxsep - minsep) / bin_width) #
brute = False
npatch = 50

config = {
    "min_sep": minsep,
    "max_sep": maxsep,
    "nbins": nbins,
    "bin_type": "Linear",
    "brute": brute,
    "var_method": "jackknife",
    "cross_patch_weight": "match",
    "npatch": npatch,
}

# ---------------------------
# COSMOLOGY
# ---------------------------
cosmo = FlatLambdaCDM(H0=h * 100, Om0=0.3089)

# ---------------------------
# GENERAL HELPER FUNCTIONS
# ---------------------------

def safe_trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate y over x using np.trapz; fallback to np.trapezoid if necessary."""
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)


def angular_distance(ra1, dec1, ra2, dec2):
    """Great‑circle angular distance (degrees). Inputs in degrees."""
    ra1 = np.deg2rad(ra1)
    dec1 = np.deg2rad(dec1)
    ra2 = np.deg2rad(ra2)
    dec2 = np.deg2rad(dec2)

    cosang = (np.sin(dec1) * np.sin(dec2) +
              np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosang))


def spherical_to_cartesian(ra, dec, r):
    """Convert spherical coordinates to Cartesian (x,y,z)."""
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


# ---------------------------
# REDSHIFT DISTRIBUTION FITTING (for random generation)
# ---------------------------

def build_cdf_from_line(
    data: np.ndarray, vmin: float, vmax: float, num_points: int = 10000
) -> Tuple[interp1d, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a two‑piece linear function to the histogram of `data` between vmin and vmax,
    return inverse CDF, z_vals, pdf_vals, cdf_vals.
    """
    hist, bin_edges = np.histogram(data, bins=40, range=(vmin, vmax), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def LinearFunction(x, a, b):
        return a * x + b

    def BreakFunction(x, a1, b1, a2, xb):
        yi = lambda x_: LinearFunction(x_, a1, b1)
        yo = lambda x_: LinearFunction(xb, a1, b1) + ((x_ - xb) * a2)
        return np.piecewise(x, [x < xb, x >= xb], [yi, yo])

    bounds = [[-np.inf, -np.inf, -np.inf, vmin], [np.inf, np.inf, np.inf, vmax]]
    popt, _ = curve_fit(BreakFunction, bin_centers, hist, bounds=bounds)
    print("build_cdf_from_line: fit params:", popt)

    z_vals = np.linspace(vmin, vmax, num_points)
    pdf_vals = BreakFunction(z_vals, *popt)
    pdf_vals = np.clip(pdf_vals, a_min=0.0, a_max=None)
    integ = safe_trapz(pdf_vals, z_vals)
    if integ <= 0:
        print("Warning: PDF integral is non-positive in build_cdf_from_line.")
    else:
        pdf_vals = pdf_vals / integ

    dz = z_vals[1] - z_vals[0]
    cdf_vals = np.cumsum(pdf_vals) * dz
    if cdf_vals[-1] <= 0:
        cdf_vals = np.clip(cdf_vals, 0.0, None)
    else:
        cdf_vals = cdf_vals / cdf_vals[-1]

    cdf_inv = interp1d(cdf_vals, z_vals, bounds_error=False, fill_value=(vmin, vmax))
    return cdf_inv, z_vals, pdf_vals, cdf_vals


def build_cdf_from_parabola(
    data: np.ndarray, vmin: float, vmax: float, deg: int, num_points: int = 10000
) -> Tuple[interp1d, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a polynomial to the histogram; return inverse CDF, z_vals, pdf_vals, cdf_vals.
    """
    hist, bin_edges = np.histogram(data, bins=40, range=(vmin, vmax), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    poly = Polynomial.fit(bin_centers, hist, deg=deg)
    z_vals = np.linspace(vmin, vmax, num_points)
    pdf_vals = poly(z_vals)

    pdf_vals = np.clip(pdf_vals, a_min=0.0, a_max=None)
    integ = safe_trapz(pdf_vals, z_vals)
    if integ <= 0:
        print("Warning: PDF integral is non-positive in build_cdf_from_parabola.")
    else:
        pdf_vals = pdf_vals / integ

    dz = z_vals[1] - z_vals[0]
    cdf_vals = np.cumsum(pdf_vals) * dz
    if cdf_vals[-1] <= 0:
        cdf_vals = np.clip(cdf_vals, 0.0, None)
    else:
        cdf_vals = cdf_vals / cdf_vals[-1]

    cdf_inv = interp1d(cdf_vals, z_vals, bounds_error=False, fill_value=(vmin, vmax))
    return cdf_inv, z_vals, pdf_vals, cdf_vals


def generate_random_red(redshift: np.ndarray, nrand: int, ran_method: str, deg: int) -> np.ndarray:
    """Generate random redshifts following the chosen method."""
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

def compute_dec_weights(
    data_dec,
    rand_dec,
    alpha=1.0,
    method="auto",
    kde_threshold=1_000_000,
    nbins=40,
    spline_s=0.5,
    bw_factor=1.2,
    n_grid=300
):
    """
    Compute smooth declination weights using either KDE or spline.

    Parameters
    ----------
    data_dec : array
        Declination of data catalog
    rand_dec : array
        Declination of random catalog
    alpha : float
        Strength of correction (1 = full correction)
    method : str
        "auto", "kde", or "spline"
    kde_threshold : int
        Random catalog size below which KDE is used in auto mode
    nbins : int
        Number of bins for spline method
    spline_s : float
        Smoothing parameter for spline
    bw_factor : float
        Bandwidth multiplier for KDE

    Returns
    -------
    weights : array
        Weights for random catalog
    """

    n_rand = len(rand_dec)

    # Decide method
    if method == "auto":
        method = "kde" if n_rand < kde_threshold else "spline"

    print(f"Using {method} method for declination weights")
    epsilon = 1e-10

    # ---------------------
    #  HIST METHOD
    # ---------------------
    if method=='hist':
        hist_data, edges = np.histogram(data_dec, bins=nbins)
        hist_rand, _ = np.histogram(rand_dec, bins=edges)

        epsilon = 1e-8
        ratio = (hist_data + epsilon) / (hist_rand + epsilon)

        bin_indices = np.digitize(rand_dec, edges) - 1
        bin_indices = np.clip(bin_indices, 0, nbins - 1)

        weights = ratio[bin_indices]

    # ---------------------
    # KDE METHOD
    # ---------------------
    elif method == "kde":

        # Build KDEs
        kde_data = gaussian_kde(data_dec)
        kde_data.set_bandwidth(kde_data.factor * bw_factor)

        kde_rand = gaussian_kde(rand_dec)
        kde_rand.set_bandwidth(kde_rand.factor * bw_factor)

        # Build evaluation grid
        dec_min = min(data_dec.min(), rand_dec.min())
        dec_max = max(data_dec.max(), rand_dec.max())

        grid = np.linspace(dec_min, dec_max, n_grid)

        # Evaluate only on grid
        density_data_grid = kde_data(grid)
        density_rand_grid = kde_rand(grid)

        ratio_grid = (density_data_grid + epsilon) / (density_rand_grid + epsilon)

        # Interpolate to random positions
        weights = np.interp(rand_dec, grid, ratio_grid)


    # ---------------------
    # SPLINE METHOD
    # ---------------------
    elif method == "spline":
        hist_data, edges = np.histogram(data_dec, bins=nbins)
        hist_rand, _ = np.histogram(rand_dec, bins=edges)

        centers = 0.5 * (edges[:-1] + edges[1:])
        ratio = (hist_data + epsilon) / (hist_rand + epsilon)

        spline = UnivariateSpline(centers, ratio, s=spline_s)
        weights = spline(rand_dec)

        # Prevent negative spline artifacts
        weights = np.clip(weights, 0.01, None)

    else:
        raise ValueError("method must be 'auto', 'kde', 'spline, or 'hist'")

    # ---------------------
    # Smooth toward unity
    # ---------------------
    weights = 1.0 + alpha * (weights - 1.0)

    # ---------------------
    # Normalise mean to 1
    # ---------------------
    weights /= np.mean(weights)

    return weights


def apply_redshift_weights_spline(bin_galaxies: pd.DataFrame,
                                   rand_catalog: pd.DataFrame,
                                   target_kde: gaussian_kde,
                                   dec_weights: np.ndarray) -> None:
    """
    Compute redshift weights using a spline ratio method and assign them to
    both the galaxy and random catalogs (in‑place).

    The steps are exactly those originally implemented in the main loop:
      1. Bin the bin's galaxy redshifts.
      2. Evaluate the target KDE at the bin centers to get target counts.
      3. Compute ratio = target / bin (with epsilon).
      4. Fit a UnivariateSpline to the ratio (s=0.5, ext='const').
      5. Galaxy weights = spline(bin_redshifts), clipped [0.1,10], normalized.
      6. Random weights (redshift part) = spline(rand_redshifts), clipped [0.1,10].
      7. Combined weight = redshift_part * dec_weights, normalized.

    Note: Weighting the randoms is necessary because the Landy–Szalay estimator
    requires that the weighted random catalog have the same redshift distribution
    as the weighted galaxy sample. If only galaxies were weighted, the DR and RR
    terms would be computed with mismatched distributions, biasing ξ(s).

    Parameters
    ----------
    bin_galaxies : pd.DataFrame
        Galaxy sample for this bin (must contain 'red' column). Will get a 'weight' column.
    rand_catalog : pd.DataFrame
        Random catalog for this bin (must contain 'red' column). Will get a 'weight' column.
    target_kde : scipy.stats.gaussian_kde
        KDE of the full sample redshift distribution.
    dec_weights : np.ndarray
        Pre‑computed declination weights for the random catalog (same length as rand_catalog).
    """
    # --- Redshift homogenisation weights (fast spline method) ---
    z_bin = bin_galaxies["red"].values
    n_bins_z = 40  # adjust as needed
    hist_bin, bin_edges = np.histogram(z_bin, bins=n_bins_z, density=False)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Evaluate target KDE at bin centers (convert to counts for same units)
    target_counts = target_kde(bin_centers) * len(z_bin) * (bin_edges[1] - bin_edges[0])

    # Ratio (target / bin) with small epsilon
    eps = 1e-10
    ratio = (target_counts + eps) / (hist_bin + eps)

    # Fit a spline (s=0.5 is a good starting point; adjust if noisy)
    spline_ratio = UnivariateSpline(bin_centers, ratio, s=0.5, ext='const')

    # Galaxy weights: evaluate spline at galaxy redshifts, clip, normalize
    gal_weight_raw = spline_ratio(z_bin)
    gal_weight_raw = np.clip(gal_weight_raw, 0.1, 10.0)
    gal_weight = gal_weight_raw / np.mean(gal_weight_raw)
    bin_galaxies.loc[:, "weight"] = gal_weight

    # Random weights: evaluate spline at random redshifts, combine with dec weights
    rand_z = rand_catalog["red"].values
    rand_z_weight_raw = spline_ratio(rand_z)
    rand_z_weight_raw = np.clip(rand_z_weight_raw, 0.1, 10.0)

    # Combined weight (redshift factor * declination weights)
    combined_raw = rand_z_weight_raw * dec_weights
    combined = combined_raw / np.mean(combined_raw)
    rand_catalog["weight"] = combined


# ---------------------------
# RA/DEC GENERATION FUNCTIONS
# ---------------------------

def generate_random_radec_healpix(
    ra: np.ndarray,
    dec: np.ndarray,
    nside: int,
    num_randoms: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate RA/Dec uniformly on the sphere but restricted to Healpix pixels
    that are occupied by the input (ra, dec).
    """
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


def generate_master_radec(
    full_cat: pd.DataFrame,
    nrand_total: int,
    nside: int,
    ran_radec_method: str,
    ra_preload: Optional[np.ndarray] = None,
    dec_preload: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a master RA/Dec array of length nrand_total using the specified method.
    Returns (ra_master, dec_master).
    """
    if ran_radec_method == 'file':
        if ra_preload is None or dec_preload is None:
            raise ValueError("Method 'file' requires ra_preload and dec_preload.")
        if len(ra_preload) < nrand_total:
            raise ValueError(f"RA/Dec arrays contain {len(ra_preload)} points but {nrand_total} are needed.")
        ra_master = ra_preload[:nrand_total]
        dec_master = dec_preload[:nrand_total]

    elif ran_radec_method == 'healpix':
        print("Generating master RA/Dec from full sample's Healpix mask...")
        ra_master, dec_master = generate_random_radec_healpix(
            full_cat["ra"].values, full_cat["dec"].values, nside, nrand_total
        )

    else:
        raise ValueError(f"Unknown ran_radec_method: {ran_radec_method}")

    return ra_master, dec_master


# ---------------------------
# PLOTTING FUNCTIONS
# ---------------------------

def plot_redshift_k(cat: pd.DataFrame) -> None:
    """Redshift‑magnitude 2D histogram."""
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
    print("Saving", filename)
    save_figure(fig, filename, dpi=100)


def plot_radec_distribution(cat: pd.DataFrame, randoms: pd.DataFrame, subsample: int = None) -> None:
    """Plot histograms of RA and Dec for data and randoms."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    axes[0].hist(randoms["ra"], bins=40, density=True, histtype="step", color="k", lw=1.5, label="Randoms")
    axes[0].hist(cat["ra"], bins=40, density=True, histtype="stepfilled", color="C00", alpha=0.8, label="Galaxies")
    axes[0].set_xlabel("RA")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[1].hist(cat["dec"], bins=40, density=True, histtype="stepfilled", color="C00", alpha=0.8, label="Galaxies")
    axes[1].hist(randoms["dec"], bins=40, density=True, histtype="step", color="k", lw=1.5, label="Randoms")
    axes[1].hist(randoms["dec"], bins=40, density=True,
            weights=randoms["weight"], linestyle='--', color='red', histtype='step', label="Weighted Randoms")
    axes[1].set_xlabel("Dec")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    if subsample is None:
        filename = f"../plots/{folderName}/radec_distribution.png"
        axtitle = 'Total sample'
    else:
        filename = f"../plots/{folderName}/radec_distribution_bin{subsample}.png"
        axtitle = f"Bin {subsample}"
    fig.suptitle(axtitle)
    plt.tight_layout()
    print("Saving", filename)
    save_figure(fig, filename, dpi=100)


def plot_bin_data_and_randoms(
    gxs: pd.DataFrame,
    rxs: pd.DataFrame,
    label: str,
    plotname: str,
    gal_weights: Optional[np.ndarray] = None
):
    """Plot RA/Dec scatter and redshift histogram for one bin."""
    fig, axes = plt.subplots(3, 1, figsize=(7, 14))

    axes[0].scatter(rxs["ra"], rxs["dec"], s=1.5, color="k", alpha=0.5, label="Randoms")
    axes[0].scatter(gxs["ra"], gxs["dec"], s=1, color="C00", label="Galaxies")
    axes[0].set_xlabel("RA")
    axes[0].set_ylabel("DEC")
    axes[0].legend(loc='upper right')
    axes[0].set_title(label)

    # Redshift histogram: galaxies (unweighted)
    axes[1].hist(gxs["red"], bins=40, density=True, histtype="stepfilled",
                 color="C00", alpha=0.8, label="Galaxies (unweighted)")
    # Redshift histogram: randoms (unweighted)
    axes[1].hist(rxs["red"], bins=40, density=True, histtype="step",
                 color="k", lw=1.5, label="Randoms")
    # If galaxy weights provided, plot weighted histogram (dashed)
    if gal_weights is not None:
        axes[1].hist(gxs["red"], bins=40, density=True, weights=gal_weights,
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


def plot_xi_dist_fil_bins(
    xi_tot, varxi_tot, s_tot,
    xi_list, varxi_list, s_list,
    labels,
    plotname=None
):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axvline(102, ls=":", c="k")

    # total sample
    sig_tot = np.sqrt(varxi_tot)
    y_tot = xi_tot * s_tot**2
    err_tot = sig_tot * s_tot**2
    ax.plot(s_tot, y_tot, lw=2.5, color="k", label="All galaxies")
    ax.fill_between(s_tot, y_tot - err_tot, y_tot + err_tot, alpha=0.25, color="k")

    colors = ["C00", "C01", "C02", "C03"]
    linestyles = ["-", "--", ":", "-."]

    for xi, varxi, s, lab, c, ls in zip(xi_list, varxi_list, s_list, labels, colors, linestyles):
        sig = np.sqrt(varxi)
        y = xi * s**2
        err = sig * s**2
        ax.plot(s, y, lw=2, color=c, ls=ls, label=lab)
        ax.fill_between(s, y - err, y + err, color=c, alpha=0.25)

    # optional Luis data
    try:
        data_luis = pd.read_csv('../data/test.out')
        s1 = data_luis['s']
        xi1 = data_luis['xi']
        mask = s1 >= minsep
        ax.plot(s1[mask], xi1[mask] * s1[mask]**2, color='k', lw=1, ls='--', label='Luis')
    except FileNotFoundError:
        pass

    ax.set_xlabel(r"$s \,[h^{-1}\mathrm{Mpc}]$")
    ax.set_ylabel(r"$s^2 \xi(s)$")
    ax.legend()
    plt.tight_layout()

    if plotname:
        print("Saving", plotname)
        save_figure(fig, plotname, dpi=300)


# ---------------------------
# CATALOG LOADING / MANIPULATION
# ---------------------------

def load_catalog(sample_name: str) -> pd.DataFrame:
    """Load the requested SDSS-like catalog."""
    if sample_name == "nyu":
        #datafile = "../data/sdss_dr72safe0_zmin_0.000_zmax_0.300_sigma_5.0.csv"
        if sigma==5.0:
            datafile = "../data/sdss_dr72safe0_sigma_5.0.csv"
        elif sigma==3.0:
            datafile = "../data/sdss_dr72safe0_sigma_3.0.csv"
    elif sample_name == "sdss":
        datafile = "../data/sdss_zmin_0.000_zmax_0.300_sigma_5.0.csv"
    else:
        raise ValueError("Invalid sample")

    cat = pd.read_csv(datafile)
    if sample_name != "nyu":
        cat["dist_fil"] /= h
    if gr_min != 0:
        cat = cat[cat["gr"] > gr_min]
    return cat


def select_sample(cat: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply redshift and magnitude cuts; return full z‑cut and final sample."""
    cat_z = cat[(cat["red"] >= zmin) & (cat["red"] <= zmax)]
    if sample == "nyu":
        cat_z_mag = cat_z[cat_z["mag_abs_r"] < mag_max].copy()
    else:
        cat_z_mag = cat_z[cat_z["mag_abs_r"] - 5 * np.log10(h) < mag_max].copy()
    return cat_z, cat_z_mag


def split_by_dist_fil_bins(cat_z_mag: pd.DataFrame):
    """
    Split galaxies into dist_fil bins using selected binning strategy.
    """
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
# CORRELATION FUNCTION
# ---------------------------

def calculate_xi(data: pd.DataFrame, randoms: pd.DataFrame, config: dict, data_weights=None, sample_name: str = None):
    """
    Compute xi(s) using TreeCorr NNCorrelation.

    Parameters
    ----------
    data : pd.DataFrame
        Galaxy catalog with columns 'ra','dec','red','r' (comoving distance)
    randoms : pd.DataFrame
        Random catalog with same columns plus 'weight' (combined random weights)
    config : dict
        TreeCorr configuration
    data_weights : array, optional
        Weights for galaxies (if None, unit weights are used)
    sample_name : str, optional
        Identifier for output files
    """
    data = data.copy()
    randoms = randoms.copy()
    data[["x", "y", "z"]] = np.column_stack(
        spherical_to_cartesian(data["ra"].values, data["dec"].values, data["r"].values)
    )
    randoms[["x", "y", "z"]] = np.column_stack(
        spherical_to_cartesian(randoms["ra"].values, randoms["dec"].values, randoms["r"].values)
    )

    dd = treecorr.NNCorrelation(config)
    dr = treecorr.NNCorrelation(config)
    rr = treecorr.NNCorrelation(config)

    if data_weights is None:
        data_weights = np.ones(len(data))

    gcat = treecorr.Catalog(
        x=data["x"], y=data["y"], z=data["z"],
        w=data_weights,
        npatch=npatch
    )

    rcat = treecorr.Catalog(
        x=randoms["x"], y=randoms["y"], z=randoms["z"],
        w=randoms["weight"].values,
        patch_centers=gcat.patch_centers
    )

    dd.process(gcat)
    rr.process(rcat)
    dr.process(gcat, rcat)

    xi, varxi = dd.calculateXi(rr=rr, dr=dr)

    # Optional: write files for debugging
    if sample_name is not None:
        rr.write(f"../data/rr_{sample_name}.txt")
        dd.write(f"../data/dd_{sample_name}.txt")
        dr.write(f"../data/dr_{sample_name}.txt")

    return xi, varxi, dd.meanr


def compute_xi_for_bins(bins, randoms_bins_list, config):
    xi_list, varxi_list, s_list = [], [], []
    for i, (gxs, rxs) in enumerate(zip(bins, randoms_bins_list)):
        print(f"Computing xi for dist_fil bin {i} (N={len(gxs)})")
        # Use galaxy weights if present
        data_w = gxs["weight"].values if "weight" in gxs.columns else None
        xi, varxi, s = calculate_xi(gxs, rxs, config, data_weights=data_w, sample_name=f"bin{i}")
        xi_list.append(xi)
        varxi_list.append(varxi)
        s_list.append(s)
    return xi_list, varxi_list, s_list


# ---------------------------
# MAIN PROCEDURE
# ---------------------------

def main():
    # Print parameters
    print(f"""
    Running with parameters:
    - Sample: {sample}
    - Sigma for filament detection: {sigma}
    - Hubble constant h: {h}
    - Redshift range: {zmin} to {zmax}
    - Magnitude cut: {mag_max}
    - gr color cut: {gr_min}
    - Random catalog method: {ran_radec_method}
    - Redshift generation method: {ran_method}{f' (degree {deg})' if ran_method == 'poly' else ''}
    - nside: {nside}
    - nrand_mult: {nrand_mult}
    - dist_bin_mode: {dist_bin_mode}
    - dist_bin_intervals: {dist_bin_intervals if dist_bin_mode == "custom_intervals" else "N/A"}
    - dist_bin_edges: {dist_bin_edges if dist_bin_mode == "fixed" else "N/A"}
    """)

    # Loading catalogue and applying cuts
    cat_full = load_catalog(sample)
    cat_z, cat_z_mag = select_sample(cat_full)

    # Precomputing comoving distances
    cat_z_mag.loc[:, "r"] = cosmo.comoving_distance(cat_z_mag["red"].values).value * h

    # Plot K vs Redshift
    plot_redshift_k(cat_full)

    # --- Preload RA/Dec if method='file' ---
    if ran_radec_method == 'file':
        print("Reading RA/Dec file once for all bins...")
        if os.path.exists(RADec_filepath):
            radec_data = pd.read_csv(RADec_filepath)
            ra_random_file = radec_data["ra"].values
            dec_random_file = radec_data["dec"].values
            print(f"RA/Dec loaded: {len(ra_random_file)} points")
        else:
            raise FileNotFoundError(f"RA/Dec file not found: {RADec_filepath}")
    else:
        ra_random_file = dec_random_file = None

    # --- Determine total number of random points needed for all bins ---
    nrand_total = nrand_mult * len(cat_z_mag)

    # --- Generate master RA/Dec for the full sample (shared by all bins) ---
    print("Generating master RA/Dec array for full sample...")
    master_ra, master_dec = generate_master_radec(
        full_cat=cat_z_mag,
        nrand_total=nrand_total,
        nside=nside,
        ran_radec_method=ran_radec_method,
        ra_preload=ra_random_file,
        dec_preload=dec_random_file
    )
    print(f"Master RA/Dec generated: {len(master_ra)} points")

    # --- Number of randoms for the full sample (used for total xi) ---
    nrand_full = nrand_mult * len(cat_z_mag)

    # --- Generate random catalog for full sample using the master RA/Dec ---
    ra_rand_full = master_ra[:nrand_full]
    dec_rand_full = master_dec[:nrand_full]

    # Generate redshifts for full sample
    red_full = generate_random_red(cat_z_mag["red"].values, nrand_full, ran_method,
                                   deg if ran_method == "poly" else None)
    random_data = pd.DataFrame({"ra": ra_rand_full, "dec": dec_rand_full, "red": red_full})
    random_data["r"] = cosmo.comoving_distance(random_data["red"].values).value * h

    # Compute Dec weights for full sample
    rand_weights = compute_dec_weights(
        cat_z_mag["dec"].values,
        random_data["dec"].values,
        nbins=40,
        method="kde",
        alpha=1
    )
    random_data["weight"] = rand_weights

    print("Computing xi for full sample")
    xi_tot, varxi_tot, s_tot = calculate_xi(cat_z_mag, random_data, config, sample_name="total")

    # Cutting off any galaxies with dist_fil > 100 to avoid outliers dominating the binning
    #cat_z_mag = cat_z_mag[cat_z_mag["dist_fil"] <= 70].copy()

    print("Splitting galaxies by dist_fil bins")
    bins, labels, _ = split_by_dist_fil_bins(cat_z_mag)

    # --- Define target redshift distribution from full sample ---
    target_kde = gaussian_kde(cat_z_mag["red"].values)
    target_kde.set_bandwidth(target_kde.factor * 1.2)   # adjust smoothing as needed

    # --- Generate random catalogs per bin with redshift homogenisation ---
    randoms_bins_list = []
    start_idx = 0  # for slicing master arrays

    for i, bin_df in enumerate(bins):
        print(f'---- Generating random catalog for dist_fil bin {i} -----')
        nrand_bin = nrand_mult * len(bin_df)

        # Slice master arrays
        end_idx = start_idx + nrand_bin
        if end_idx > len(master_ra):
            raise ValueError(f"Master RA/Dec exhausted: need {end_idx} but only {len(master_ra)} available.")
        ra_bin = master_ra[start_idx:end_idx]
        dec_bin = master_dec[start_idx:end_idx]
        start_idx = end_idx

        # Generate redshifts for this bin (from its own distribution)
        red_bin = generate_random_red(bin_df["red"].values, nrand_bin, ran_method,
                                      deg if ran_method == "poly" else None)

        rand_bin = pd.DataFrame({"ra": ra_bin, "dec": dec_bin, "red": red_bin})
        rand_bin["r"] = cosmo.comoving_distance(rand_bin["red"].values).value * h

        # ---- Declination weights (angular selection) ----
        dec_weights = compute_dec_weights(
            bin_df["dec"].values,
            rand_bin["dec"].values,
            nbins=40,
            method="kde",
            alpha=1
        )

        # --- Redshift homogenisation weights (using the exact spline method) ---
        apply_redshift_weights_spline(
            bin_galaxies=bin_df,
            rand_catalog=rand_bin,
            target_kde=target_kde,
            dec_weights=dec_weights
        )

        randoms_bins_list.append(rand_bin)

    print("Plotting RA/Dec distribution for full sample")
    plot_radec_distribution(cat_z_mag, random_data)

    print("Computing xi for each dist_fil bin")
    xi_list, varxi_list, s_list = compute_xi_for_bins(bins, randoms_bins_list, config)

    print("Plotting spatial + redshift distributions")
    plot_bin_data_and_randoms(
        cat_z_mag,
        random_data,
        label="Full Sample",
        plotname=f"../plots/{folderName}/bin_full_data_randoms.png",
        gal_weights=cat_z_mag["weight"].values if "weight" in cat_z_mag.columns else None

    )
    for i, (gxs, rxs, lab) in enumerate(zip(bins, randoms_bins_list, labels)):
        plot_bin_data_and_randoms(
            gxs,
            rxs,
            label=lab,
            plotname=f"../plots/{folderName}/bin_{i}_data_randoms.png",
            gal_weights=gxs["weight"].values if "weight" in gxs.columns else None
        )
        plot_radec_distribution(gxs, rxs, subsample=i)

    print("Plotting xi for total + dist_fil bins")
    plotname = f"../plots/{folderName}/xi_dist_fil_bins.png"
    plot_xi_dist_fil_bins(
        xi_tot, varxi_tot, s_tot,
        xi_list, varxi_list, s_list,
        labels,
        plotname=plotname
    )


if __name__ == "__main__":
    main()