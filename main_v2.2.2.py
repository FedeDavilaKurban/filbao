"""
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
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import healpy as hp
import treecorr
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from numpy.polynomial.polynomial import Polynomial
from matplotlib.colors import LogNorm
from astropy.cosmology import FlatLambdaCDM

# ---------------------------
# PARAMETERS 
# ---------------------------

# ---- Sample----------
sample = 'nyu'
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
    [(0, 5)],
    [(40, 80)],        
]

# Used if dist_bin_mode is "percentile" or "equal_width"
nbins_dist = 4   

# Only used if dist_bin_mode == "fixed"
dist_bin_edges = [0, 5, 30]  # example edges in h^-1 Mpc

# ------ Angular circular cut ------
use_angular_cut = False
ra_center = 185.0  # degrees
dec_center = 35.0  # degrees
theta_max = 38.0   # angular radius in degrees

# ------ Random catalog parameters ------
nside = 128  # Healpix nside
nrand_mult = 15  # Nr/Nd
common_RADec = True # Whether to use the same RA/Dec arrays for all bins (True) or generate separate RA/Dec for each bin (False)

# --- Method for generating RA/Dec ---
# Options:
#   'healpix'  : generate from Healpix mask of the data (original method)
#   'file'     : read RA/Dec from an external file (requires read_RADec=True and RADec_filepath)
#   'beta_mask': fit Beta distributions to data, then apply mask from external random file
ran_radec_method = 'beta_mask'   # <-- set to desired method

# Parameters for method='file' (kept for compatibility)
RADec_filepath = '../data/random_catalog_beta_N1500000_nside128.csv'

# Parameter for method='beta_mask'
radec_mask_file = '../data/lss_randoms_combined_cut.csv'  # external random catalog to define the mask for RA/Dec generation

# ------ Output folder --------
folderName = f'z{zmin:.2f}-{zmax:.2f}_mag{mag_max:.1f}_gr{gr_min:.1f}_nrand{nrand_mult}_RADECmethod{ran_radec_method}'
if use_angular_cut:
    folderName += f'_circle'

# Create output folder if it doesn't exist
output_folder = f"../plots/{folderName}/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ------ Correlation function parameters ------
minsep = 20.
maxsep = 150.0
bin_width = 3.5 #Mpc
nbins = int((maxsep - minsep) / bin_width) #
brute = False
npatch = 30

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
# HELPER FUNCTIONS
# ---------------------------

def safe_trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate y over x using np.trapz; fallback to np.trapezoid if necessary."""
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)


def build_cdf_from_line(
    data: np.ndarray, vmin: float, vmax: float, num_points: int = 10000
) -> Tuple[interp1d, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a two‑piece linear function to the histogram of `data` between vmin and vmax,
    return inverse CDF, z_vals, pdf_vals, cdf_vals.
    """
    hist, bin_edges = np.histogram(data, bins=50, range=(vmin, vmax), density=True)
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
    hist, bin_edges = np.histogram(data, bins=50, range=(vmin, vmax), density=True)
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


# ---------------------------
# FUNCTIONS FOR GENERATING RA/DEC
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


def fit_beta_distribution(data):
    """
    Fit a Beta distribution to the data (after normalising to [0,1]).
    Returns an inverse CDF function, the fitted parameters (alpha, beta), and the data range.
    """
    data_min, data_max = data.min(), data.max()
    data_norm = (data - data_min) / (data_max - data_min)
    eps = 1e-6
    data_norm = np.clip(data_norm, eps, 1 - eps)

    a, b, loc, scale = stats.beta.fit(data_norm, floc=0, fscale=1)

    def inv_cdf(u):
        samples_norm = stats.beta.ppf(u, a, b)
        return data_min + samples_norm * (data_max - data_min)

    return inv_cdf, (a, b), (data_min, data_max)


def load_radec_mask(mask_file: str, nside: int) -> np.ndarray:
    """
    Load an external random catalog (with 'ra','dec' columns) and build a Healpix mask.
    Returns a boolean array of length npix where True means the pixel is inside the footprint.
    """
    print(f"Loading RA/Dec mask from {mask_file}")
    rand_cat = pd.read_csv(mask_file)
    ra = rand_cat['ra'].values
    dec = rand_cat['dec'].values

    npix = hp.nside2npix(nside)
    mask = np.zeros(npix, dtype=bool)

    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    pixels = hp.ang2pix(nside, theta, phi)

    mask[pixels] = True
    return mask


def generate_random_radec_beta_mask(
    cat: pd.DataFrame,
    nrand: int,
    nside: int,
    mask_file: str,
    cached_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate RA/Dec for random catalog using Beta fits on `cat` RA and Dec,
    then apply Healpix mask from `mask_file`. If `cached_mask` is provided, reuse it.
    Returns arrays (ra, dec, mask) where mask is the updated cached mask (for potential reuse).
    """
    # Fit Beta distributions to the input catalog's RA and Dec
    ra_inv_cdf, ra_params, ra_range = fit_beta_distribution(cat["ra"].values)
    dec_inv_cdf, dec_params, dec_range = fit_beta_distribution(cat["dec"].values)

    # Load or use cached mask
    if cached_mask is None:
        mask = load_radec_mask(mask_file, nside)
    else:
        mask = cached_mask

    # Generate points until we have at least nrand valid ones
    ra_valid = np.array([], dtype=float)
    dec_valid = np.array([], dtype=float)

    while len(ra_valid) < nrand:
        batch_size = max(nrand * 2, 10000)
        u_ra = np.random.uniform(0, 1, batch_size)
        u_dec = np.random.uniform(0, 1, batch_size)

        ra_batch = ra_inv_cdf(u_ra)
        dec_batch = dec_inv_cdf(u_dec)

        theta = np.radians(90.0 - dec_batch)
        phi = np.radians(ra_batch)
        pix = hp.ang2pix(nside, theta, phi)

        valid = mask[pix]
        ra_valid = np.concatenate([ra_valid, ra_batch[valid]])
        dec_valid = np.concatenate([dec_valid, dec_batch[valid]])

        if len(ra_valid) > nrand * 10:
            print("Warning: Too many iterations in generate_random_radec_beta_mask. Check mask coverage.")
            break

    return ra_valid[:nrand], dec_valid[:nrand], mask


def generate_master_radec(
    full_cat: pd.DataFrame,
    nrand_total: int,
    nside: int,
    ran_radec_method: str,
    ra_preload: Optional[np.ndarray] = None,
    dec_preload: Optional[np.ndarray] = None,
    mask_file: Optional[str] = None,
    cached_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Generate a master RA/Dec array of length nrand_total using the specified method.
    Returns (ra_master, dec_master, updated_cached_mask).
    """
    if ran_radec_method == 'file':
        if ra_preload is None or dec_preload is None:
            raise ValueError("Method 'file' requires ra_preload and dec_preload.")
        if len(ra_preload) < nrand_total:
            raise ValueError(f"RA/Dec arrays contain {len(ra_preload)} points but {nrand_total} are needed.")
        ra_master = ra_preload[:nrand_total]
        dec_master = dec_preload[:nrand_total]
        new_mask = cached_mask

    elif ran_radec_method == 'healpix':
        print("Generating master RA/Dec from full sample's Healpix mask...")
        ra_master, dec_master = generate_random_radec_healpix(
            full_cat["ra"].values, full_cat["dec"].values, nside, nrand_total
        )
        new_mask = cached_mask

    elif ran_radec_method == 'beta_mask':
        if mask_file is None:
            raise ValueError("Method 'beta_mask' requires mask_file.")
        print("Generating master RA/Dec using Beta fits and external mask...")
        ra_master, dec_master, new_mask = generate_random_radec_beta_mask(
            full_cat, nrand_total, nside, mask_file, cached_mask
        )
    else:
        raise ValueError(f"Unknown ran_radec_method: {ran_radec_method}")

    return ra_master, dec_master, new_mask


# ---------------------------
# I/O & PLOTTING HELPERS
# ---------------------------

def ensure_dir_exists(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)


def save_figure(fig, path: str, dpi: int = 300):
    ensure_dir_exists(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# MAIN PROCEDURE
# ---------------------------

def load_catalog(sample_name: str) -> pd.DataFrame:
    """Load the requested SDSS-like catalog."""
    if sample_name == "nyu":
        datafile = "../data/sdss_dr72safe0_zmin_0.000_zmax_0.300_sigma_5.0.csv"
    elif sample_name == "sdss":
        datafile = "../data/sdss_zmin_0.000_zmax_0.300_sigma_5.0.csv"
    else:
        raise ValueError("Invalid sample")

    cat = pd.read_csv(datafile)
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


def apply_angular_cut(cat: pd.DataFrame,
                      ra_center: float,
                      dec_center: float,
                      theta_max: float) -> pd.DataFrame:
    """Keep only objects within theta_max (deg) from (ra_center, dec_center)."""
    ang = angular_distance(cat["ra"].values, cat["dec"].values,
                           ra_center, dec_center)
    mask = ang <= theta_max
    return cat.loc[mask].copy()


def plot_redshift_k(cat: pd.DataFrame) -> None:
    """Redshift‑magnitude 2D histogram."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist2d(cat["red"], cat["mag_abs_r"], bins=50, cmap="Blues", norm=LogNorm())
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
    axes[0].hist(randoms["ra"], bins=50, density=True, histtype="step", color="k", lw=1.5, label="Randoms")
    axes[0].hist(cat["ra"], bins=50, density=True, histtype="stepfilled", color="C00", alpha=0.8, label="Galaxies")
    axes[0].set_xlabel("RA")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[1].hist(randoms["dec"], bins=50, density=True, histtype="step", color="k", lw=1.5, label="Randoms")
    axes[1].hist(cat["dec"], bins=50, density=True, histtype="stepfilled", color="C00", alpha=0.8, label="Galaxies")
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


def calculate_xi(data: pd.DataFrame, randoms: pd.DataFrame, config: dict, sample_name: str = None):
    """
    Compute xi(s) using TreeCorr NNCorrelation.
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

    gcat = treecorr.Catalog(x=data["x"], y=data["y"], z=data["z"], npatch=npatch)
    rcat = treecorr.Catalog(x=randoms["x"], y=randoms["y"], z=randoms["z"],
                            patch_centers=gcat.patch_centers)

    dd.process(gcat)
    rr.process(rcat)
    dr.process(gcat, rcat)

    xi, varxi = dd.calculateXi(rr=rr, dr=dr)

    if sample_name is not None:
        rr.write("../data/rr.txt")
        dd.write("../data/dd.txt")
        dr.write("../data/dr.txt")
    else:
        rr.write(f"../data/rr_{sample_name}.txt")
        dd.write(f"../data/dd_{sample_name}.txt")
        dr.write(f"../data/dr_{sample_name}.txt")
    return xi, varxi, dd.meanr


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


def compute_xi_for_bins(bins, randoms_bins_list, config):
    xi_list, varxi_list, s_list = [], [], []
    for i, (gxs, rxs) in enumerate(zip(bins, randoms_bins_list)):
        print(f"Computing xi for dist_fil bin {i} (N={len(gxs)})")
        xi, varxi, s = calculate_xi(gxs, rxs, config, sample_name=f"bin{i}")
        xi_list.append(xi)
        varxi_list.append(varxi)
        s_list.append(s)
    return xi_list, varxi_list, s_list


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


def plot_bin_data_and_randoms(
    gxs: pd.DataFrame,
    rxs: pd.DataFrame,
    label: str,
    plotname: str,
):
    """Plot RA/Dec scatter and redshift histogram for one bin."""
    fig, axes = plt.subplots(3, 1, figsize=(7, 14))

    axes[0].scatter(rxs["ra"], rxs["dec"], s=1.5, color="k", alpha=0.5, label="Randoms")
    axes[0].scatter(gxs["ra"], gxs["dec"], s=1, color="C00", label="Galaxies")
    axes[0].set_xlabel("RA")
    axes[0].set_ylabel("DEC")
    axes[0].legend(loc='upper right')
    axes[0].set_title(label)

    axes[1].hist(gxs["red"], bins=50, density=True, histtype="stepfilled",
                 color="C00", alpha=0.8, label="Galaxies")
    axes[1].hist(rxs["red"], bins=50, density=True, histtype="step",
                 color="k", lw=1.5, label="Randoms")
    axes[1].set_xlabel("Redshift")
    axes[1].set_ylabel("PDF")
    axes[1].legend()

    axes[2].hist(gxs["dist_fil"], bins=50, density=True, color="C03", alpha=0.9)
    axes[2].set_xlabel(r"$r_{\rm fil}\,[h^{-1}\mathrm{Mpc}]$")
    axes[2].set_ylabel("PDF")

    plt.tight_layout()
    save_figure(fig, plotname, dpi=200)


def main():
    # Print parameters
    print(f"""
Running with parameters:
- Sample: {sample}
- Redshift range: {zmin} to {zmax}
- Magnitude cut: {mag_max}
- gr color cut: {gr_min}
- Random catalog method: {ran_radec_method}
- nside: {nside}
- nrand_mult: {nrand_mult}
- dist_bin_mode: {dist_bin_mode}
- dist_bin_intervals: {dist_bin_intervals if dist_bin_mode == "custom_intervals" else "N/A"}
- dist_bin_edges: {dist_bin_edges if dist_bin_mode == "fixed" else "N/A"}
- use_angular_cut: {use_angular_cut}
- Angular cut center: (RA={ra_center if use_angular_cut else "N/A"}, Dec={dec_center if use_angular_cut else "N/A"})
- Angular cut radius: {theta_max} degrees if use_angular_cut else "N/A"
          """)

    # Loading catalogue and applying cuts
    cat_full = load_catalog(sample)
    cat_z, cat_z_mag = select_sample(cat_full)

    # Precomputing comoving distances
    cat_z_mag.loc[:, "r"] = cosmo.comoving_distance(cat_z_mag["red"].values).value * h

    # Apply angular cut if needed
    if use_angular_cut:
        cat_z_mag = apply_angular_cut(cat_z_mag, ra_center, dec_center, theta_max)

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

    # --- Generate master RA/Dec if common_RADec is True ---
    if common_RADec:
        print("Generating master RA/Dec array for full sample...")
        master_ra, master_dec, cached_mask = generate_master_radec(
            full_cat=cat_z_mag,
            nrand_total=nrand_total,
            nside=nside,
            ran_radec_method=ran_radec_method,
            ra_preload=ra_random_file,
            dec_preload=dec_random_file,
            mask_file=radec_mask_file if ran_radec_method == 'beta_mask' else None,
            cached_mask=None
        )
        print(f"Master RA/Dec generated: {len(master_ra)} points")
    else:
        master_ra = master_dec = None
        cached_mask = None

    # --- Number of randoms for the full sample (used for total xi) ---
    nrand_full = nrand_mult * len(cat_z_mag)

    # --- Generate random catalog for full sample ---
    if common_RADec:
        # Full sample random catalog uses the master RA/Dec (first nrand_full points)
        ra_rand_full = master_ra[:nrand_full]
        dec_rand_full = master_dec[:nrand_full]
    else:
        # Generate separate RA/Dec for the full sample (old behaviour)
        print("Creating Random Catalogue for full sample (per‑bin generation not active)")
        master_ra_full, master_dec_full, _ = generate_master_radec(
            full_cat=cat_z_mag,
            nrand_total=nrand_total,
            nside=nside,
            ran_radec_method=ran_radec_method,
            ra_preload=ra_random_file,
            dec_preload=dec_random_file,
            mask_file=radec_mask_file if ran_radec_method == 'beta_mask' else None,
            cached_mask=None
        )
        ra_rand_full = master_ra_full[:nrand_full]
        dec_rand_full = master_dec_full[:nrand_full]
        # cached_mask is not needed further because per‑bin will generate fresh if not common
        cached_mask = None

    # Generate redshifts for full sample
    red_full = generate_random_red(cat_z_mag["red"].values, nrand_full, ran_method,
                                   deg if ran_method == "poly" else None)
    random_data = pd.DataFrame({"ra": ra_rand_full, "dec": dec_rand_full, "red": red_full})
    random_data["r"] = cosmo.comoving_distance(random_data["red"].values).value * h

    if use_angular_cut:
        print("Applying angular cut to full randoms...")
        random_data = apply_angular_cut(random_data, ra_center, dec_center, theta_max)

    print("Computing xi for full sample")
    xi_tot, varxi_tot, s_tot = calculate_xi(cat_z_mag, random_data, config, sample_name="total")

    print("Splitting galaxies by dist_fil bins")
    bins, labels, _ = split_by_dist_fil_bins(cat_z_mag)

    if use_angular_cut:
        bins = [apply_angular_cut(b, ra_center, dec_center, theta_max) for b in bins]

    # --- Generate random catalogs per bin ---
    randoms_bins_list = []
    start_idx = 0  # for slicing master arrays

    for i, bin_df in enumerate(bins):
        print(f'---- Generating random catalog for dist_fil bin {i} -----')
        nrand_bin = nrand_mult * len(bin_df)

        if common_RADec:
            # Slice master arrays
            end_idx = start_idx + nrand_bin
            if end_idx > len(master_ra):
                raise ValueError(f"Master RA/Dec exhausted: need {end_idx} but only {len(master_ra)} available.")
            ra_bin = master_ra[start_idx:end_idx]
            dec_bin = master_dec[start_idx:end_idx]
            start_idx = end_idx
        else:
            # Generate fresh RA/Dec for this bin using its own distribution
            # We discard the updated mask because we don't reuse it across bins
            ra_bin, dec_bin = generate_master_radec(
                full_cat=bin_df,
                nrand_total=nrand_bin,
                nside=nside,
                ran_radec_method=ran_radec_method,
                ra_preload=ra_full if common_RADec else None,
                dec_preload=dec_full if common_RADec else None,
                mask_file=radec_mask_file if ran_radec_method == 'beta_mask' else None,
                cached_mask=cached_mask
            )[:2]   # take only RA and Dec, ignore mask

        # Generate redshifts for this bin
        red_bin = generate_random_red(bin_df["red"].values, nrand_bin, ran_method,
                                      deg if ran_method == "poly" else None)

        rand_bin = pd.DataFrame({"ra": ra_bin, "dec": dec_bin, "red": red_bin})
        rand_bin["r"] = cosmo.comoving_distance(rand_bin["red"].values).value * h

        if use_angular_cut:
            rand_bin = apply_angular_cut(rand_bin, ra_center, dec_center, theta_max)

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
    )
    for i, (gxs, rxs, lab) in enumerate(zip(bins, randoms_bins_list, labels)):
        plot_bin_data_and_randoms(
            gxs,
            rxs,
            label=lab,
            plotname=f"../plots/{folderName}/bin_{i}_data_randoms.png",
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