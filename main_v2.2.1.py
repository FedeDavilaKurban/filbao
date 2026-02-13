"""
v2.2.1

- Option to cut data and randoms around an angular circle.
This is to avoid edge effects.
- Option to choose dist_fil bins by percentile, fixed edges, or equal width.

v2.2

-Compute random RA and Dec for each bin (instead of using a master RA and Dec)
-Implemented possibility of reading RA/Dec from file (if common_RADec and read_RADec are both True) 
or generating them on the fly (if common_RADec is True but read_RADec is False).


v2.1

Computes correlations for bins in dist_fil (instead of just filament vs non-filament).
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import healpy as hp
import treecorr
import matplotlib.pyplot as plt

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
zmin, zmax = 0.1, 0.15  # Redshift range
mag_max = -21.
ran_method = 'random_choice'  # ['random_choice', 'piecewise', 'poly']
if ran_method == 'poly':
    deg = 5  # degree of polynomial for redshift distribution fit 
if zmax == 0.12: 
    mag_max = -20.0  # Maximum magnitude
elif zmax == 0.2:
    mag_max = -21.2  # Maximum magnitude
gr_min = 0.8

# ------ dist_fil binning ------
dist_bin_mode = "custom_intervals"
# Options:
#   "percentile"  → automatic equal-count bins
#   "fixed"       → user-defined bin edges
#   "equal_width" → uniform width bins between min/max
#   "custom_intervals" → user-defined arbitrary intervals (can be non-contiguous and overlapping)

# Used only if dist_bin_mode == "custom_intervals"
dist_bin_intervals = [
    [(0, 20)],        # Bin 0
    [(30, 60)],       # Bin 1
]

nbins_dist = 4   # used for percentile or equal_width modes

# Only used if dist_bin_mode == "fixed"
dist_bin_edges = [0, 5, 30]  # example edges in h^-1 Mpc

# ------ Angular circular cut ------
use_angular_cut = True
ra_center = 185.0  # degrees
dec_center = 35.0  # degrees
theta_max = 36.0   # angular radius in degrees

# ------ Random catalog parameters ------
nside = 256  # Healpix nside
nrand_mult = 30  # Nr/Nd
common_RADec = True  # Whether to use the same RA/Dec mask for all bins (True) or generate separate RA/Dec for each bin (False)
read_RADec = True # Whether to read RA/Dec from file (True) or generate randomly (False); only applies if common_RADec is True
RADec_filepath = '../data/lss_randoms_combined_cut.csv'  # Filepath for RA/Dec if read_RADec is True

# ------ Output naming modifier --------
name_modifier = f'z{zmin:.2f}-{zmax:.2f}_mag{mag_max:.0f}_gr{gr_min:.1f}_nrand{nrand_mult}'
if use_angular_cut:
    name_modifier += f'_circle'

# ------ Correlation function parameters ------
minsep = 10.
maxsep = 150.0
nbins = 30
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
    # np.trapz exists in NumPy; keep this wrapper to be defensive
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        # older numpy alternate name
        return np.trapezoid(y, x)


def build_cdf_from_line(
    data: np.ndarray, vmin: float, vmax: float, num_points: int = 10000
) -> Tuple[interp1d, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a two-piece linear function to the histogram of `data` between vmin and vmax,
    return inverse CDF (interp1d), z_vals, pdf_vals, cdf_vals.
    Logic preserved from original code.
    """
    # Histogram
    hist, bin_edges = np.histogram(data, bins=50, range=(vmin, vmax), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def LinearFunction(x, a, b):
        return a * x + b

    def BreakFunction(x, a1, b1, a2, xb):
        yi = lambda x_: LinearFunction(x_, a1, b1)
        yo = lambda x_: LinearFunction(xb, a1, b1) + ((x_ - xb) * a2)
        return np.piecewise(x, [x < xb, x >= xb], [yi, yo])

    bounds = [[-np.inf, -np.inf, -np.inf, vmin], [np.inf, np.inf, np.inf, vmax]]

    popt, pcov = curve_fit(BreakFunction, bin_centers, hist, bounds=bounds)
    assert len(popt) == 4
    print("build_cdf_from_line: fit params:", popt)

    z_vals = np.linspace(vmin, vmax, num_points)
    pdf_vals = BreakFunction(z_vals, *popt)

    # Ensure positive PDF and normalize
    pdf_vals = np.clip(pdf_vals, a_min=0.0, a_max=None)
    integ = safe_trapz(pdf_vals, z_vals)
    if integ <= 0:
        # avoid division by zero; keep pdf as zeros (still returnable)
        print("Warning: PDF integral is non-positive in build_cdf_from_line.")
    else:
        pdf_vals = pdf_vals / integ

    # CDF
    dz = z_vals[1] - z_vals[0]
    cdf_vals = np.cumsum(pdf_vals) * dz
    if cdf_vals[-1] <= 0:
        cdf_vals = np.clip(cdf_vals, 0.0, None)
    else:
        cdf_vals = cdf_vals / cdf_vals[-1]

    # inverse CDF
    cdf_inv = interp1d(cdf_vals, z_vals, bounds_error=False, fill_value=(vmin, vmax))
    return cdf_inv, z_vals, pdf_vals, cdf_vals


def build_cdf_from_parabola(
    data: np.ndarray, vmin: float, vmax: float, deg: int, num_points: int = 10000
) -> Tuple[interp1d, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a polynomial to the histogram (using numpy.polynomial.Polynomial.fit).
    Returns inverse CDF, z_vals, pdf_vals, cdf_vals.
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


def generate_random_radec(ra: np.ndarray, dec: np.ndarray, nside: int, num_randoms: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate nrand random RA/Dec pairs that fall on Healpix pixels marked by the input (ra,dec).
    Implementation preserved from original code.
    """
    npix = hp.nside2npix(nside)
    mask = np.zeros(npix, dtype=int)

    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    pixels = hp.ang2pix(nside, theta, phi)

    mask[pixels] = 1

    # Iterate until we have enough valid random points
    ra_random = np.random.uniform(0.0, 360.0, num_randoms)
    u = np.random.uniform(-1.0, 1.0, num_randoms)
    dec_random_rad = np.arcsin(u)
    dec_random = np.degrees(dec_random_rad)

    theta_random = np.radians(90.0 - dec_random)
    phi_random = np.radians(ra_random)
    random_pixels = hp.ang2pix(nside, theta_random, phi_random)

    valid_indices = mask[random_pixels] == 1
    ra_random = ra_random[valid_indices]
    dec_random = dec_random[valid_indices]

    while ra_random.shape[0] < num_randoms:
        additional_needed = num_randoms - ra_random.shape[0]
        ra_additional = np.random.uniform(0.0, 360.0, additional_needed * 2)
        u_additional = np.random.uniform(-1.0, 1.0, additional_needed * 2)
        dec_additional_rad = np.arcsin(u_additional)
        dec_additional = np.degrees(dec_additional_rad)

        theta_additional = np.radians(90.0 - dec_additional)
        phi_additional = np.radians(ra_additional)
        additional_pixels = hp.ang2pix(nside, theta_additional, phi_additional)

        valid_additional_indices = mask[additional_pixels] == 1
        ra_valid = ra_additional[valid_additional_indices]
        dec_valid = dec_additional[valid_additional_indices]

        ra_random = np.concatenate((ra_random, ra_valid))
        dec_random = np.concatenate((dec_random, dec_valid))

    if ra_random.shape[0] < num_randoms:
        raise ValueError("Not enough random points generated. Increase num_randoms or adjust nside.")

    return ra_random[:num_randoms], dec_random[:num_randoms]


def generate_random_red(redshift: np.ndarray, nrand: int, ran_method: str, deg: int) -> np.ndarray:
    """
    Generate random redshifts following the chosen method.
    Methods preserved: 'poly', 'piecewise', 'random_choice'.
    """
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
    """
    Great-circle angular distance (degrees)
    Inputs in degrees.
    """
    ra1 = np.deg2rad(ra1)
    dec1 = np.deg2rad(dec1)
    ra2 = np.deg2rad(ra2)
    dec2 = np.deg2rad(dec2)

    cosang = (
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )

    cosang = np.clip(cosang, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosang))

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
    """Load the requested SDSS-like catalog (logic preserved)."""
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


def select_sample(cat: pd.DataFrame) -> pd.DataFrame:
    """Apply redshift and magnitude cuts (preserves original behavior)."""
    cat_z = cat[(cat["red"] >= zmin) & (cat["red"] <= zmax)]
    if sample == "nyu":
        cat_z_mag = cat_z[cat_z["mag_abs_r"] < mag_max].copy()  # <-- add .copy()
    else:
        cat_z_mag = cat_z[cat_z["mag_abs_r"] - 5 * np.log10(h) < mag_max].copy()  # <-- add .copy()
    return cat_z, cat_z_mag

def apply_angular_cut(cat: pd.DataFrame,
                      ra_center: float,
                      dec_center: float,
                      theta_max: float) -> pd.DataFrame:
    """
    Keep only objects within theta_max (deg)
    from (ra_center, dec_center).
    """
    ang = angular_distance(
        cat["ra"].values,
        cat["dec"].values,
        ra_center,
        dec_center,
    )

    mask = ang <= theta_max
    return cat.loc[mask].copy()

def generate_random_catalog(cat: pd.DataFrame, nside: int, nrand_mult: int, 
                            ra_preload: np.ndarray = None, dec_preload: np.ndarray = None) -> pd.DataFrame:
    """
    Create random RA/Dec and redshift matching the mask and redshift distribution.
    If ra_preload and dec_preload are provided, use them directly.
    Otherwise, generate RA/Dec using the Healpix mask of cat.
    """
    redshift = cat["red"].values
    nrand = int(nrand_mult * len(redshift))

    if ra_preload is not None and dec_preload is not None:
        if len(ra_preload) < nrand:
            raise ValueError(f"RA/Dec arrays contain {len(ra_preload)} points but {nrand} are needed.")
        ra_random = ra_preload[:nrand]
        dec_random = dec_preload[:nrand]
    else:
        # Generate RA/Dec using the Healpix mask from this catalog
        print("Generating random RA/Dec from catalog mask...")
        ra_random, dec_random = generate_random_radec(cat["ra"].values, cat["dec"].values, nside, nrand)

    red_random = generate_random_red(redshift, nrand, ran_method=ran_method, deg=deg if ran_method == "poly" else None)

    return pd.DataFrame({"ra": ra_random, "dec": dec_random, "red": red_random})



def plot_redshift_k(cat: pd.DataFrame) -> None:
    """Recreate the redshift-magnitude 2D histogram (preserved plotting behavior)."""
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
    filename = f"../plots/redshift_magnitude_{name_modifier}.png"
    print("Saving", filename)
    save_figure(fig, filename, dpi=100)


# def H(ra, dec, z, h_val):
#     """Return Cartesian coordinates (x,y,z) using astropy FlatLambdaCDM comoving_distance."""
#     ra = np.array(ra, dtype=np.float32)
#     dec = np.array(dec, dtype=np.float32)
#     red = np.array(z, dtype=np.float32)

#     r = np.float32(cosmo.comoving_distance(red).value) * h_val
#     x = r * np.cos(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
#     y = r * np.sin(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
#     zc = r * np.sin(np.deg2rad(dec))
#     return x, y, zc

def spherical_to_cartesian(ra, dec, r):
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    cos_dec = np.cos(dec_rad)

    x = r * cos_dec * np.cos(ra_rad)
    y = r * cos_dec * np.sin(ra_rad)
    z = r * np.sin(dec_rad)

    return x, y, z


def calculate_xi(data: pd.DataFrame, randoms: pd.DataFrame, config: dict, sample_name: str = None):
    """
    Compute xi(s) using TreeCorr NNCorrelation (preserved logic).
    Note: this preserves the original file-writing logic (including the possibly inverted sample check).
    """
    data.loc[:, ["x", "y", "z"]] = np.column_stack(
    spherical_to_cartesian(data["ra"].values, data["dec"].values, data["r"].values)
    )
    randoms.loc[:, ["x", "y", "z"]] = np.column_stack(
        spherical_to_cartesian(randoms["ra"].values, randoms["dec"].values, randoms["r"].values)
    )

    dd = treecorr.NNCorrelation(config)
    dr = treecorr.NNCorrelation(config)
    rr = treecorr.NNCorrelation(config)

    gcat = treecorr.Catalog(x=data["x"], y=data["y"], z=data["z"], npatch=npatch)
    rcat = treecorr.Catalog(x=randoms["x"], y=randoms["y"], z=randoms["z"], patch_centers=gcat.patch_centers)

    dd.process(gcat)
    rr.process(rcat)
    dr.process(gcat, rcat)

    xi, varxi = dd.calculateXi(rr=rr, dr=dr)

    # Preserve original file naming behavior (kept intentionally identical to input script)
    if sample_name is not None:
        rr.write("../data/rr.txt")
        dd.write("../data/dd.txt")
        dr.write("../data/dr.txt")
    else:
        rr.write(f"../data/rr_{sample_name}.txt")
        dd.write(f"../data/dd_{sample_name}.txt")
        dr.write(f"../data/dr_{sample_name}.txt")
    return xi, varxi, dd.meanr

def calculate_crossxi(data1: pd.DataFrame, data2: pd.DataFrame, randoms1: pd.DataFrame, randoms2: pd.DataFrame, config: dict, sample_name: str = None):
    """
    Compute cross xi(s) using TreeCorr NNCorrelation (preserved logic).
    Note: this preserves the original file-writing logic (including the possibly inverted sample check).
    """
    data1.loc[:, ["x","y","z"]] = np.column_stack(spherical_to_cartesian(data1["ra"].values, data1["dec"].values, data1["r"].values))
    randoms1.loc[:, ["x","y","z"]] = np.column_stack(spherical_to_cartesian(randoms1["ra"].values, randoms1["dec"].values, randoms1["r"].values))
    data2.loc[:, ["x","y","z"]] = np.column_stack(spherical_to_cartesian(data2["ra"].values, data2["dec"].values, data2["r"].values))
    randoms2.loc[:, ["x","y","z"]] = np.column_stack(spherical_to_cartesian(randoms2["ra"].values, randoms2["dec"].values, randoms2["r"].values))

    dd = treecorr.NNCorrelation(config)
    dr = treecorr.NNCorrelation(config)
    rd = treecorr.NNCorrelation(config)
    rr = treecorr.NNCorrelation(config)

    gcat1 = treecorr.Catalog(x=data1["x"], y=data1["y"], z=data1["z"], npatch=npatch)
    gcat2 = treecorr.Catalog(x=data2["x"], y=data2["y"], z=data2["z"], patch_centers=gcat1.patch_centers)
    rcat1 = treecorr.Catalog(x=randoms1["x"], y=randoms1["y"], z=randoms1["z"], patch_centers=gcat1.patch_centers)
    rcat2 = treecorr.Catalog(x=randoms2["x"], y=randoms2["y"], z=randoms2["z"], patch_centers=gcat1.patch_centers)

    dd.process(gcat1, gcat2)
    rr.process(rcat1, rcat2)
    dr.process(gcat1, rcat2)
    rd.process(rcat1, gcat2)

    xi, varxi = dd.calculateXi(rr=rr, dr=dr, rd=rd)

    # Preserve original file naming behavior (kept intentionally identical to input script)
    if sample_name is not None:
        rr.write("../data/rr_cross.txt")
        dd.write("../data/dd_cross.txt")
        dr.write("../data/dr_cross.txt")
        rd.write("../data/rd_cross.txt")
    else:
        rr.write(f"../data/rr_cross_{sample_name}.txt")
        dd.write(f"../data/dd_cross_{sample_name}.txt")
        dr.write(f"../data/dr_cross_{sample_name}.txt")
        rd.write(f"../data/rd_cross_{sample_name}.txt")
    return xi, varxi, dd.meanr


def plot_xi(xi, varxi, s, xi_fil, varxi_fil, s_fil, \
            xi_nonfil=None, varxi_nonfil=None, s_nonfil=None, \
            xi_cross=None, varxi_cross=None, s_cross=None, \
            filgxs=None, nonfilgxs=None, cat_z_mag=None,
            plotname=None) -> None:

    sig = np.sqrt(varxi)
    sig_fil = np.sqrt(varxi_fil)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axvline(102, ls=":", c="k")
    capsize = 3
    ax.errorbar(s, xi * s ** 2, yerr=sig * s ** 2, color="C00", lw=2, label="All Galaxies", capsize=capsize)
    ax.errorbar(s_fil, xi_fil * s_fil ** 2, yerr=sig_fil * s_fil ** 2, color="C03", lw=2, \
                label=f"Filament Galaxies ({filgxs['dist_fil'].min():.1f} < $r_{{\\mathrm{{fil}}}}$ [Mpc $h^{{-1}}$] < {filgxs['dist_fil'].max():.1f})", \
                    ls="--", capsize=capsize)

    if xi_nonfil is not None and varxi_nonfil is not None and s_nonfil is not None:
        sig_nonfil = np.sqrt(varxi_nonfil)
        ax.errorbar(s_nonfil, xi_nonfil * s_nonfil ** 2, yerr=sig_nonfil * s_nonfil ** 2, color="C02", lw=2, \
                    label=f"Non-filament Galaxies ({nonfilgxs['dist_fil'].min():.1f} < $r_{{\\mathrm{{fil}}}}$ [Mpc $h^{{-1}}$] < {nonfilgxs['dist_fil'].max():.1f})", \
                        ls=":", capsize=capsize)
    
    if xi_cross is not None and varxi_cross is not None and s_cross is not None:
        sig_cross = np.sqrt(varxi_cross)
        ax.errorbar(s_cross, xi_cross * s_cross ** 2, yerr=sig_cross * s_cross ** 2, color="C01", lw=2, label="Cross Filament/Non-Filament", ls="-.", capsize=capsize)

    # optional reference data from file
    data_luis = pd.read_csv('../data/test.out')
    s1 = data_luis['s']
    xi1 = data_luis['xi']
    ax.plot(s1, xi1 * s1 ** 2, color='k', lw=1, ls='--', label='Luis')

    # optional check
    if xi_cross is not None:
        fF = len(filgxs) / len(cat_z_mag)
        fN = len(nonfilgxs) / len(cat_z_mag)

        xi_recon = (
            fF**2 * xi_fil +
            fN**2 * xi_nonfil +
            2.0 * fF * fN * xi_cross
        )
        plt.plot(s, xi_recon * s**2, color='gray', lw=1, ls=':', label='Reconstructed xi')

    # Labels and legend
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$s²\xi(s)$")
    ax.legend()

    plt.tight_layout()
    if plotname:
        print("Saving", plotname)
        save_figure(fig, plotname, dpi=300)

def split_by_dist_fil_bins(cat_z_mag: pd.DataFrame):
    """
    Split galaxies into dist_fil bins using selected binning strategy.
    Supports:
        - percentile
        - equal_width
        - fixed
        - custom_intervals
    """

    values = cat_z_mag["dist_fil"].values

    # ---------------------------
    # CUSTOM INTERVAL MODE
    # ---------------------------
    if dist_bin_mode == "custom_intervals":

        bins = []
        labels = []

        for i, interval_list in enumerate(dist_bin_intervals):

            mask_total = np.zeros_like(values, dtype=bool)

            label_parts = []

            for (lo, hi) in interval_list:
                mask = (values >= lo) & (values <= hi)
                mask_total |= mask
                label_parts.append(f"{lo}-{hi}")

            subset = cat_z_mag.loc[mask_total].copy()
            bins.append(subset)

            label = " ∪ ".join(label_parts)
            labels.append(f"$r_{{fil}} \\in [{label}]$")

        print("Custom bins:")
        for i, b in enumerate(bins):
            print(f"Bin {i}: N = {len(b)}")

        return bins, labels, None

    # ---------------------------
    # Original modes below
    # ---------------------------
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
        lo = edges[i]
        hi = edges[i + 1]

        if i == len(edges) - 2:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)

        subset = cat_z_mag.loc[mask].copy()
        bins.append(subset)

        labels.append(f"${lo:.1f} < r_{{fil}} \\leq {hi:.1f}$")

    return bins, labels, edges

# def build_randoms_for_bins(bins, random_data):
#     """
#     Generate random catalogs for each dist_fil bin,
#     matching redshift distributions.
#     """
#     randoms_bins = []

#     for gxs in bins:
#         red_rand = generate_random_red(
#             gxs["red"].values,
#             len(random_data),
#             ran_method=ran_method,
#             deg=deg,
#         )
#         rand = pd.DataFrame({
#             "ra": random_data["ra"].values,
#             "dec": random_data["dec"].values,
#             "red": red_rand,
#         })
#         randoms_bins.append(rand)

#     return randoms_bins

def compute_xi_for_bins(bins, randoms_bins_list, config):
    xi_list = []
    varxi_list = []
    s_list = []

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

    # --- total sample ---
    sig_tot = np.sqrt(varxi_tot)
    y_tot = xi_tot * s_tot**2
    err_tot = sig_tot * s_tot**2

    ax.plot(
        s_tot,
        y_tot,
        lw=2.5,
        color="k",
        label="All galaxies",
    )

    ax.fill_between(
        s_tot,
        y_tot - err_tot,
        y_tot + err_tot,
        alpha=0.25,
        color="k"
    )

    colors = ["C00", "C01", "C02", "C03"]
    linestyles = ["-", "--", ":", "-."]

    for xi, varxi, s, lab, c, ls in zip(
        xi_list, varxi_list, s_list, labels, colors, linestyles
    ):
        sig = np.sqrt(varxi)
        y = xi * s**2
        err = sig * s**2

        ax.plot(
            s,
            y,
            lw=2,
            color=c,
            ls=ls,
            label=lab,
        )

        ax.fill_between(
            s,
            y - err,
            y + err,
            color=c,
            alpha=0.25,
        )
    # Include Luis data
    data_luis = pd.read_csv('../data/test.out')
    s1 = data_luis['s']
    xi1 = data_luis['xi']
    ax.plot(s1, xi1 * s1**2, color='k', lw=1, ls='--', label='Luis')

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
    """
    Plot RA/Dec and redshift distributions for one dist_fil bin,
    similar to plot_data_and_randoms but for a single population.
    """
    fig, axes = plt.subplots(3, 1, figsize=(7, 14))

    # --- RA/DEC ---
    axes[0].scatter(
        rxs["ra"], rxs["dec"],
        s=1.5, color="k", alpha=0.5, label="Randoms"
    )
    axes[0].scatter(
        gxs["ra"], gxs["dec"],
        s=1, color="C00", label="Galaxies"
    )
    axes[0].set_xlabel("RA")
    axes[0].set_ylabel("DEC")
    axes[0].legend(loc='upper right')
    axes[0].set_title(label)

    # --- Redshift histogram ---
    axes[1].hist(
        gxs["red"],
        bins=50,
        density=True,
        histtype="stepfilled",
        color="C00",
        alpha=0.8,
        label="Galaxies",
    )
    axes[1].hist(
        rxs["red"],
        bins=50,
        density=True,
        histtype="step",
        color="k",
        lw=1.5,
        label="Randoms",
    )
    axes[1].set_xlabel("Redshift")
    axes[1].set_ylabel("PDF")
    axes[1].legend()

    # --- dist_fil histogram (sanity check) ---
    axes[2].hist(
        gxs["dist_fil"],
        bins=50,
        density=True,
        color="C03",
        alpha=0.9,
    )
    axes[2].set_xlabel(r"$r_{\rm fil}\,[h^{-1}\mathrm{Mpc}]$")
    axes[2].set_ylabel("PDF")

    plt.tight_layout()
    save_figure(fig, plotname, dpi=200)


def main():
    print("Loading catalog...")
    cat_full = load_catalog(sample)
    cat_z, cat_z_mag = select_sample(cat_full)

    print("Precomputing comoving distances...")
    cat_z_mag.loc[:, "r"] = cosmo.comoving_distance(cat_z_mag["red"].values).value * h

    if use_angular_cut:
        print("Applying angular cut to data...")
        cat_z_mag = apply_angular_cut(
            cat_z_mag,
            ra_center,
            dec_center,
            theta_max,
        )

    plot_redshift_k(cat_full)

    print("Creating Random Catalogue for full sample")
    if common_RADec and read_RADec:
        print("Reading RA/Dec file once for all bins...")
        if os.path.exists(RADec_filepath):
            radec_data = pd.read_csv(RADec_filepath)
            ra_full = radec_data["ra"].values
            dec_full = radec_data["dec"].values
            print(f"RA/Dec loaded: {len(ra_full)} points")
        else:
            raise FileNotFoundError(f"RA/Dec file not found: {RADec_filepath}")
    else:
        ra_full = dec_full = None  # no preloaded RA/Dec

        random_data = generate_random_catalog(cat_z_mag, nside, nrand_mult,
                                            ra_full=ra_full, dec_full=dec_full)

    # Full sample
    random_data = generate_random_catalog(cat_z_mag, nside, nrand_mult,
                                        ra_preload=ra_full, dec_preload=dec_full)
    random_data["r"] = cosmo.comoving_distance(random_data["red"].values).value * h

    if use_angular_cut:
        print("Applying angular cut to full randoms...")
        random_data = apply_angular_cut(
            random_data,
            ra_center,
            dec_center,
            theta_max,
        )

    print("Computing xi for full sample")
    xi_tot, varxi_tot, s_tot = calculate_xi(
        cat_z_mag,
        random_data,
        config,
        sample_name="total"
    )

    print("Splitting galaxies by dist_fil bins")
    bins, labels, percentiles = split_by_dist_fil_bins(cat_z_mag)

    if use_angular_cut:
        bins = [
            apply_angular_cut(b, ra_center, dec_center, theta_max)
            for b in bins
        ]

    randoms_bins_list = []
    for i in range(len(bins)):
        print(f'---- Generating random catalog for dist_fil bin {i} -----')
        rand_bin = generate_random_catalog(
                                            bins[i],
                                            nside,
                                            nrand_mult,
                                            ra_preload=ra_full if common_RADec else None,
                                            dec_preload=dec_full if common_RADec else None
                                        )

        if use_angular_cut:
            rand_bin = apply_angular_cut(
                rand_bin,
                ra_center,
                dec_center,
                theta_max,
            )
            
        # PRECOMPUTE COMOVING DISTANCE HERE
        rand_bin.loc[:, "r"] = cosmo.comoving_distance(rand_bin["red"].values).value * h

        randoms_bins_list.append(rand_bin)


    print("Computing xi for each dist_fil bin")
    xi_list, varxi_list, s_list = compute_xi_for_bins(
        bins, randoms_bins_list, config
    )

    print("Plotting spatial + redshift distributions")
    # Plot full data first
    plot_bin_data_and_randoms(
        cat_z_mag,
        random_data,
        label="Full Sample",
        plotname=f"../plots/bin_full_data_randoms_{name_modifier}.png",
    )
    # Plot each bin separately
    for i, (gxs, rxs, lab) in enumerate(zip(bins, randoms_bins_list, labels)):
        #print(len(gxs), len(rxs))
        #print(rxs[:10])
        plot_bin_data_and_randoms(
            gxs,
            rxs,
            label=lab,
            plotname=f"../plots/bin_{i}_data_randoms_{name_modifier}.png",
        )

    print("Plotting xi for total + dist_fil bins")
    plotname = f"../plots/xi_dist_fil_bins_{name_modifier}.png"
    plot_xi_dist_fil_bins(
        xi_tot, varxi_tot, s_tot,
        xi_list, varxi_list, s_list,
        labels,
        plotname=plotname
    )


if __name__ == "__main__":
    main()
