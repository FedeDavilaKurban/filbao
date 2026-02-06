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
sample = 'nyu'
h = 0.6774  # Hubble constant
#zmin, zmax = 0.07, 0.12  # Redshift range
zmin, zmax = 0.07, 0.2  # Redshift range
ran_method = 'random_choice'  # ['random_choice', 'piecewise', 'poly']
if zmax == 0.12: 
    mag_max = -20.0  # Maximum magnitude
elif zmax == 0.2:
    mag_max = -21.2  # Maximum magnitude
gr_min = 0.8
deg = 4
#dist_min = 5.0
#dist_max = 10.0
nside = 256  # Healpix nside
nrand_mult = 3  # Nr/Nd

name_modifier = f'z{zmin:.2f}-{zmax:.2f}_mag{mag_max:.0f}_gr{gr_min:.1f}_nrand{nrand_mult}'

minsep = 10.
maxsep = 150.0
nbins = 30
brute = False
npatch = 10

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
        return np.trapz(y, x)
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
    data: np.ndarray, vmin: float, vmax: float, deg: int = 2, num_points: int = 10000
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


def generate_random_radec(ra: np.ndarray, dec: np.ndarray, nside: int, nrand: int) -> Tuple[np.ndarray, np.ndarray]:
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

    # number of uniform trial points (original code used int(10e6))
    num_randoms = nrand

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

    while ra_random.shape[0] < nrand:
        additional_needed = nrand - ra_random.shape[0]
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

    # if ra_random.shape[0] < nrand:
    #     raise ValueError("Not enough random points generated. Increase num_randoms or adjust nside.")

    return ra_random[:nrand], dec_random[:nrand]


def generate_random_red(redshift: np.ndarray, nrand: int, ran_method: str, deg: int = 5) -> np.ndarray:
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
        cat_z_mag = cat_z[cat_z["mag_abs_r"] < mag_max]
    else:
        cat_z_mag = cat_z[cat_z["mag_abs_r"] - 5 * np.log10(h) < mag_max]
    return cat_z, cat_z_mag


def compute_and_save_random_catalog(cat_full: pd.DataFrame, cat_z: pd.DataFrame) -> pd.DataFrame:
    print('Computing random catalog with method:', ran_method)
    """Create random RA/Dec and redshift matching the mask and redshift distribution."""
    ra = cat_full["ra"].values
    dec = cat_full["dec"].values
    redshift = cat_z["red"].values

    nrand = int(nrand_mult * len(ra))
    print(nrand)
    #ra_random, dec_random = generate_random_radec(cat_full["ra"].values, cat_full["dec"].values, nside, nrand)
    # Read Random RA/Dec from file (pre-generated for speed)
    random_radec_file = f"../data/random_sample_healpy_128.csv"
    random_radec = pd.read_csv(random_radec_file)
    ra_random = random_radec["ra"].values[:nrand]
    dec_random = random_radec["dec"].values[:nrand]
    red_random = generate_random_red(redshift, nrand, ran_method=ran_method, deg=deg)

    print("Random RA/Dec generated:", len(ra_random))
    print("Random Redshifts generated:", len(red_random))

    random_data = pd.DataFrame({"ra": ra_random, "dec": dec_random, "red": red_random})
    return random_data


def plot_redshift_k(cat_full: pd.DataFrame, cat_z_mag: pd.DataFrame, random_data: pd.DataFrame) -> None:
    """Recreate the redshift-magnitude 2D histogram (preserved plotting behavior)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist2d(cat_full["red"], cat_full["mag_abs_r"], bins=50, cmap="Blues", norm=LogNorm())
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


def split_filaments_and_randoms(cat_z_mag: pd.DataFrame, random_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split filament vs non-filament galaxies and make corresponding random catalogs."""
    p25, p50, p75 = np.percentile(cat_z_mag["dist_fil"], [25, 50, 75])
    p33, p66 = np.percentile(cat_z_mag["dist_fil"], [33, 66])

    filgxs = cat_z_mag[cat_z_mag["dist_fil"] <= p50]
    nonfilgxs = cat_z_mag[cat_z_mag["dist_fil"] > p50]

    nrand = len(random_data)
    random_filgxs_red = generate_random_red(filgxs["red"].values, nrand, ran_method=ran_method, deg=deg)
    random_filgxs = pd.DataFrame({"ra": random_data["ra"], "dec": random_data["dec"], "red": random_filgxs_red})

    random_nonfilgxs_red = generate_random_red(nonfilgxs["red"].values, nrand, ran_method=ran_method, deg=deg)
    random_nonfilgxs = pd.DataFrame({"ra": random_data["ra"], "dec": random_data["dec"], "red": random_nonfilgxs_red})

    return filgxs, nonfilgxs, random_filgxs, random_nonfilgxs


def plot_data_and_randoms(filgxs, nonfilgxs, random_filgxs, random_nonfilgxs) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 18), sharey=False)

    axes[0, 0].scatter(filgxs["ra"], filgxs["dec"], color="C00", s=1, label="Filament galaxies")
    axes[0, 1].scatter(nonfilgxs["ra"], nonfilgxs["dec"], color="C00", s=1, label="Non-filament galaxies")

    axes[1, 0].scatter(random_filgxs["ra"], random_filgxs["dec"], color="k", alpha=0.2, s=1.5, label="Random")
    axes[1, 1].scatter(random_nonfilgxs["ra"], random_nonfilgxs["dec"], color="k", alpha=0.2, s=1.5, label="Random")

    axes[1, 0].scatter(filgxs["ra"], filgxs["dec"], color="C00", s=1, label="Filament galaxies")
    axes[1, 1].scatter(nonfilgxs["ra"], nonfilgxs["dec"], color="C00", s=1, label="Non-filament galaxies")

    axes[2, 0].hist(filgxs["red"], bins=50, color="C00", density=True, histtype="stepfilled", label="Filament galaxies")
    axes[2, 0].hist(random_filgxs["red"], bins=50, color="C01", density=True, histtype="stepfilled", alpha=0.7, label="Random")

    axes[2, 1].hist(nonfilgxs["red"], bins=50, color="C00", density=True, histtype="stepfilled", label="Non-filament galaxies")
    axes[2, 1].hist(random_nonfilgxs["red"], bins=50, color="C01", density=True, histtype="stepfilled", alpha=0.7, label="Random")

    axes[1, 0].set_xlabel("RA", fontsize=12)
    axes[0, 0].set_ylabel("DEC", fontsize=12)
    axes[1, 0].set_ylabel("DEC", fontsize=12)
    axes[2, 0].set_xlabel("Redshift", fontsize=12)
    axes[2, 1].set_xlabel("Redshift", fontsize=12)

    axes[0, 0].legend(loc=1)
    axes[0, 1].legend(loc=1)
    axes[1, 0].legend(loc=1)
    axes[1, 1].legend(loc=1)
    axes[2, 0].legend(loc=2)
    axes[2, 1].legend(loc=2)

    filename = f"../plots/data_{name_modifier}.png"
    save_figure(fig, filename, dpi=300)


def plot_dist_fil(cat_z_mag: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(cat_z_mag["dist_fil"], bins=50, color="C03", density=True)
    ax.hist(cat_z_mag["dist_fil"], bins=50, color="C00", density=True)

    p25, p50, p75 = np.percentile(cat_z_mag["dist_fil"], [25, 50, 75])
    p33, p66 = np.percentile(cat_z_mag["dist_fil"], [33, 66])

    ax.axvline(x=p25, color="k", linestyle=":")
    ax.axvline(x=p50, color="k", linestyle=":")
    ax.axvline(x=p75, color="k", linestyle=":")

    ax.set_xlabel("Distance to filament (Mpc/h)")
    filename = f"../plots/dist_fil_hist_{name_modifier}.png"
    save_figure(fig, filename, dpi=100)
    print("percentiles p25,p50,p75:", p25, p50, p75)
    print("percentiles p33,p66:", p33, p66)


def H(ra, dec, z, h_val):
    """Return Cartesian coordinates (x,y,z) using astropy FlatLambdaCDM comoving_distance."""
    ra = np.array(ra, dtype=np.float32)
    dec = np.array(dec, dtype=np.float32)
    red = np.array(z, dtype=np.float32)

    r = np.float32(cosmo.comoving_distance(red).value) * h_val
    x = r * np.cos(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
    y = r * np.sin(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
    zc = r * np.sin(np.deg2rad(dec))
    return x, y, zc


def calculate_xi(data: pd.DataFrame, randoms: pd.DataFrame, config: dict, sample_name: str = None):
    """
    Compute xi(s) using TreeCorr NNCorrelation (preserved logic).
    Note: this preserves the original file-writing logic (including the possibly inverted sample check).
    """
    randoms["x"], randoms["y"], randoms["z"] = H(randoms["ra"], randoms["dec"], randoms["red"], h)
    data.loc[:, "x"], data.loc[:, "y"], data.loc[:, "z"] = H(data["ra"], data["dec"], data["red"], h)

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
    randoms1["x"], randoms1["y"], randoms1["z"] = H(randoms1["ra"], randoms1["dec"], randoms1["red"], h)
    data1.loc[:, "x"], data1.loc[:, "y"], data1.loc[:, "z"] = H(data1["ra"], data1["dec"], data1["red"], h)

    randoms2["x"], randoms2["y"], randoms2["z"] = H(randoms2["ra"], randoms2["dec"], randoms2["red"], h)
    data2.loc[:, "x"], data2.loc[:, "y"], data2.loc[:, "z"] = H(data2["ra"], data2["dec"], data2["red"], h)

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
    ax.errorbar(s_fil, xi_fil * s_fil ** 2, yerr=sig_fil * s_fil ** 2, color="C03", lw=2, label="Filament Galaxies", ls="--", capsize=capsize)

    if xi_nonfil is not None and varxi_nonfil is not None and s_nonfil is not None:
        sig_nonfil = np.sqrt(varxi_nonfil)
        ax.errorbar(s_nonfil, xi_nonfil * s_nonfil ** 2, yerr=sig_nonfil * s_nonfil ** 2, color="C02", lw=2, label="Non-Filament Galaxies", ls=":", capsize=capsize)
    
    if xi_cross is not None and varxi_cross is not None and s_cross is not None:
        sig_cross = np.sqrt(varxi_cross)
        ax.errorbar(s_cross, xi_cross * s_cross ** 2, yerr=sig_cross * s_cross ** 2, color="C01", lw=2, label="Cross Filament/Non-Filament", ls="-.", capsize=capsize)

    # optional reference data from file
    data_luis = pd.read_csv('../data/test.out')
    s1 = data_luis['s']
    xi1 = data_luis['xi']
    ax.plot(s1, xi1 * s1 ** 2, color='k', lw=1, ls='--', label='Luis')

    # optional check
    if filgxs is None or nonfilgxs is None or cat_z_mag is None:
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


def main():
    print("Loading catalog...")
    cat_full = load_catalog(sample)
    cat_z, cat_z_mag = select_sample(cat_full)

    plot_redshift_k(cat_full, cat_z_mag, None)

    print("Creating Random Catalogue...")
    random_data = compute_and_save_random_catalog(cat_full, cat_z_mag)
    print("Data size: ", len(cat_z_mag))
    print("Random size: ", len(random_data))

    filgxs, nonfilgxs, random_filgxs, random_nonfilgxs = split_filaments_and_randoms(cat_z_mag, random_data)
    plot_data_and_randoms(filgxs, nonfilgxs, random_filgxs, random_nonfilgxs)
    plot_dist_fil(cat_z_mag)

    print("Filament galaxies:", len(filgxs))
    print("Non-filament galaxies:", len(nonfilgxs))

    print("Calculating xi")
    xi, varxi, s = calculate_xi(cat_z_mag, random_data, config, sample_name=sample)

    print("Calculating xi_fil")
    xi_fil, varxi_fil, s_fil = calculate_xi(filgxs, random_filgxs, config, sample_name=sample)

    print("Calculating xi_nonfil")
    xi_nonfil, varxi_nonfil, s_nonfil = calculate_xi(nonfilgxs, random_nonfilgxs, config, sample_name=sample)

    print("Calculating cross_xi")
    random_all = random_data.copy()     # One master random catalog
    random_filgxs = random_all    # Use the SAME randoms for both
    random_nonfilgxs = random_all
    xi_cross, varxi_cross, s_cross = calculate_crossxi(filgxs, nonfilgxs, random_filgxs, random_nonfilgxs, config, sample_name=sample)

    # Uncomment when skipping filament/non-filament calculation
    # xi_fil, varxi_fil, s_fil = xi, varxi, s
    # xi_nonfil, varxi_nonfil, s_nonfil = xi, varxi, s

    plotname = f"../plots/xi_{name_modifier}.png"
    plot_xi(xi, varxi, s, \
            xi_fil, varxi_fil, s_fil, \
            xi_nonfil, varxi_nonfil, s_nonfil, \
            xi_cross, varxi_cross, s_cross, \
            filgxs, nonfilgxs, cat_z_mag, \
            plotname)


if __name__ == "__main__":
    main()
