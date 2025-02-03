import numpy as np
import matplotlib.pyplot as plt
from Corrfunc.mocks import DDrppi_mocks
from Corrfunc.utils import convert_3d_counts_to_cf
import pandas as pd 
import healpy as hp
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import Polynomial

# Step 1: Read data and randoms
def read_data(zmin=0.05,zmax=0.15):
    # Load your data and randoms as Pandas DataFrames
    sdss = pd.read_csv('../data/sdss_zmin_0.000_zmax_0.300_sigma_5.0.csv')
    sdss  = sdss[(sdss["red"] > zmin)&(sdss["red"] < zmax)]
    return sdss

def read_randoms(filename):
    randoms = pd.read_csv(filename) 
    return randoms


# Step 3: Create random catalog
def build_cdf_from_parabola(data, num_points=10000):
    # Create a histogram of the redshifts
    vmin, vmax = data.min(), data.max()
    hist, bin_edges = np.histogram(data, bins=50, range=(vmin, vmax), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Fit a parabola (second-degree polynomial) to the histogram
    poly = Polynomial.fit(bin_centers, hist, deg=2)
    z_vals = np.linspace(vmin, vmax, num_points)
    pdf_vals = poly(z_vals)

    # Ensure the PDF is positive and normalized
    pdf_vals = np.clip(pdf_vals, a_min=0, a_max=None)
    pdf_vals /= np.trapz(pdf_vals, z_vals)  # Normalize the PDF

    # Build the CDF
    cdf_vals = np.cumsum(pdf_vals) * (z_vals[1] - z_vals[0])
    cdf_vals /= cdf_vals[-1]  # Normalize to [0, 1]

    # Create inverse CDF via interpolation
    cdf_inv = interp1d(cdf_vals, z_vals, bounds_error=False, fill_value=(vmin, vmax))
    return cdf_inv, z_vals, pdf_vals, cdf_vals
  
def generate_random_catalog(nrand, data, nside=128, write=False):
    """Generate a random catalog with uniform RA, DEC, and redshift."""
    # Set the resolution (nside)
    #nside = 128  # Approx 55 arcmin resolution (adjust as needed)
    #nrand = 20 # Times the size of the data (around 10% is left after mask)

    ra = data['ra'].values
    dec = data['dec'].values
    z = data['red'].values

    # Total number of pixels in the map
    npix = hp.nside2npix(nside)

    # Initialize a mask array (1 = valid, 0 = invalid)
    mask = np.zeros(npix, dtype=int)

    # Convert RA, Dec to Healpix indices
    theta = np.radians(90 - dec)  # Declination to colatitude
    phi = np.radians(ra)          # Right Ascension to longitude
    pixels = hp.ang2pix(nside, theta, phi)

    # Mark these pixels as valid
    mask[pixels] = 1

    num_randoms = len(ra)*nrand  # Number of random points

    # Generate uniform random RA, Dec
    ra_random = np.random.uniform(0, 360, num_randoms)  # RA: 0 to 360 degrees
    dec_random = np.random.uniform(-90, 90, num_randoms)  # Dec: -90 to 90 degrees

    # Convert RA, Dec to Healpix indices
    theta_random = np.radians(90 - dec_random)
    phi_random = np.radians(ra_random)
    random_pixels = hp.ang2pix(nside, theta_random, phi_random)

    # Apply the mask
    valid_indices = mask[random_pixels] == 1
    ra_random = ra_random[valid_indices]
    dec_random = dec_random[valid_indices]

    # Read redshift distribution and apply to randoms
    # Build the inverse CDF from a smoothed KDE fit
    cdf_inv_z, z_vals, pdf_vals, cdf_vals = build_cdf_from_parabola(z)

    # Generate random redshifts
    u = np.random.uniform(0, 1, len(ra_random))
    red_random = cdf_inv_z(u)

    random_data = pd.DataFrame({
        'ra': ra_random,
        'dec': dec_random,
        'red': red_random
    })

    print(len(ra))
    print(len(ra_random))

    if write==True: random_data.to_csv(f'../data/random_sample_healpy_{nside}_{len(random_data)}.csv', index=False)

    return random_data

# Step 4: Run Corrfunc to compute xi(mu, s)
def calculate_xi_rp_pi(ra_data, dec_data, z_data, ra_rand, dec_rand, z_rand, bins, pimax, nthreads=1):
    """Calculate xi(mu, s) using Corrfunc."""

    # Pair counts with Corrfunc
    DD_counts = DDrppi_mocks(1, 2, binfile=bins, pimax=pimax, \
        RA1=ra_data, DEC1=dec_data, CZ1=3e5*z_data,\
            nthreads=nthreads)
    DR_counts = DDrppi_mocks(0, 2, binfile=bins, pimax=pimax, \
        RA1=ra_data, DEC1=dec_data, CZ1=3e5*z_data, RA2=ra_rand, DEC2=dec_rand, CZ2=3e5*z_rand,\
            nthreads=nthreads)
    RR_counts = DDrppi_mocks(1, 2, binfile=bins, pimax=pimax, \
        RA1=ra_rand, DEC1=dec_rand, CZ1=3e5*z_rand,\
            nthreads=nthreads)

    # Convert to correlation function
    xi = convert_3d_counts_to_cf(len(ra_data), len(ra_data), len(ra_rand), len(ra_rand), \
        DD_counts, DR_counts, DR_counts, RR_counts)
    return xi


# Step 5: Plot xi(mu, s)
def plot_xi_rp_pi_contours(xi, bins, pimax, filename):
    """Plot xi(mu, s) as a contour plot with s on the y-axis and mu on the inverted x-axis."""
    rp_bins = 0.5 * (bins[:-1] + bins[1:])  # Midpoints of s bins
    #rp_bins = bins
    pi_bins = np.arange(0, pimax)
    #pi_bins_centers = 0.5 * (pi_bins[:-1] + pi_bins[1:])  # Midpoints of mu bins
    pi_bins_centers = pi_bins
    print(np.shape(rp_bins))
    print(np.shape(pi_bins_centers))

    xi = xi.reshape(len(rp_bins), len(pi_bins))

    print(np.shape(xi))

    # Create the contour plot
    plt.figure(figsize=(8, 6))
    cs = plt.contourf(rp_bins, pi_bins_centers, xi, levels=20, cmap='viridis')  
    cbar = plt.colorbar(cs)
    cbar.set_label(r'$\xi(\mu, s)$')

    plt.xlabel(r'$r_p$')  # Negative mu since it's inverted
    plt.ylabel(r'$\pi$')
    plt.title(r'Two-Point Correlation Function $\xi(r_p, \pi)$')
    plt.savefig(filename)
    plt.close()

def plot_xil(filename):

    def get_xi0246(xi,mu,rs):
    
        import numpy as np
        

        nbins_s = len(rs)-1
        nbins_m = len(mu)

        xi = xi.reshape(nbins_s, nbins_m)

        xi_sm = xi

        dmu = 1.0/nbins_m
        
        #rs = corr.D1D2.coords['r']
        #mu = corr.D1D2.coords['mu']
        
        xi_s0 = np.zeros(nbins_s)
        xi_s2 = np.zeros(nbins_s)
        xi_s4 = np.zeros(nbins_s)
        xi_s6 = np.zeros(nbins_s)
        
        sr = np.zeros(nbins_s)
        rm = np.zeros(nbins_m)
        
        l0 = 0.0
        l1 = 1.0
        l2 = 2.0
        l3 = 3.0
        
        for i in range(nbins_s):
            
            sr[i] = rs[i]
            
            for j in range(nbins_m):
                rm[j]=mu[j]
                xi_s0[i]  += (4.0*l0+1.0)*xi_sm[i,j]*1.0*dmu 
                xi_s2[i]  += (4.0*l1+1.0)*xi_sm[i,j]*((3*rm[j]**2 - 1.0)/2.0)*dmu
                xi_s4[i]  += (4.0*l2+1.0)*xi_sm[i,j]*((35*rm[j]**4 - 30*rm[j]**2 + 3.0)/8.0)*dmu
                xi_s6[i]  += (4.0*l3+1.0)*xi_sm[i,j]*((231*rm[j]**6 - 315*rm[j]**4 + 105*rm[j]**2 - 5)/16.0)*dmu
        
        return xi_s0, xi_s2, xi_s4, xi_s6

    # Load the data from the .npz file
    file_path = filename+'.npz'
    data = np.load(file_path)

    s = data['s']  # Edges of s bins
    mu = data['mu']  # 
    xi = data['xi']  # Structured array for 2D correlation function

    xi_l = get_xi0246(xi,np.linspace(0,1,mu),s)

    #nr = 2.5+np.linspace(5.,150.,nbins_s+1)[:-1]
    r = bins[:-1]
    plt.plot(r,(xi_l[0])*r**2,label=r'$\xi_0$',c='C00')
    #plt.plot(r,(-xi_l[1]),label=r'$\xi_2$',c='C01')
    #plt.plot(r,(r**2)*(-xi_l[2]),label=r'$\xi_4$',c='C02')

    plt.xlabel('s')
    plt.ylabel(r'$s^2\xi_{\ell}$')
    plt.legend()
    plt.savefig(filename+'_xil.png')
    plt.close()

# Main execution
if __name__ == "__main__":

    import time

    t1=time.time()

    zmin, zmax = 0.05, 0.15
    nthreads = 128
    sample = 'filgxs' #['filgxs','nonfilgxs','allgxs']

    print(zmin,zmax,sample)

    # Read data
    sdss = read_data(zmin,zmax)

    # Select filament galaxies
    if sample=='filgxs': sdss = sdss[sdss['dist_fil']<=3.]
    if sample=='nonfilgxs': sdss = sdss[sdss['dist_fil']>=8.]

    # Generate Randoms
    randoms = generate_random_catalog(60, sdss, nside=128, write=False)

    # Dilute for testing
    #sdss = sdss.sample(n=20000)
    #randoms = randoms[:20000]

    # Calculate xi(mu, s)
    ra_data = sdss['ra'].values
    dec_data = sdss['dec'].values
    z_data = sdss['red'].values
    ra_rand = randoms['ra'].values
    dec_rand = randoms['dec'].values
    z_rand = randoms['red'].values
    bins = np.linspace(10, 50, 41)  # Separation bins in Mpc/h
    pimax = 40  # 
    xi = calculate_xi_rp_pi(ra_data, dec_data, z_data, ra_rand, dec_rand, z_rand, bins=bins, pimax=pimax, nthreads=nthreads)

    # Plot xi(mu, s)
    filename = f'rppi_corrfunc_{sample}_Nd{len(sdss)}_Nr{len(randoms)}'
    plot_xi_rp_pi_contours(xi, bins, pimax, filename+'.png')

    # Save results
    np.savez(filename+'.npz', xi=xi, rp=bins, pimax=pimax)

    # PLot xi_l
    #plot_xil(filename)

    t2 = time.time()

    print((t2-t1)/60,'min')