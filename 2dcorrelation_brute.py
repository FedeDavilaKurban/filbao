def H(ra, dec, z):

    ra  = np.array(ra,  dtype=np.float64) 
    dec = np.array(dec, dtype=np.float64) 
    z   = np.array(z,   dtype=np.float64) 

    r = np.float64(cosmo.comoving_distance(z).value)
    x = r*np.cos(np.deg2rad(ra))*np.cos(np.deg2rad(dec))
    y = r*np.sin(np.deg2rad(ra))*np.cos(np.deg2rad(dec))
    z = r*np.sin(np.deg2rad(dec))
    return x, y, z

import pandas as pd
import numpy as np
import treecorr
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)

# Read Gxs
zmin, zmax = 0.01, 0.15
cat_sdss = pd.read_csv('./sdss_zmin_0.000_zmax_0.300_sigma_5.0.csv')
cat_sdss  = cat_sdss[(cat_sdss["red"] > zmin)*(cat_sdss["red"] < zmax)]

# Filament Gxs
filgxs = cat_sdss[cat_sdss['dist_fil']<=3.]

# Non-filament Gxs
nonfilgxs = cat_sdss[cat_sdss['dist_fil']>8.25]

# Read Randoms & Build Catalog
cat_random = pd.read_csv('./random_sample_healpy_128_4483837.csv')
cat_random["x"], cat_random["y"], cat_random["z"] = H(cat_random["ra"], cat_random["dec"], cat_random["red"])

import os
import numpy.ctypeslib as npct

# Compilar
exe = 'force_brute'
comando = 'gcc -c -g -fPIC -lm -fopenmp %s.c -o %s.o' % (exe, exe)
os.system(comando)

# Ejecutar
comando = 'gcc -g -shared -lm -fopenmp %s.o -o %s.so' % (exe, exe)
os.system(comando)

print("Compila")

array_1d_double = npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')

#Cargar libreria
libcd = npct.load_library('%s.so' % exe, '.')

libcd.calcular_correlacion.argtypes = [\
    npct.ctypes.c_int,\
    npct.ctypes.c_int,\
    npct.ctypes.c_int,\
    npct.ctypes.c_double,\
    npct.ctypes.c_double,\
    npct.ctypes.c_double,\
    npct.ctypes.c_double,\
    array_1d_double,\
    array_1d_double,\
    array_1d_double,\
    array_1d_double,\
    array_1d_double,\
    array_1d_double,\
    array_1d_double]

libcd.calcular_autocorrelacion.argtypes = [\
    npct.ctypes.c_int,\
    npct.ctypes.c_int,\
    npct.ctypes.c_double,\
    npct.ctypes.c_double,\
    npct.ctypes.c_double,\
    npct.ctypes.c_double,\
    array_1d_double,\
    array_1d_double,\
    array_1d_double,\
    array_1d_double]

ra_d  = np.deg2rad(np.ascontiguousarray(cat_sdss['ra'].to_numpy(dtype=np.float64)))
dec_d = np.deg2rad(np.ascontiguousarray(cat_sdss['dec'].to_numpy(dtype=np.float64)))
xd = np.ascontiguousarray(cat_sdss['x'].to_numpy(dtype=np.float64))
yd = np.ascontiguousarray(cat_sdss['y'].to_numpy(dtype=np.float64))
zd = np.ascontiguousarray(cat_sdss['z'].to_numpy(dtype=np.float64))
rd = np.sqrt(xd*xd + yd*yd + zd*zd)

ra_r  = np.deg2rad(np.ascontiguousarray(cat_random['ra'].to_numpy(dtype=np.float64)))
dec_r = np.deg2rad(np.ascontiguousarray(cat_random['dec'].to_numpy(dtype=np.float64)))
xr = np.ascontiguousarray(cat_random['x'].to_numpy(dtype=np.float64))
yr = np.ascontiguousarray(cat_random['y'].to_numpy(dtype=np.float64))
zr = np.ascontiguousarray(cat_random['z'].to_numpy(dtype=np.float64))
rr = np.sqrt(xr*xr + yr*yr + zr*zr)

nbins     = np.int32(200)
rmin_perp = np.float64(0.0)
rmax_perp = np.float64(200.0)
rmin_pll  = np.float64(0.0)
rmax_pll  = np.float64(200.0)
ngald     = np.int32(len(cat_sdss))
ngalr     = np.int32(len(cat_random))

#pairs_DD = np.zeros(nbins*nbins, dtype=np.float64)
#libcd.calcular_autocorrelacion(ngald,nbins,rmin_perp,rmax_perp,rmin_pll,rmax_pll,ra_d,dec_d,rd,pairs_DD)
#np.savetxt('pairs_DD_rpmin_%.1f_rpmax_%.1f_pimin_%.1f_pmax_%.1f_nbins_%d_ndata_%d_ndata_%d.out' % (rmin_perp, rmax_perp, rmin_pll, rmax_pll, nbins, ngald, ngald), pairs_DD)

#pairs_DR = np.zeros(nbins*nbins, dtype=np.float64)
#libcd.calcular_correlacion(ngald,ngalr,nbins,rmin_perp,rmax_perp,rmin_pll,rmax_pll,ra_d,dec_d,rd,ra_r,dec_r,rr,pairs_DR)
#np.savetxt('pairs_DR_rpmin_%.1f_rpmax_%.1f_pimin_%.1f_pmax_%.1f_nbins_%d_ndata_%d_nrand_%d.out' % (rmin_perp, rmax_perp, rmin_pll, rmax_pll, nbins, ngald, ngalr), pairs_DR)

pairs_RR = np.zeros(nbins*nbins, dtype=np.float64)
libcd.calcular_autocorrelacion(ngalr,nbins,rmin_perp,rmax_perp,rmin_pll,rmax_pll,ra_r,dec_r,rr,pairs_RR)
np.savetxt('pairs_RR_rpmin_%.1f_rpmax_%.1f_pimin_%.1f_pmax_%.1f_nbins_%d_nrand_%d_nrand_%d.out' % (rmin_perp, rmax_perp, rmin_pll, rmax_pll, nbins, ngalr, ngalr), pairs_RR)
