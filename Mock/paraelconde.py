import sys
import os
import numpy as np
import pandas as pd
import healpy as hp
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d

def create_sdss_mask(nside, plot=False): 
  
  cat_sdss = pd.read_csv('/home/lpereyra/BAO/CONDE/sdss_zmin_0.000_zmax_0.300_sigma_5.0.csv.gz',compression='gzip')
  
  npix = hp.nside2npix(nside)
  mask = np.zeros(npix, dtype=np.int32)
  pixeles = hp.ang2pix(nside, cat_sdss['ra'].values, cat_sdss['dec'].values, lonlat=True)
  mask[pixeles] = 1

  if plot == True:
    import matplotlib.pyplot as plt
    hp.mollview(mask)
    plt.show()

  return mask

H_0  = 67.74
cosmo = FlatLambdaCDM(H0=H_0, Om0=0.3089)
z_space = np.linspace(0., 1., 500)
r_space = np.float64(cosmo.comoving_distance(z_space).value)
aux_dis2red = interp1d(r_space, z_space, kind="cubic")
lbox = 1000.0/cosmo.h
chunksize = 1_000_000
cut_redshift = 0.17
cut_mag = -21.0
nside = 128

assert(cut_redshift <= aux_dis2red(lbox*0.5))

# LE PONE LA MASCARA DEL SDSS A LA ESFERA
mask_sdss = create_sdss_mask(nside)  

filename = './TO_MOCK.csv'
count = 0
data_index, data_real = [], []
for chunk in pd.read_csv(filename, sep=",", na_values=r'\N', chunksize=chunksize):

  print("Step %d..." % (count))
  index = np.arange(len(chunk), dtype=np.int64) + count*chunksize
  x   = chunk["x"]/cosmo.h
  y   = chunk["y"]/cosmo.h
  z   = chunk["z"]/cosmo.h
  vx  = chunk["vx"]
  vy  = chunk["vy"]
  vz  = chunk["vz"]
  mag = chunk["magstarsdssr"]

  mask  = mag < cut_mag 
  
  mag = mag[mask]
  x   =  x[mask] % lbox # Periodicidad
  y   =  y[mask] % lbox #
  z   =  z[mask] % lbox #
  vx  = vx[mask]
  vy  = vy[mask]
  vz  = vz[mask]  
  index = index[mask]

  # Resta el centro de la caja
  x -= 0.5*lbox 
  y -= 0.5*lbox 
  z -= 0.5*lbox 
  r = np.sqrt(x**2 + y**2 + z**2)
  r = np.where(r == 0, 1e-10, r) # Reemplaza ceros por un valor muy pequeño
  
  x /= r 
  y /= r 
  z /= r 
  
  vrad = (x*vx + y*vy + z*vz)
  dcom = r + vrad/H_0 # ACA LE SUMO LAS VELOCIDADES PECULIARES
  red  = aux_dis2red(dcom)
  
  RA, Dec = hp.vec2ang(np.vstack((x, y, z)).T, lonlat=True) # 
  pixeles = hp.ang2pix(nside, RA, Dec, lonlat=True) # si caigo en un pixel de la mascara

  mask  = red < cut_redshift # ESTE SERIA EL CORTE EN REDSHIFT
  mask *= mask_sdss[pixeles] == 1 # MASCARA DEL SDSS
  
  mag  = mag[mask]
  x    =  x[mask]*dcom
  y    =  y[mask]*dcom
  z    =  z[mask]*dcom
  vx   = vx[mask]
  vy   = vy[mask]
  vz   = vz[mask]  
  RA   = RA[mask]
  Dec  = Dec[mask]
  red  = red[mask]
  dcom = dcom[mask]
  index = index[mask]

  tmp = np.vstack((RA,Dec,red)).T 
  tmp_xyz = np.vstack((x,y,z)).T 
  data_real.append(np.float64(tmp))
  data_index.append(np.int64(index))

  print("End Step %d..." % (count))
  count += 1

print("Nloop %d" % count) 
data_real  = np.concatenate(data_real)
data_index = np.concatenate(data_index)
cat = pd.DataFrame({'id':data_index, 'ra':data_real[:,0], 'dec':data_real[:,1], 'red':data_real[:,2]})

# SI NECESITO EL MOCK SELECCIONADO LOS FILAMENTOS

fname_fil = "./mock_withfilament.csv"
cat_fil = pd.read_csv(fname_fil)
# los campos de cat_fil son
#'id','ifil', 'dfil', 'long', 'vx', 'vy', 'vz','vx_s', 'vy_s', 'vz_s'
#
#'id': id gal
#'ifil': id_fil
#'dfil': distancia perpendicular al filamento mas cercano [Mpc/h]
#'long': longitud del filamento mas cercano [Mpc/h]
#'vx',   'vy',   'vz':   versor del filamento mas cercano
#'vx_s', 'vy_s', 'vz_s': versor suavizado del filamento mas cercano
#'dfil', 'long', 'vx', 'vy', 'vz','vx_s', 'vy_s', 'vz_s'
#'id','ifil', 'dfil', 'long', 'vx', 'vy', 'vz','vx_s', 'vy_s', 'vz_s'
cat = pd.merge(cat, cat_fil, on='id', how='inner')
cat = cat[cat["dfil"] > 8.5] # ACA FILTRO
ndata = len(cat)

print(ndata)
