import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate
import seaborn as sns 

nbins     = np.int32(200)
rmin_perp = np.float64(0.0)
rmax_perp = np.float64(200.0)
rmin_pll  = np.float64(0.0)
rmax_pll  = np.float64(200.0)
ngald     = 401359
ngalr     = 4483837

#pairs_DD = np.loadtxt('pairs_DD_rpmin_%.1f_rpmax_%.1f_pimin_%.1f_pmax_%.1f_nbins_%d.out' % (rmin_perp, rmax_perp, rmin_pll, rmax_pll, nbins))
#pairs_RR = np.loadtxt('pairs_RR_rpmin_%.1f_rpmax_%.1f_pimin_%.1f_pmax_%.1f_nbins_%d.out' % (rmin_perp, rmax_perp, rmin_pll, rmax_pll, nbins))
pairs_DD = np.loadtxt('pairs_DD_rpmin_%.1f_rpmax_%.1f_pimin_%.1f_pmax_%.1f_nbins_%d_ndata_%d_ndata_%d.out' \
           % (rmin_perp, rmax_perp, rmin_pll, rmax_pll, nbins, ngald, ngald))
pairs_DR = np.loadtxt('pairs_DR_rpmin_%.1f_rpmax_%.1f_pimin_%.1f_pmax_%.1f_nbins_%d_ndata_%d_nrand_%d.out' \
           % (rmin_perp, rmax_perp, rmin_pll, rmax_pll, nbins, ngald, ngalr))
pairs_RR = np.loadtxt('pairs_RR_rpmin_%.1f_rpmax_%.1f_pimin_%.1f_pmax_%.1f_nbins_%d_nrand_%d_nrand_%d.out' \
           % (rmin_perp, rmax_perp, rmin_pll, rmax_pll, nbins, ngalr, ngalr))

pairs_DD = pairs_DD.reshape((nbins,nbins))
pairs_DR = pairs_DR.reshape((nbins,nbins))
pairs_RR = pairs_RR.reshape((nbins,nbins))
#skip     = 5
#pairs_DD[:skip,:] = np.nan
#pairs_DR[:skip,:] = np.nan
#pairs_DD[:,:skip] = np.nan
#pairs_DR[:,:skip] = np.nan

skip      = 120
nbins     = skip 
rmax_perp = np.float64(skip)
rmax_pll  = np.float64(skip)
pairs_DD= pairs_DD[:skip,:skip]
pairs_DR= pairs_DR[:skip,:skip]
pairs_RR= pairs_RR[:skip,:skip]

pairs_DD = pairs_DD.T / ((ngald-1)*ngald*0.5)
pairs_DR = pairs_DR.T / (ngald*ngalr)
pairs_RR = pairs_RR.T / ((ngalr-1)*ngalr*0.5)

#xi = (2*ngalr*pairs_DD)/(pairs_DR*ngald) - 1
#xi = (ngalr*ngalr*pairs_DD)/(pairs_RR*ngald*ngald) - 1
xi = (pairs_DD - 2*pairs_DR + pairs_RR)/(pairs_RR)

C = np.zeros((2*nbins, 2*nbins))
C[:nbins, :nbins] = np.flipud(np.fliplr(xi))
C[:nbins, nbins:] = np.flipud(xi)
C[nbins:, :nbins] = np.fliplr(xi)
C[nbins:, nbins:] = xi

x = np.linspace(rmin_perp, rmax_perp, nbins+1)
y = np.linspace(rmin_pll,  rmax_pll,  nbins+1)
x = 0.5*(x[1:]+x[:-1])
y = 0.5*(y[1:]+y[:-1])
X, Y = np.meshgrid(np.concatenate((-x[::-1],x)), np.concatenate((-y[::-1],y)))
func = interpolate.RectBivariateSpline(X[0,:], Y[:,0], C, s=0)

xnew_edges, ynew_edges = np.mgrid[-rmax_perp:rmax_perp:(nbins+1)*1j, -rmax_pll:rmax_pll: (nbins+1)*1j]
Xnew = xnew_edges[:-1, :-1] + np.diff(xnew_edges[:2, 0])[0] / 2.
Ynew = ynew_edges[:-1, :-1] + np.diff(ynew_edges[0, :2])[0] / 2.
Cnew = func(Xnew[:,0], Ynew[0,:]).T

#plt.imshow(xi, interpolation='nearest', origin='lower', cmap=plt.cm.Purples, extent=[rmin_perp, rmax_perp, rmin_pll, rmax_pll])
#contours = plt.contour(X, Y, xi, 10, colors='black')

levels = np.log10(np.geomspace(xi.min()-1e-3,xi.max()+1e-3,20))
#levels = np.array([0, 0.01, 0.03,0.08, 0.19, 0.46, 1.14, 2.81, 6.93, 17.12])
#levels[1:] = np.log10(levels[1:])
#levels[0]  = -2.5

cmap = sns.color_palette("Spectral_r", as_cmap=True)
#cs = plt.contourf(X, Y, C, 10**levels, cmap=cmap, norm=mpl.colors.LogNorm())

#fig, (ax1, ax2) = plt.subplots(1, 2)

#cs = ax1.contourf(X   , Y   , C   , 10**levels, cmap=cmap, norm=mpl.colors.LogNorm())
cs = plt.contourf(Xnew, Ynew, Cnew, 10**levels, cmap=cmap, norm=mpl.colors.LogNorm())
#norm = mpl.colors.Normalize(0, levels)#len(levels))
#plt.imshow(C, interpolation='nearest', cmap=plt.cm.Purples, origin='lower', extent=[-rmax_perp, rmax_perp, -rmax_pll, rmax_pll])

#cs = plt.contourf(X, Y, C, levels, cmap=cmap) #, norm=norm)
#contours = plt.contour(Xnew, Ynew, C, levels, colors='black')
#cs = plt.contourf(Xnew, Ynew, Cnew, levels, cmap=cmap, locator=mpl.ticker.LogLocator()) #, norm=norm)
#contours = plt.contour(Xnew, Ynew, Cnew, levels, colors='black', locator=mpl.ticker.LogLocator())
#plt.clabel(contours, contours.levels, fontsize=8)
#plt.colorbar(cs)
plt.show()
