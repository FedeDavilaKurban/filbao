import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

nbins     = np.int32(200)
rmin_perp = np.float64(0.0)
rmax_perp = np.float64(200.0)
rmin_pll  = np.float64(0.0)
rmax_pll  = np.float64(200.0)
ngald     = 401359
#ngalr     = 561286
ngalr     = 4483837

#pairs_DD = np.loadtxt('pairs_DD_rpmin_%.1f_rpmax_%.1f_pimin_%.1f_pmax_%.1f_nbins_%d.out' % (rmin_perp, rmax_perp, rmin_pll, rmax_pll, nbins))
#pairs_RR = np.loadtxt('pairs_RR_rpmin_%.1f_rpmax_%.1f_pimin_%.1f_pmax_%.1f_nbins_%d.out' % (rmin_perp, rmax_perp, rmin_pll, rmax_pll, nbins))
pairs_DD = np.loadtxt('pairs_DD_rpmin_%.1f_rpmax_%.1f_pimin_%.1f_pmax_%.1f_nbins_%d_ndata_%d_ndata_%d.out' \
           % (rmin_perp, rmax_perp, rmin_pll, rmax_pll, nbins, ngald, ngald))
pairs_RR = np.loadtxt('pairs_RR_rpmin_%.1f_rpmax_%.1f_pimin_%.1f_pmax_%.1f_nbins_%d_nrand_%d_nrand_%d.out' \
           % (rmin_perp, rmax_perp, rmin_pll, rmax_pll, nbins, ngalr, ngalr))

#pairs_DD[0] -= ngald
#pairs_RR[0] -= ngalr
pairs_DD = pairs_DD.reshape((nbins,nbins))
pairs_RR = pairs_RR.reshape((nbins,nbins))
#skip     = 20
#pairs_DD[:skip,:]  = np.nan
#pairs_RR[:skip,:]  = np.nan
#pairs_DD[:,:skip] = np.nan
#pairs_RR[:,:skip] = np.nan
pairs_DD = pairs_DD.T
pairs_RR = pairs_RR.T

xi = (ngalr*ngalr*pairs_DD)/(pairs_RR*ngald*ngald) - 1

x = np.linspace(rmin_perp, rmax_perp, nbins)
y = np.linspace(rmin_perp, rmax_pll, nbins)
X, Y = np.meshgrid(np.concatenate((-x[::-1],x)), np.concatenate((-y[::-1],y)))

C = np.zeros((2*nbins, 2*nbins))
C[:nbins, :nbins] = np.flipud(np.fliplr(xi))
C[:nbins, nbins:] = np.flipud(xi)
C[nbins:, :nbins] = np.fliplr(xi)
C[nbins:, nbins:] = xi

#X, Y = np.meshgrid(x,y)
#plt.imshow(xi, interpolation='nearest', origin='lower', cmap=plt.cm.Purples, extent=[rmin_perp, rmax_perp, rmin_pll, rmax_pll])
#contours = plt.contour(X, Y, xi, 10, colors='black')

levels = np.geomspace(1e-6,18,10)
#plt.imshow(C, interpolation='nearest', cmap=plt.cm.Purples, origin='lower', extent=[-rmax_perp, rmax_perp, -rmax_pll, rmax_pll])

cs = plt.contourf(X, Y, C, levels, cmap=plt.cm.PuBu_r)
contours = plt.contour(X, Y, C, levels, colors='black')
plt.clabel(contours, contours.levels, fontsize=8)
plt.show()
