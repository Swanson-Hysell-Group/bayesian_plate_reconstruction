import numpy as np
import numpy.polynomial.legendre as legendre
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs

import mcplates

cmap_green = LinearSegmentedColormap.from_list('vphi', [(0, '#ffffff'), (0.2, '#edf8e9'), (
    0.4, '#bae4b3'), (0.6, '#74c476'), (0.8, '#31a354'), (1.0, '#006d2c')], gamma=0.5)
cmap_green.set_bad('w', alpha=0.0)

cmap_blue = LinearSegmentedColormap.from_list('vphi', [(0, '#ffffff'), (0.2, '#eff3ff'), (
    0.4, '#bdd7e7'), (0.6, '#6baed6'), (0.8, '#3182bd'), (1.0, '#08519c')], gamma=0.5)
cmap_blue.set_bad('w', alpha=0.0)
cmap_red = LinearSegmentedColormap.from_list('vphi', [(0, '#ffffff'), (0.2, '#fee5d9'), (
    0.4, '#fcae91'), (0.6, '#fb6a4a'), (0.8, '#de2d26'), (1.0, '#a50f15')], gamma=0.5)
cmap_red.set_bad('w', alpha=0.0)



d2r = np.pi/180.
r2d = 180./np.pi

f_lon, f_lat = 30.,30.
f_kappa = 20.
w_lon, w_lat = 90.,70.
w_kappa=-5.



vmf = mcplates.VonMisesFisher('vmf', lon_lat=(f_lon,f_lat), kappa=f_kappa)
watson = mcplates.WatsonGirdle('watson', lon_lat=(w_lon,w_lat), kappa=w_kappa)
uniform = mcplates.VonMisesFisher('watson', lon_lat=(0,0), kappa=0.)
vmf_samples = np.array([vmf.random() for i in range(50)])
wat_samples = np.array([watson.random() for i in range(200)])
u_samples = np.array([uniform.random() for i in range(200)])

reg_lat = np.linspace(-np.pi/2., np.pi/2., 181, endpoint=True)*180./np.pi
reg_lon = np.linspace(0., 2.*np.pi, 361, endpoint=True)*180./np.pi
mesh_lon, mesh_lat = np.meshgrid(reg_lon, reg_lat)
uniform_vals = np.empty_like(mesh_lon)
fisher_vals = np.empty_like(mesh_lon)
watson_vals = np.empty_like(mesh_lon)
for i, lat in enumerate(reg_lat):
    for j,lon in enumerate(reg_lon):
        fisher_vals[i,j] = np.exp(mcplates.vmf_logp( np.array([lon,lat]), np.array([f_lon, f_lat]), kappa=f_kappa ))
        watson_vals[i,j] = np.exp(mcplates.watson_girdle_logp( np.array([lon,lat]), np.array([w_lon, w_lat]), kappa=w_kappa ))
        uniform_vals[i,j] = np.exp(mcplates.vmf_logp( np.array([lon,lat]), np.array([0, 0]), 0.0))


fig = plt.figure(figsize=(4,12))

ax = fig.add_subplot(3,1,1, projection = ccrs.Orthographic(30,30))
c = ax.pcolormesh(mesh_lon,mesh_lat, fisher_vals,  transform=ccrs.PlateCarree(), cmap=cmap_red)
ax.scatter(vmf_samples[:,0], vmf_samples[:,1], transform=ccrs.PlateCarree(), c='darkred', edgecolors='darkred')
ax.gridlines()
ax.set_global()
ax.set_title("von Mises-Fisher distribution")

ax = fig.add_subplot(3,1,2, projection = ccrs.Orthographic(30,30))
c = ax.pcolormesh(mesh_lon,mesh_lat, watson_vals,  transform=ccrs.PlateCarree(), cmap=cmap_blue)
ax.scatter(wat_samples[:,0], wat_samples[:,1], transform=ccrs.PlateCarree(), c='b', edgecolors='b')
ax.gridlines()
ax.set_global()
ax.set_title("Watson girdle distribution")

ax = fig.add_subplot(3,1,3, projection = ccrs.Orthographic(30,30))
c = ax.pcolormesh(mesh_lon,mesh_lat, uniform_vals,  transform=ccrs.PlateCarree(), cmap=cmap_green, vmin=-0.2, vmax=1.0)
ax.scatter(u_samples[:,0], u_samples[:,1], transform=ccrs.PlateCarree(), c='darkgreen', edgecolors='darkgreen')
ax.gridlines()
ax.set_global()
ax.set_title("Uniform distribution")

#plt.show()
plt.savefig("spherical_distributions.pdf")
