import os, sys
import numpy as np
import numpy.random as random
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import scipy.special as sp
import scipy.stats as st

import pymc

sys.path.insert(1, os.path.abspath('../../mcplates'))
import mcplates

plt.style.use('../bpr.mplstyle')

dbname = 'euler_pole_magnitude_prior.pickle'

def generate_data_samples():
     data = np.loadtxt('euler_pole_magnitude_samples.txt')
     lon_samples = data[:,0]
     lat_samples = data[:,1]
     val_samples = data[:,2]
     return lon_samples,lat_samples,val_samples

def generate_data_samples_unweighted():
    plate_id_to_code = { 0 : 'an',

                         1 : 'au',
                         2 : 'nb',
                         3 : 'pa',
                         4 : 'eu',
                         5 : 'na',
                         6 : 'nz',
                         7 : 'co',
                         8 : 'ca',
                         9 : 'ar',
                        10 : 'ps',
                        11 : 'sa',
                        12 : 'in',
                        13 : 'jf' }
                         

    morvel = pd.read_table("NNR-MORVEL56.txt", delim_whitespace=True).set_index('Abbreviation')
    vals = [morvel['AngularRate'][p] for p in plate_id_to_code]
    return np.empty_like(vals), np.empty_like(vals), vals

mu_lat = 90.
mu_lon =0.
kappa=-0.8
#Setup grid
n = 90
reg_lat = np.linspace(-90., 90., 181., endpoint=True)
reg_lon = np.linspace(0., 360., 2*n+1, endpoint = True )
mesh_lon, mesh_lat = np.meshgrid(reg_lon, reg_lat)

#Evaluate Watson girdle distribution
mesh_vals = np.empty_like(mesh_lon)
for i, lat in enumerate(reg_lat):
    for j,lon in enumerate(reg_lon):
        mesh_vals[i,j] = np.exp(mcplates.watson_girdle_logp( np.array([lon,lat]), np.array([mu_lon, mu_lat]), kappa ))


# Sample rotation rates for current plate configuration
lon_samples,lat_samples,val_samples = generate_data_samples()

#plot magnitude probabilities
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(1,2,1)
ax1.hist( val_samples, bins=[0., 0.4, 0.8, 1.2], normed=True)
x = np.linspace(0., 3., 100)
ax1.plot(x, st.expon.pdf(x, scale=1.), label=r'$\lambda = 1.0$', lw=3)
ax1.plot(x, st.expon.pdf(x, scale=1./2.5), label=r'$\lambda = 2.5$', lw=3)
ax1.plot(x, 0.25*np.ones_like(x), label=r'$U(0,4)$', lw=3)
ax1.legend(loc='upper right')
ax1.set_xlabel(r'Rotation rate $\,^\circ / \mathrm{Myr}$')
ax1.set_ylabel(r'Probability density')
ax1.set_title('(a) Rate prior')


#plot position probabilities
ax2 = fig.add_subplot(1,2,2, projection = ccrs.Orthographic(0., 30.))
c = ax2.pcolormesh(mesh_lon,mesh_lat, mesh_vals, cmap='copper', transform=ccrs.PlateCarree(),rasterized=True)
ax2.gridlines()
ax2.set_global()
ax2.set_title('(b) Position prior')
plt.colorbar(c, ax=ax2)

plt.tight_layout()
#plt.show()
plt.savefig("euler_pole_prior.pdf")

