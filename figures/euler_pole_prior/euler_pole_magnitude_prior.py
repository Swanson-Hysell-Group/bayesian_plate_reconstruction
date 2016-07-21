import os
import numpy as np
import numpy.random as random
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import scipy.special as sp
import scipy.stats as st

import pymc

import mcplates

plt.style.use('../bpr.mplstyle')

dbname = 'euler_pole_magnitude_prior.pickle'

def generate_data_samples():
    if os.path.isfile('euler_pole_magnitude_samples.txt'):
         data = np.loadtxt('euler_pole_magnitude_samples.txt')
         lon_samples = data[:,0]
         lat_samples = data[:,1]
         val_samples = data[:,2]
    else:

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
        plate_data = np.loadtxt("WhichPlate.dat")
        vals = plate_data[:,2]

        nlons = 256.
        nlats = 128.
        dlon = 360./nlons
        dlat = 180./nlats

        vals = vals.reshape(nlats,nlons)


        n_samples = 1000
        val_samples = np.zeros(n_samples)
        lon_samples = np.zeros(n_samples)
        lat_samples = np.zeros(n_samples)

        uniform_lon_lat_sampler = mcplates.VonMisesFisher('lon_lat_sampler', lon_lat=(0.,0.), kappa=0.)

        i=0
        while i < n_samples:
            sample = uniform_lon_lat_sampler.random()
            lon = sample[0]
            lat = sample[1]
            try:
                lon_index = np.floor(lon/dlon)
                lat_index = np.floor((90.-lat)/dlat)
                plate_id = vals[lat_index, lon_index]
                plate_code = plate_id_to_code[plate_id]

                elat = morvel['Latitude'][plate_code]
                elon = morvel['Longitude'][plate_code]
                emag = morvel['AngularRate'][plate_code]

                lon_samples[i] = lon
                lat_samples[i] = lat
                val_samples[i] = emag 
                i += 1
            except KeyError:
                continue
        np.savetxt('euler_pole_magnitude_samples.txt', zip(lon_samples,lat_samples,val_samples))

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
    print vals
    return np.empty_like(vals), np.empty_like(vals), vals


# Sample rotation rates for current plate configuration
lon_samples,lat_samples,val_samples = generate_data_samples()

rate = pymc.Uniform('rate', 0., 5.)
angular_distance = np.empty( len(val_samples)+1, dtype=object)
angular_distance[len(val_samples)] = rate

for i,s in enumerate(val_samples):
  angular_distance[i] = pymc.Exponential('rate_'+str(i), rate, value=s, observed=True )

model = pymc.Model(angular_distance)
mcmc = pymc.MCMC(model, db='pickle', dbname=dbname)

if os.path.isfile(dbname):
    db = pymc.database.pickle.load(dbname)
else:
    pymc.MAP(model).fit()
    mcmc.sample(10000)
    mcmc.db.close()
    db = pymc.database.pickle.load(dbname)

rate_trace = db.trace('rate')[:]
plt.hist( rate_trace, normed=True)
plt.show()

plt.hist( val_samples, bins=3, normed=True)
x = np.linspace(0., 3., 100)
plt.plot(x, st.expon.pdf(x, scale=1.), label=r'$\lambda = 1.0$', lw=3)
plt.plot(x, st.expon.pdf(x, scale=1./2.), label=r'$\lambda = 2.0$', lw=3)
#plt.plot(x, st.expon.pdf(x, scale=10.), label=r'$\lambda = 1/10$', lw=3)
plt.legend(loc='upper right')
plt.xlabel(r'Rotation rate $\,^\circ / \mathrm{Myr}$')
plt.ylabel(r'Probability density')
plt.show()
