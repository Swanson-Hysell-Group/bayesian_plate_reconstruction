from __future__ import print_function
import itertools
import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys

import mcplates

plt.style.use('../bpr.mplstyle')
from mcplates.plot import cmap_red, cmap_green, cmap_blue

# Shift all longitudes by 180 degrees to get around some plotting
# issues. This is error prone, so it should be fixed eventually
lon_shift = 0.

# List of colors to use
#colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
#          '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']


# Parse input
#Get number of euler rotations
if len(sys.argv) < 2:
    raise Exception("Please enter the number of Euler rotations to fit.")
n_euler_rotations = int(sys.argv[1])
if n_euler_rotations < 1:
    raise Exception("Number of Euler rotations must be greater than zero")
# Read in optional map projection info
if len(sys.argv) > 2:
    proj_type = sys.argv[2].split(',')[0]
    assert proj_type == 'M' or proj_type == 'O'
    proj_lon = float(sys.argv[2].split(',')[1]) - lon_shift
    if proj_type == 'O':
        proj_lat = float(sys.argv[2].split(',')[2])
else:
    proj_type = 'M'
    proj_lon = -lon_shift
    proj_lat = 0.

print("Fitting Australia APW track with "+str(n_euler_rotations)+" Euler rotation" + ("" if n_euler_rotations == 1 else "s") )


data = pd.read_csv("GPMDB_Australia_50Ma.csv")
# Give unnamed column an appropriate name
data.sort_values('HIGHAGE', ascending=False, inplace=True)

# Add poles from Australia pole list
poles = []
for i, row in data.iterrows():
    pole_lat = row['PLAT']
    pole_lon = row['PLONG'] - lon_shift
    a95 = row['ED95']
    age = (row['HIGHAGE']+row['LOWAGE'])/2.

    sigma_age = (row['LOWAGE'], row['HIGHAGE'])

    pole = mcplates.PaleomagneticPole(
        pole_lon, pole_lat, angular_error=a95, age=age, sigma_age=sigma_age)
    poles.append(pole)

# Add pole with low error for present day pole
poles.append( mcplates.PaleomagneticPole( 0., 90., angular_error=1., age=0., sigma_age=0.01) )

# Reference position on the Australian continent
slat = -25.3  # Uluru lat
slon = 131.0 - lon_shift  # Uluru lon
uluru = mcplates.PlateCentroid(slon, slat)

path = mcplates.APWPath(
    'australia_apw_' + str(n_euler_rotations), poles, n_euler_rotations)
path.create_model(site_lon_lat=(slon, slat), watson_concentration=-0.0)


## Create a function that reproduces the APW path
## of Mueller et al 1993, for Australia relative
## to a hotspot reference frame.
class muller_etal_1993_apw_path_function(object):
    def __init__(self):
        # enumerate euler poles, rotation angles, and changepoints for the Cenozoic
        self.angles = np.array([7.27, 13.23-7.27, 22.47-13.23, 26.0-22.47, 27.67-26.0, 27.86-27.67, 28.72-27.86])
        self.changepoints = np.array([10.4, 20.5, 35.5, 42.7, 50.3, 58.6, 68.5])
        
        self.e1 = mcplates.EulerPole( 40.6+180., -23.7, self.angles[0]/self.changepoints[0]) 
        self.e2 = mcplates.EulerPole( 31.4+180., -27.1, self.angles[1]/(self.changepoints[1]-self.changepoints[0])) 
        self.e3 = mcplates.EulerPole( 30.1+180., -22.7, self.angles[2]/(self.changepoints[2]-self.changepoints[1])) 
        self.e4 = mcplates.EulerPole( 27.8+180., -24.0, self.angles[3]/(self.changepoints[3]-self.changepoints[2])) 
        self.e5 = mcplates.EulerPole( 25.9+180., -23.3, self.angles[4]/(self.changepoints[4]-self.changepoints[3])) 
        self.e6 = mcplates.EulerPole( 26.2+180., -23.0, self.angles[5]/(self.changepoints[5]-self.changepoints[4])) 
        self.e7 = mcplates.EulerPole( 26.8+180., -18.3, self.angles[6]/(self.changepoints[6]-self.changepoints[5])) 
        self.euler_poles = [self.e1,self.e2,self.e3,self.e4,self.e5,self.e6,self.e7]

    def __call__(self,age):
        pole = mcplates.Pole( 0., -90., 1.)
        current_age=0
        for euler, angle, change in zip(self.euler_poles, self.angles, self.changepoints):
            if age > change:
                pole.rotate( euler, -angle )
                current_age = change
            else:
                pole.rotate( euler, -euler.rate*(age-current_age))
                break
        return pole.longitude, pole.latitude

    def speed(self, age):
        assert(age < self.changepoints[-1] and age >= 0)
        for i,c in enumerate(self.changepoints):
            if c > age:
                return self.euler_poles[i].speed_at_point(uluru)

muller_apw = muller_etal_1993_apw_path_function()

def plot_synthetic_paths():

    if proj_type == 'M':
        ax = plt.axes(projection=ccrs.Mollweide(proj_lon))
    elif proj_type == 'O':
        ax = plt.axes(projection = ccrs.Orthographic(proj_lon,proj_lat))

    ax.gridlines()
    ax.set_global()

    mcplates.plot.plot_continent(ax, 'australia', rotation_pole=mcplates.Pole(0., 90., 1.0), angle=-lon_shift, color='k')
    mcplates.plot.plot_continent(ax, 'antarctica', rotation_pole=mcplates.Pole(0., 90., 1.0), angle=-lon_shift, color='k')

    direction_samples = path.euler_directions()

    dist_colors = itertools.cycle([cmap_blue, cmap_red, cmap_green])
    for directions in direction_samples:
        mcplates.plot.plot_distribution(ax, directions[:, 0], directions[:, 1], cmap=dist_colors.next(), resolution=30)


    pathlons, pathlats = path.compute_synthetic_paths(n=200)
    for pathlon, pathlat in zip(pathlons, pathlats):
        ax.plot(pathlon-180., -pathlat, transform=ccrs.PlateCarree(),
                color='b', alpha=0.05)

    colorcycle = itertools.cycle(colors)
    for p in poles[:-1]:
        p.plot(ax, south_pole=True, color=colorcycle.next())

    ax.scatter(slon, slat, transform=ccrs.PlateCarree(), marker="*", s=100)

    for e in muller_apw.euler_poles:
        ax.scatter(e.longitude, e.latitude, color='r', transform=ccrs.PlateCarree(), marker="*", s=100)

    ages = np.linspace(0., 60., 100.)
    pathlons = np.empty_like(ages)
    pathlats = np.empty_like(ages)
    for i,a in enumerate(ages):
       pathlons[i],pathlats[i] = muller_apw(a)
    ax.plot(pathlons, pathlats, 'r', lw=3, transform=ccrs.PlateCarree())

    #plt.show()
    plt.savefig("australia_paths_" + str(n_euler_rotations)+".pdf")


def plot_age_samples():
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    colorcycle = itertools.cycle(colors)
    for p, age_samples in zip(poles[:-1], path.ages()[:-1]):
        c = colorcycle.next()
        age = np.linspace(0., 65, 1000)
        if p.age_type == 'gaussian':
            dist = st.norm.pdf(age, loc=p.age, scale=p.sigma_age)
        else:
            dist = st.uniform.pdf(age, loc=p.sigma_age[
                                  0], scale=p.sigma_age[1] - p.sigma_age[0])
        ax1.fill_between(age, 0, dist, color=c, alpha=0.6)
        ax2.hist(age_samples, color=c, normed=True, alpha=0.6)
    ax1.set_ylim(0., 1.)
    ax2.set_ylim(0., 1.)
    ax2.set_xlabel('Age (Ma)')
    ax1.set_ylabel('Prior probability')
    ax2.set_ylabel('Posterior probability')
    plt.tight_layout()
    #plt.show()
    plt.savefig("australia_ages_" + str(n_euler_rotations)+".pdf")


def plot_synthetic_poles():
    if proj_type == 'M':
        ax = plt.axes(projection=ccrs.Mollweide(proj_lon))
    elif proj_type == 'O':
        ax = plt.axes(projection = ccrs.Orthographic(proj_lon,proj_lat))

    ax.gridlines()
    ax.set_global()

    direction_samples = path.euler_directions()

    mcplates.plot.plot_continent(ax, 'australia', rotation_pole=mcplates.Pole(0., 90., 1.0), angle=-lon_shift, color='k')
    mcplates.plot.plot_continent(ax, 'antarctica', rotation_pole=mcplates.Pole(0., 90., 1.0), angle=-lon_shift, color='k')

    dist_colors = itertools.cycle([cmap_blue, cmap_red, cmap_green])
    for directions in direction_samples:
        mcplates.plot.plot_distribution(ax, directions[:, 0], directions[:, 1], cmap=dist_colors.next(), resolution=30)


    colorcycle = itertools.cycle(colors)
    lons, lats, ages = path.compute_synthetic_poles(n=100)
    for i in range(len(poles)-1):
        c = colorcycle.next()
        poles[i].plot(ax, south_pole=True, color=c)
        ax.scatter(lons[:, i]-180., -lats[:, i], color=c,
                   transform=ccrs.PlateCarree())

    ax.scatter(slon, slat, transform=ccrs.PlateCarree(), marker="*", s=100)

    for e in muller_apw.euler_poles:
        ax.scatter(e.longitude, e.latitude, color='r', transform=ccrs.PlateCarree(), marker="*", s=100)

    #plt.show()
    plt.savefig("australia_poles_" + str(n_euler_rotations)+".pdf")


def plot_plate_speeds():
    euler_directions = path.euler_directions()
    euler_rates = path.euler_rates()
    numbers = iter(['First', 'Second', 'Third', 'Fourth'])

    fig = plt.figure()
    i = 1
    for directions, rates in zip(euler_directions, euler_rates):
        speed_samples = np.empty_like(rates)
        for j in range(len(rates)):
            euler = mcplates.EulerPole(
                directions[j, 0], directions[j, 1], rates[j])
            speed_samples[j] = euler.speed_at_point(uluru)

        ax = fig.add_subplot(1, n_euler_rotations, i)
        ax.hist(speed_samples, bins=30, normed=True)
        if n_euler_rotations > 1:
            ax.set_title(numbers.next() + ' rotation')
        ax.set_xlabel('Plate speed (cm/yr)')
        ax.set_ylabel('Probability density')
        i += 1

    plt.tight_layout()
    #plt.show()
    plt.savefig("australia_speeds_" + str(n_euler_rotations)+".pdf")


def speed_time_plot():
    ages = np.linspace(0., 65., 1000.)
    speeds = np.array( [ muller_apw.speed(a) for a in ages] )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ages,speeds)
    plt.show()


def latitude_time_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ages = [p.age for p in poles]
    #100 is the n_segments returned by synthetic pole computation
    times = np.linspace( max(ages), min(ages), 100)
    pathlons, pathlats = path.compute_synthetic_paths(n=100)

    for pathlat in pathlats:
        ax.plot(times, -pathlat, 'b', alpha=0.03)

    #compute apw path from hotspot data
    hotspot_lons = np.empty_like(times)
    hotspot_lats = np.empty_like(times)
    for i,a in enumerate(times):
       hotspot_lons[i],hotspot_lats[i] = muller_apw(a)
    ax.plot(times, hotspot_lats, 'r', lw=3)

    colorcycle = itertools.cycle(colors)
    for p in poles[:-1]:
        ax.errorbar( p.age, -p.latitude, yerr= [p.angular_error,], \
                     xerr = [(p.sigma_age[1]-p.sigma_age[0])/2.,], \
                     color = colorcycle.next(), fmt='-')
    ax.set_xlabel("Age (Ma)")
    ax.set_ylabel("Latitude")

    #plt.show()
    plt.savefig("australia_latitude_" + str(n_euler_rotations)+".pdf")


if __name__ == "__main__":
    import os
    if os.path.isfile(path.dbname):
        print("Loading MCMC results from disk...")
        path.load_mcmc()
        print("Done.")
        #print("MAP logp: ", path.find_MAP())
    else:
        path.sample_mcmc(10000)
        #print("MAP logp: ", path.logp_at_max)
    plot_synthetic_paths()
    plot_age_samples()
    plot_synthetic_poles()
    plot_plate_speeds()
    latitude_time_plot()
