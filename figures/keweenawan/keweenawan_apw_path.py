from __future__ import print_function
import itertools
import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys

import mcplates
from pymc.utils import hpd

plt.style.use('../bpr.mplstyle')
from mcplates.plot import cmap_red, cmap_green, cmap_blue

# Shift all longitudes by 180 degrees to get around some plotting
# issues. This is error prone, so it should be fixed eventually
lon_shift = 180.

# List of colors to use
colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
          '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
#colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']
dist_colors_short = ['darkblue', 'darkred', 'darkgreen']

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

print("Fitting Keweenawan APW track with "+str(n_euler_rotations)+" Euler rotation" + ("" if n_euler_rotations == 1 else "s") )


data = pd.read_csv("pole_means.csv")
# Give unnamed column an appropriate name
data.rename(columns={'Unnamed: 0': 'Name',\
                     'Unnamed: 14': 'GaussianOrUniform'},\
            inplace=True)
data = data[data.Name != 'Osler_N'] #Huge error, does not contribute much to the model
data = data[data.PoleName != 'Abitibi'] # Standstill at the beginning, not realistic to fit
data = data[data.PoleName != 'Haliburton'] #Much younger, far away pole, difficutlt to fit
data.sort_values('AgeNominal', ascending=False, inplace=True)

poles = []
for i, row in data.iterrows():
    pole_lat = row['PLat']
    pole_lon = row['PLon'] - lon_shift
    a95 = row['A95']
    age = row['AgeNominal']

    if row['GaussianOrUniform'] == 'gaussian':
        sigma_age = row['gaussian_2sigma'] / 2.
    elif row['GaussianOrUniform'] == 'uniform':
        sigma_age = (row['AgeLower'], row['AgeUpper'])
    else:
        raise Exception("Unrecognized age error type")

    pole = mcplates.PaleomagneticPole(
        pole_lon, pole_lat, angular_error=a95, age=age, sigma_age=sigma_age)
    poles.append(pole)

slat = 46.8  # Duluth lat
slon = 360. - 92.1 - lon_shift  # Duluth lon
duluth = mcplates.PlateCentroid(slon, slat)

path = mcplates.APWPath(
    'keweenawan_apw_' + str(n_euler_rotations), poles, n_euler_rotations)
path.create_model(site_lon_lat=(slon, slat), watson_concentration=0.0)


def plot_synthetic_paths():

    if proj_type == 'M':
        ax = plt.axes(projection=ccrs.Mollweide(proj_lon))
    elif proj_type == 'O':
        ax = plt.axes(projection = ccrs.Orthographic(proj_lon,proj_lat))

    ax.gridlines()
    ax.set_global()

    direction_samples = path.euler_directions()

    dist_colors = itertools.cycle([cmap_blue, cmap_red, cmap_green])
    for directions in direction_samples:
        mcplates.plot.plot_distribution(ax, directions[:, 0], directions[:, 1], cmap=dist_colors.next(), resolution=60)

    mcplates.plot.plot_continent(ax, 'laurentia', rotation_pole=mcplates.Pole(0., 90., 1.0), angle=-lon_shift, color='k')

    pathlons, pathlats = path.compute_synthetic_paths(n=200)
    for pathlon, pathlat in zip(pathlons, pathlats):
        ax.plot(pathlon, pathlat, transform=ccrs.PlateCarree(),
                color='b', alpha=0.05)

    colorcycle = itertools.cycle(colors)
    for p in poles:
        p.plot(ax, color=colorcycle.next())

    ax.scatter(slon, slat, transform=ccrs.PlateCarree(), c='k', marker="*", s=100)
    #plt.show()
    plt.savefig("keweenawan_paths_" + str(n_euler_rotations)+".pdf")


def plot_age_samples():
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    colorcycle = itertools.cycle(colors)
    for p, age_samples in zip(poles, path.ages()):
        c = colorcycle.next()
        age = np.linspace(1070, 1115, 1000)
        if p.age_type == 'gaussian':
            dist = st.norm.pdf(age, loc=p.age, scale=p.sigma_age)
        else:
            dist = st.uniform.pdf(age, loc=p.sigma_age[
                                  0], scale=p.sigma_age[1] - p.sigma_age[0])
        ax1.fill_between(age, 0, dist, color=c, alpha=0.6)
        ax2.hist(age_samples, color=c, normed=True, alpha=0.6)
    ax1.set_ylim(0., 1.)
    ax2.set_ylim(0., 1.)
    ax1.set_xlim(1070, 1115)
    ax2.set_xlim(1070, 1115)
    ax2.set_xlabel('Age (Ma)')
    ax1.set_ylabel('Prior probability')
    ax2.set_ylabel('Posterior probability')
    plt.tight_layout()
    plt.savefig("keweenawan_ages_" + str(n_euler_rotations)+".pdf")


def plot_synthetic_poles():
    if proj_type == 'M':
        ax = plt.axes(projection=ccrs.Mollweide(proj_lon))
    elif proj_type == 'O':
        ax = plt.axes(projection = ccrs.Orthographic(proj_lon,proj_lat))

    ax.gridlines()
    ax.set_global()

    direction_samples = path.euler_directions()

    dist_colors = itertools.cycle([cmap_blue, cmap_red, cmap_green])
    for directions in direction_samples:
        mcplates.plot.plot_distribution(ax, directions[:, 0], directions[:, 1], cmap=dist_colors.next(), resolution=60)

    mcplates.plot.plot_continent(ax, 'laurentia', rotation_pole=mcplates.Pole(0., 90., 1.0), angle=-lon_shift, color='k')

    colorcycle = itertools.cycle(colors)
    lons, lats, ages = path.compute_synthetic_poles(n=100)
    for i in range(len(poles)):
        c = colorcycle.next()
        poles[i].plot(ax, color=c)
        ax.scatter(lons[:, i], lats[:, i], color=c,
                   transform=ccrs.PlateCarree())

    ax.scatter(slon, slat, transform=ccrs.PlateCarree(), marker="*", c='k', s=100)
    #plt.show()
    plt.savefig("keweenawan_poles_" + str(n_euler_rotations)+".pdf")


def plot_plate_speeds():
    euler_directions = path.euler_directions()
    euler_rates = path.euler_rates()

    # Get a list of intervals for the rotations
    if n_euler_rotations > 1:
        changepoints = [ np.median(c) for c in path.changepoints() ]
    else:
        changepoints = []
    age_list = [p.age for p in poles]
    changepoints.insert( 0, max(age_list) )
    changepoints.append( min(age_list) )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Plate speed (cm/yr)')
    ax.set_ylabel('Probability density')

    xmin = 1000.
    xmax = 0.
    colorcycle = itertools.cycle( dist_colors_short )
    for i, (directions, rates) in enumerate(zip(euler_directions, euler_rates)):

        #comptute plate speeds
        speed_samples = np.empty_like(rates)
        for j in range(len(rates)):
            euler = mcplates.EulerPole(
                directions[j, 0], directions[j, 1], rates[j])
            speed_samples[j] = euler.speed_at_point(duluth)

        c = next(colorcycle)

        #plot histogram
        ax.hist(speed_samples, bins=30, normed=True, alpha=0.5, color=c, label='%i - %i Ma'%(changepoints[i], changepoints[i+1]))

        # plot median, credible interval
        credible_interval = hpd(speed_samples, 0.05)
        median = np.median(speed_samples)
        ax.axvline( median, lw=2, color=c )
        ax.axvline( credible_interval[0], lw=2, color=c, linestyle='dashed')
        ax.axvline( credible_interval[1], lw=2, color=c, linestyle='dashed')

        xmin = max(0., min( xmin, median - 2.*(median-credible_interval[0])))
        xmax = max( xmax, median + 2.*(credible_interval[1]-median))

    if n_euler_rotations > 1:
        ax.legend(loc='upper right')
    ax.set_xlim(xmin, xmax)

    #plt.show()
    plt.savefig("keweenawan_speeds_" + str(n_euler_rotations)+".pdf")


if __name__ == "__main__":
    import os
    if os.path.isfile(path.dbname):
        path.load_mcmc()
    else:
        path.sample_mcmc(1000000)
    plot_synthetic_paths()
    plot_age_samples()
    plot_synthetic_poles()
    plot_plate_speeds()
