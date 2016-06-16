import itertools
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

import cartopy.crs as ccrs

import pymc
import mcplates

plt.style.use('bpr.mplstyle')
from mcplates.plot import cmap_red, cmap_green

dbname = 'one_euler_pole'
n_euler_poles=1

# Generate a synthetic data set
ages = [0., 10., 20., 30.]
start_age = 0.
hidden_start_pole = [-30., 0.]
hidden_euler_pole = [0., 0.]
hidden_euler_rate = 180./30.

#Make a dummy APW path to create the synthetic data
dummy_pole_position_fn = mcplates.APWPath.generate_pole_position_fn( n_euler_poles, start_age)
pole_list = []
for a in ages:
    lon_lat = dummy_pole_position_fn(hidden_start_pole, a, hidden_euler_pole, hidden_euler_rate)
    pole_list.append( mcplates.PaleomagneticPole( lon_lat[0], lon_lat[1], angular_error = 10., age=a, sigma_age = 0.01))

path = mcplates.APWPath( dbname, pole_list, n_euler_poles )
path.create_model()


def plot_result():

    fig = plt.figure( figsize=(8,4) )
    ax = fig.add_subplot(1,2,1, projection = ccrs.Orthographic(0.,15.))
    ax.gridlines()
    ax.set_global()

    colors = itertools.cycle([cmap_red, cmap_green])
    direction_samples = path.euler_directions()
    for directions in direction_samples:
        mcplates.plot.plot_distribution( ax, directions[:,0], directions[:,1], resolution=60, cmap=colors.next())

    pathlons, pathlats = path.compute_synthetic_paths(n=200)
    for pathlon,pathlat in zip(pathlons,pathlats):
        ax.plot(pathlon,pathlat, transform=ccrs.PlateCarree(), color='darkred', alpha=0.05 )

    for p in pole_list:
        p.plot(ax)
    ax.set_title('(a)')

    ax = fig.add_subplot(1,2,2)
    rate_samples = path.euler_rates()
    ax.hist(rate_samples, bins=15, normed=True, edgecolor='none', color='darkred', alpha=0.5)

    ax.set_title('(b)')
    ax.set_xlabel(r'Rotation rate $\,^\circ / \mathrm{Myr}$')
    ax.set_ylabel(r'Posterior probability density')
    plt.tight_layout()
    plt.savefig("one_euler_pole.pdf")
    #plt.show()

if __name__ == "__main__":
    import os 
    if os.path.isfile(dbname+'.pickle'):
        path.load_mcmc()
    else:
        path.sample_mcmc(30000)
    plot_result()
