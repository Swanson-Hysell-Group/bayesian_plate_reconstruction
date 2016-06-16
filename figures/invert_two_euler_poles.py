import itertools
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

import cartopy.crs as ccrs

import pymc
import mcplates

plt.style.use('bpr.mplstyle')
from mcplates.plot import cmap_red, cmap_green

dbname = 'two_euler_poles'
n_euler_poles=2

# Generate a synthetic data set
ages = [40., 30., 20., 10., 0.]
start_age = 40.
hidden_start_pole = [-60., 0.]
hidden_euler_poles = [ [-60., 41.], [60., 41.] ]
hidden_euler_rates = [130./20., 130./20.]
hidden_changepoint = 20.
#Make a dummy APW path to create the synthetic data
dummy_pole_position_fn = mcplates.APWPath.generate_pole_position_fn( n_euler_poles, start_age)
pole_list = []
for a in ages:
    lon_lat = dummy_pole_position_fn(hidden_start_pole, a,
                                     hidden_euler_poles[0], hidden_euler_poles[1],
                                     hidden_euler_rates[0], hidden_euler_rates[1], hidden_changepoint)
    pole_list.append( mcplates.PaleomagneticPole( lon_lat[0], lon_lat[1], angular_error = 10., age=a, sigma_age = 0.01))

path = mcplates.APWPath( dbname, pole_list, n_euler_poles )
path.create_model()


def plot_result():

    fig = plt.figure( figsize=(8,4) )
    ax = fig.add_subplot(1,2,1, projection = ccrs.Orthographic(0.,30.))
    #ax = fig.add_subplot(1,2,1, projection = ccrs.Mollweide(0.))
    ax.gridlines()
    ax.set_global()

    colors = itertools.cycle([cmap_red, cmap_green])
    direction_samples = path.euler_directions()
    for directions in direction_samples:
        mcplates.plot.plot_distribution( ax, directions[:,0], directions[:,1], resolution=60, cmap=colors.next())

    n_paths=100
    interval = max(1, int(len(path.mcmc.db.trace('rate_0')[:]) / n_paths))
    pathlons, pathlats = path.compute_synthetic_paths(n=n_paths)
    changepoints = path.changepoints()[0][::interval]
    for pathlon,pathlat,change in zip(pathlons,pathlats,changepoints):
        switch = int(float(len(pathlon))*change/(max(ages)-min(ages)))
        ax.plot(pathlon[:switch],pathlat[:switch], transform=ccrs.PlateCarree(), color='darkred', alpha=0.05 )
        ax.plot(pathlon[switch:],pathlat[switch:], transform=ccrs.PlateCarree(), color='darkgreen', alpha=0.05 )

    for p in pole_list:
        p.plot(ax)
    ax.set_title('(a)')

    ax = fig.add_subplot(1,2,2)
    rate_samples = path.euler_rates()
    ax.hist(rate_samples[0], bins=15, normed=True, edgecolor='none', color='darkred', alpha=0.5)
    ax.hist(rate_samples[1], bins=15, normed=True, edgecolor='none', color='darkgreen', alpha=0.5)

    ax.set_title('(b)')
    ax.set_xlabel(r'Rotation rate $\,^\circ / \mathrm{Myr}$')
    ax.set_ylabel(r'Posterior probability density')
    plt.tight_layout()
    plt.savefig("two_euler_poles.pdf")
    #plt.show()

if __name__ == "__main__":
    import os 
    if os.path.isfile(dbname+'.pickle'):
        path.load_mcmc()
    else:
        path.sample_mcmc(100000)
    plot_result()
