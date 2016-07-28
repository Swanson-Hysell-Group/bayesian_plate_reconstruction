import matplotlib.pyplot as plt
import matplotlib.image as image
import cartopy.crs as ccrs
import numpy as np
import mcplates

imm = image.imread('mountain_small.png')
imv = image.imread('volcano_small.png')

ax = plt.axes(projection = ccrs.Orthographic(0.0, 30.))
ax.gridlines(xlocs=np.linspace(0.,360., 7), ylocs=np.linspace(-90., 90., 7.))
ax.set_global()

n_euler_poles=1
# Generate a synthetic data set
ages = [0., 10., 20., 30., 40., 50.,]
start_age = 50.
hidden_start_pole = [0., 90.]
hidden_euler_pole = [40., -20.]
hidden_euler_rate = 1.5

#Make a dummy APW path to create the synthetic data
dummy_pole_position_fn = mcplates.APWPath.generate_pole_position_fn( n_euler_poles, start_age)
pole_list = []
for a in ages:
    lon_lat = dummy_pole_position_fn(hidden_start_pole, a, hidden_euler_pole, hidden_euler_rate)
    pole_list.append( mcplates.PaleomagneticPole( lon_lat[0], lon_lat[1], angular_error = 5., age=a, sigma_age = 0.01))
for p in pole_list[1:-1]:
    p.plot(ax, color='b')

#Make a plate
times = np.linspace(min(ages), max(ages), 100.)
pb1_lon = np.empty_like(times)
pb1_lat = np.empty_like(times)
pb2_lon = np.empty_like(times)
pb2_lat = np.empty_like(times)
for i,t in enumerate(times):
    start1 = ( 30., 60.)
    start2 = ( 40., 15.)
    lon_lat = dummy_pole_position_fn(start1, t, hidden_euler_pole, hidden_euler_rate)
    pb1_lon[i] = lon_lat[0]
    pb1_lat[i] = lon_lat[1]
    lon_lat = dummy_pole_position_fn(start2, t, hidden_euler_pole, hidden_euler_rate)
    pb2_lon[i] = lon_lat[0]
    pb2_lat[i] = lon_lat[1]

p1 =  mcplates.rotations.spherical_to_cartesian(pb1_lon[-1], pb1_lat[-1], 1.0)
p2 =  mcplates.rotations.spherical_to_cartesian(pb2_lon[-1], pb2_lat[-1], 1.0)
p3 =  mcplates.rotations.spherical_to_cartesian(pb1_lon[0], pb1_lat[0], 1.0)
p4 =  mcplates.rotations.spherical_to_cartesian(pb2_lon[0], pb2_lat[0], 1.0)
#Spreading ridge
p5 =  mcplates.rotations.spherical_to_cartesian(pb1_lon[-4], pb1_lat[-4], 1.0)
p6 =  mcplates.rotations.spherical_to_cartesian(pb2_lon[-5], pb2_lat[-5], 1.0)

factor = np.linspace(0., 1., 100)
pb3_lon = np.empty_like(factor)
pb3_lat = np.empty_like(factor)
pb4_lon = np.empty_like(factor)
pb4_lat = np.empty_like(factor)
pb5_lon = np.empty_like(factor)
pb5_lat = np.empty_like(factor)
for i,f in enumerate(factor):
    lon, lat, r = mcplates.rotations.cartesian_to_spherical( p2 * f + (1.-f)*p1 )
    pb3_lon[i] = lon
    pb3_lat[i] = lat
    lon, lat, r = mcplates.rotations.cartesian_to_spherical( p3 * f + (1.-f)*p4 )
    pb4_lon[i] = lon
    pb4_lat[i] = lat
    lon, lat, r = mcplates.rotations.cartesian_to_spherical( p6 * f + (1.-f)*p5 )
    pb5_lon[i] = lon
    pb5_lat[i] = lat

ax.plot(pb1_lon, pb1_lat, lw=2, c='k', transform=ccrs.PlateCarree())
ax.plot(pb2_lon, pb2_lat, lw=2, c='k', transform=ccrs.PlateCarree())
ax.plot(pb3_lon, pb3_lat, lw=2, c='k', transform=ccrs.PlateCarree())
ax.plot(pb4_lon, pb4_lat, lw=2, c='k', transform=ccrs.PlateCarree())
ax.plot(pb5_lon, pb5_lat, lw=2, c='k', transform=ccrs.PlateCarree())

# Make volcanoes
for a in ages[1:-1]:
    lon_lat = dummy_pole_position_fn([35,45], a, hidden_euler_pole, hidden_euler_rate)
    if a==ages[-2]:
        ax.imshow(imv,extent = [lon_lat[0], lon_lat[0]+10, lon_lat[1], lon_lat[1]+10], origin="upper", transform=ccrs.PlateCarree())
    else:
        ax.imshow(imm,extent = [lon_lat[0], lon_lat[0]+10, lon_lat[1], lon_lat[1]+10], origin="upper", transform=ccrs.PlateCarree())

# Make plate motion arrow
times = np.linspace(ages[1], ages[-2])
arrow_lon = np.empty_like(times)
arrow_lat = np.empty_like(times)
for i,t in enumerate(times):
    lon_lat = dummy_pole_position_fn([40,25], t, hidden_euler_pole, hidden_euler_rate)
    arrow_lon[i] = lon_lat[0]
    arrow_lat[i] = lon_lat[1]
ax.plot(arrow_lon, arrow_lat, lw=4, c='k', transform=ccrs.PlateCarree())
ax.arrow(arrow_lon[0], arrow_lat[0], (arrow_lon[0]-arrow_lon[1])*1.e-10, (arrow_lat[0]-arrow_lat[1])*1.e-10, transform=ccrs.PlateCarree(), head_width=3.0, head_length=3.0, fc='k', ec='k', lw=3)


# Make subduction zone
n_arrows = 20
stride = int(len(pb4_lon)/n_arrows)
for i in range(1,n_arrows):
    ax.arrow(pb4_lon[i*stride], pb4_lat[i*stride], (pb4_lat[i*stride] - pb4_lat[i*stride+1])*1.e-10, (pb4_lon[i*stride+1]-pb4_lon[i*stride])*1.e-10, transform=ccrs.PlateCarree(), head_width=2.0, head_length=2.0, fc='w', ec='k', lw=2)

ax.scatter( hidden_euler_pole[0], hidden_euler_pole[1], c='b', marker='o', s=30, transform=ccrs.PlateCarree())

#plt.show()
plt.savefig("paleomagnetic_euler_pole.png", dpi=600)
plt.savefig("paleomagnetic_euler_pole.pdf")
