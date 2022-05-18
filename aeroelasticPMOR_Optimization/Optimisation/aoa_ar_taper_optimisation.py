""" Optimisation of aoa, ar and taper using only the forces surrogate
Author: Pablo de Felipe
Date: 18/05/22"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import Rbf
from scipy.optimize import fsolve
import pickle

###############START FUNCTION DEFFINITIONS#####################################
def weight(w0,w1,ar,taper):
    """ Function to calculate the weight of the aircraft using Raymer methdos
    Args:
        w0 (float)      : Weight of aircraft excluding wing
        w1 (floar)      : Constant to control wing weight
        AR (float)      : aspect ratio
        taper (float)   : taper ratio
    Returns:
        weight (flooat) : Aircraft weight in kg!
        """
    weight = w0+w1*np.sqrt(ar)*(taper)**0.05
    return weight

def equilibrium(aoa_deg,*data):
    """ Function to evaluate equilibrium
    Args:
        aoa_deg (float): aoa_deg
        data            : Data for the problem
    Returns:
        residual: difference between lift and weight
        """
    ar, taper, w0, w1, lift_surr = data   # automatic unpacking
    # Pack the data so it can be used by the surrogate
    xp = {'aoa_deg':aoa_deg,'ar':ar,'taper':taper}
    residual = lift_surr.eval_surrogate(xp)-9.8*weight(w0,w1,ar,taper)
    return residual

#########################END FUNCTION DEFFINITIONS#############################


# Get the surrogates from pickle data
surrogate_dir = '/home/pablodfs/FYP/Projects-SHARPy/aeroelasticPMOR_Optimization/surrogate_model/surrogate_built'
optimisation_dir = '/home/pablodfs/FYP/Projects-SHARPy/aeroelasticPMOR_Optimization/optimisation'

# Lift surrogate
liftsurr_filepath = surrogate_dir+'/lift_aoa_ar_taper_surr_v1.pickle'
dragsurr_filepath = surrogate_dir+'/drag_aoa_ar_taper_surr_v1.pickle'

# open a file, where you stored the pickled data
file = open(liftsurr_filepath, 'rb')
lift_surr = pickle.load(file)
# close the file
file.close()

# Drag surrogate
file = open(dragsurr_filepath, 'rb')
drag_surr = pickle.load(file)
# close the file
file.close()

# Discretise the parameter space
ar_values = np.linspace(20,44,100)
taper_values = np.linspace(0.2,1.0,100)

# Constants for the problem
w0 = 4.25    # Aircraft weigth excluding wings
w1 = 3*np.sqrt(2) # Constant for the wing
u_inf = 10.  # m/s
Sref = 32.   # m^2
rho = 1.225  # kg/m^3
dCldAlpha = 2*np.pi

# Find the aoa angles which satisfy
aoa_deg = np.zeros([len(ar_values),len(taper_values)])
for i in range(len(ar_values)):
    for j in range(len(taper_values)):
        data = (ar_values[i],taper_values[j],w0,w1,lift_surr.surr)
        aoa_deg0 = (weight(w0, w1, ar_values[i],taper_values[j]) * 9.8 /
                    (0.5 * rho * u_inf ** 2 * Sref * dCldAlpha)) * 180/np.pi
        aoa_deg[i,j] = fsolve(equilibrium,aoa_deg0,args=data)

print(aoa_deg)
# Build 2D matrices for ar and taper
ar = np.zeros([len(ar_values),len(taper_values)])
taper = np.zeros([len(ar_values),len(taper_values)])

for i in range(len(ar_values)):
    ar[i,:] = ar_values[i]
for j in range(len(taper_values)):
    taper[:,j] = taper_values[j]
# Creating color map
my_cmap = plt.get_cmap('viridis')

# Creating figure
fig = plt.figure(figsize=(14, 9))
ax = plt.axes(projection='3d')

# Creating plot
surf = ax.plot_surface(ar, taper, aoa_deg,
                       cmap = my_cmap,
                       edgecolor='none')
fig.colorbar(surf, ax = ax,
             shrink = 0.5, aspect = 5)
ax.set_xlabel('ar')
ax.set_ylabel('taper')
ax.set_zlabel('aoa_deg')
# show plot
plt.show()

# Calculate drag
drag = np.zeros(ar.shape)
for i in range(len(ar_values)):
    for j in range(len(taper_values)):
        xp = {'aoa_deg':[aoa_deg[i,j]],'ar':[ar[i,j]],'taper':[taper[i,j]]}
        drag[i,j] = drag_surr.surr.eval_surrogate(xp)

print(drag)
# Find the minimum of drag
min_drag = np.amin(drag)
min_drag_i=np.where(drag == min_drag)
min_drag_i = np.array(min_drag_i)
min_drag_i = min_drag_i.flatten()
print(min_drag_i)
# Creating figure
fig = plt.figure(figsize=(14, 9))
ax = plt.axes(projection='3d')

# Creating plot
surf = ax.plot_surface(ar, taper, drag,
                       cmap = my_cmap,
                       edgecolor='k')
ax.scatter(ar[min_drag_i[0],min_drag_i[1]],
           taper[min_drag_i[0],min_drag_i[1]],
           min_drag,
           c='r',marker='o',s=50)
fig.colorbar(surf, ax = ax,
             shrink = 0.5, aspect = 5)
cset = ax.contourf(ar, taper, drag,
                   zdir ='z',
                   offset = np.min(drag),
                   cmap = my_cmap)
cset = ax.contourf(ar, taper, drag,
                   zdir ='x',
                   offset =-5,
                   cmap = my_cmap)
cset = ax.contourf(ar, taper, drag,
                   zdir ='y',
                   offset = 5,
                   cmap = my_cmap)

ax.set_xlabel('ar')
ax.set_ylabel('taper')
ax.set_zlabel('drag')
# show plot
plt.show()
print(ar[min_drag_i[0],min_drag_i[1]])

# aoa_deg_values = np.linspace(0,3.6,7)
# xp = np.linspace(0,3.6,100)
# yp0 = ar_values[0]*np.ones(xp.shape)
# yp1 = ar_values[1]*np.ones(xp.shape)
# yp2 = ar_values[2]*np.ones(xp.shape)
# yp3 = ar_values[3]*np.ones(xp.shape)
# yp4 = ar_values[4]*np.ones(xp.shape)
# yp5 = ar_values[5]*np.ones(xp.shape)
# yp6 = ar_values[6]*np.ones(xp.shape)
#
# zp0 = taper_values[0]*np.ones(xp.shape)
# zp1 = taper_values[1]*np.ones(xp.shape)
# zp2 = taper_values[2]*np.ones(xp.shape)
# zp3 = taper_values[3]*np.ones(xp.shape)
# zp4 = taper_values[4]*np.ones(xp.shape)
# zp5 = taper_values[5]*np.ones(xp.shape)
# zp6 = taper_values[6]*np.ones(xp.shape)
#
# xp0 = {'aoa_deg':xp,'ar':yp0,'taper':zp6}
# xp1 = {'aoa_deg':xp,'ar':yp1,'taper':zp6}
# xp2 = {'aoa_deg':xp,'ar':yp2,'taper':zp6}
# xp3 = {'aoa_deg':xp,'ar':yp3,'taper':zp6}
# xp4 = {'aoa_deg':xp,'ar':yp4,'taper':zp6}
# xp5 = {'aoa_deg':xp,'ar':yp5,'taper':zp6}
# xp6 = {'aoa_deg':xp,'ar':yp6,'taper':zp6}
#
# fig, ax = plt.subplots()
#
# ax.plot(xp,drag_surr.surr.eval_surrogate(xp0),'b-')
# ax.plot(xp,drag_surr.surr.eval_surrogate(xp2),'r-')
# ax.plot(xp,drag_surr.surr.eval_surrogate(xp4),'g-')
# ax.plot(xp,drag_surr.surr.eval_surrogate(xp6),'k-')
# ax.legend(['AR=20','AR=28','AR=36','AR=44'])
# ax.grid(True)
# ax.set_ylabel('Drag (N)')
# ax.set_xlabel('AoA (deg)')
#
# plt.show()
