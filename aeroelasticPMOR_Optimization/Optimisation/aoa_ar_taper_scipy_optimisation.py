"""Script to optimise the FORCES surrogate for 3D: aoa, ar and taper with the
SciPy.Optimize package
Author: Pablo de Felipe
Date: 18/05/22"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import minimize
from scipy.optimize import Bounds

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

def equilibrium(x,*data):
    """ Function to evaluate equilibrium
    Args:
        aoa_deg (float): aoa_deg
        data            : Data for the problem
    Returns:
        residual: difference between lift and weight
        """
    w0, w1, lift_surr = data   # automatic unpacking
    aoa_deg = x[0]
    ar = x[1]
    taper = x[2]
    # Pack the data so it can be used by the surrogate
    xp = {'aoa_deg':[aoa_deg],'ar':[ar],'taper':[taper]}
    residual = lift_surr.eval_surrogate(xp)-9.8*weight(w0,w1,ar,taper)
    return residual
def objective(x,drag_surr):
    """ Objective function being drag"""
    xp = {'aoa_deg':[x[0]],'ar':[x[1]],'taper':[x[2]]}
    drag = drag_surr.eval_surrogate(xp)
    return drag

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

# Define constants
w0 = 4.25
w1 = 3*(2**0.5)
# perform the search
#bnds = Bounds([0,3.6],[0.2,1.0],[20,44])
bnds = ((0,3.6),(20,44),(0.2,1.0))
pt = [1.2,32.,1.0]
data = (w0,w1,lift_surr.surr)
cons = {'type':'eq','fun':equilibrium,'args':data}

result = minimize(objective, pt, args=(drag_surr.surr),method='SLSQP',
                  bounds=bnds,constraints=cons,options={'disp': True})

print(result)