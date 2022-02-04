"""
Created on Fri Feb  4 09:19:20 2022
Script to test the structured latin hypercube
@author: pablodfs
"""

import DoE 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

if  (__name__ == '__main__'):       
    num_dim = 2
    N = 25
    P = DoE.structured_hypercube(N, num_dim) # 2D DoE
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
   
   
    plt.scatter(P[:,0], P[:,1], marker='.', c='b', s=100)     
    major_ticks = np.arange(0, N+1, 2)
    minor_ticks = np.arange(0, N+1, N-1)
    
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_xlim([-0.1,N])
    ax.set_ylim([-0.1,N])
    # And a corresponding grid
    ax.grid(which='both')
    
    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=1)
    ax.grid(which='major', alpha=0.2)
    
    plt.show()