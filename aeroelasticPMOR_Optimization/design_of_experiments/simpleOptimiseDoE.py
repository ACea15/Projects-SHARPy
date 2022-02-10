""" 
 Code to optimise the design of experiments using a simple algorithm
 Date: 02/02/2022
 Author: Pablo de Felipe
"""

import DoE 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def func(alpha,*args):
    """
    Function to take in a single alpha parameter and output the criterion to minimize.


    Args:
        alpha (float) : float corresponding to a single value of alpha which is
        the amount by which a point is shifted.
        P0 (np.ndarray): Array containing the initial set of points to be placed
        index (int) : Integer value containing the point on which to operate on.
    Returns:
        distance_eval (float): Floating value of the sum of the reciprocal 

    """
    # Extract the constants
    P0,index, params = args
    # Get number of points and number of dimensions
    [points_dim,num_dim] = np.shape(P0)
    vectors = DoE.get_vectors(P0)
    P = DoE.update_points(num_dim, points_dim, params) #get starting points
    for k in range(num_dim):
        P[index,k] = P0[index,k]+alpha*vectors[index,k]
    # Evaluate criteria
    distance_eval = DoE.distance_criterion(points_dim, P)

    return distance_eval

if  (__name__ == '__main__'):       
    # Code copied from original DoE.py
    num_dim = 2
    N = 20# 50 points 
    P0 = DoE.spiral_recursive([0]*num_dim, N, np.array([[0]*num_dim])) # 2D DoE
    #P = spiral_recursive([0, 0, 0, 0], N, np.array([[0, 0, 0, 0]])) # 4D DoE etc.
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
   
   
    plt.plot(P0[:,0], P0[:,1], 'r--', linewidth=0.5)
     
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
    
    alpha = np.zeros((N-1,))  
    # Select the point to optimise
    # Coding the optimiser here
    b= [(0.0,0.99)]
    bnds = b
    x0 = 0.8
    index = 0
        
    res = minimize(func, x0, args=(P0, index, alpha), method='nelder-mead', bounds=bnds,
             options={'maxiter': N*400,'xatol': 1e-8,'disp': True})
    print(res)
    initial_criteria = DoE.distance_criterion(N, P0)
    # Update the points
    
    alpha[index] = res.x
    P1 = DoE.update_points(num_dim, N, alpha)
    final_criteria = DoE.distance_criterion(N, P1)
   
    # Compare the criteria evaluation
    print('Index = ',index)
    print('Before Optimisation Crit = ', initial_criteria)
    print('After Optimisation Crit. = ', final_criteria) 
    # print the criteria for all N
        
    
    plt.scatter(P1[:,0], P1[:,1], marker='.', c='b', s=100) 
  
    print('alpha = ', alpha)