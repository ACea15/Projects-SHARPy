""" 
 Code to optimise the design of experiments 
 Date: 18/01/2022
 Author: Pablo de Felipe
"""
import DoE 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def func(alpha):
    """
    Function to take in alpha parameters and output the criterion to minimize.
    Assumes 2D, will need to figure out a way to have this as a parameter.

    Args:
        alpha (array) :narray with p-1 (where p is the number of points) 
        entries corresponding to the shift of each array

    Returns:
        distance_eval (float): Floating value of the sum of the reciprocal 

    """
    num_dim = 2
    points_dim = len(alpha)+1
    # Get initial points, should be able to pass them (Could try global)
    P0 = DoE.spiral_recursive([0,0], points_dim, np.array([[0, 0]]))
    P  = P0
    vectors = DoE.get_vectors(points_dim, P0)
    for i in range(points_dim-1):
        for k in range(num_dim):
            P[i,k] = P0[i,k]+alpha[i]*vectors[i,k]

    # Evaluate criteria
    distance_eval = DoE.distance_criterion(points_dim, P)

    return distance_eval
        
if  (__name__ == '__main__'):       
    # Code copied from original DoE.py
    num_dim = 2
    N = 20 # 50 points 
    P = DoE.spiral_recursive([0, 0], N, np.array([[0, 0]])) # 2D DoE
    #P = spiral_recursive([0, 0, 0, 0], N, np.array([[0, 0, 0, 0]])) # 4D DoE etc.
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
   
   
    plt.plot(P[:,0], P[:,1], 'r--', linewidth=0.5)
     
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
    
   
 

    # Coding the optimiser here
    b= (0,0.99)
    bnds = ((b, )*(N-1))
    x0 = np.ones((1,N-1))*0.5
        
    res = minimize(func, x0, method='nelder-mead', bounds=bnds,
             options={'maxiter': N*400,'xatol': 1e-8,'disp': True})
    print(res)
    initial_criteria = DoE.distance_criterion(N, P)
    # Update the points
    P1 = DoE.update_points(num_dim, N, res.x)
    final_criteria = DoE.distance_criterion(N, P1)
   
    # Compare the criteria evaluation
    print('Before Optimisation Crit = ', initial_criteria)
    print('After Optimisation Crit. = ', final_criteria) 
    # print the criteria for all N
    N = np.array((4,10,20,50))
    for i in range(len(N)):
        # Get recursive criterion
        P = DoE.spiral_recursive([0, 0], N[i], np.array([[0, 0]])) 
        recursive_crit = DoE.distance_criterion(N[i], P)
        # Get initial criterion
        alpha = np.zeros((N[i]-1,))*0.5
        P1 = DoE.update_points(num_dim, N[i], alpha)
        initial_crit = DoE.distance_criterion(N[i], P1)
        print('N = ',N[i],' recursive = ', recursive_crit,' initial = ',initial_crit)
        
    
    plt.scatter(P1[:,0], P1[:,1], marker='.', c='b', s=100) 
    plt.show()
    
    