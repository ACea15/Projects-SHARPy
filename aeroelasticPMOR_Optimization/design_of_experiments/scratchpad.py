""" 
 scratchpad
"""

import DoE 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

num_dim = 2

# print the criteria for all N
#N = np.array((4,10,20,50))
N = 4
i = 0
# Get recursive criterion
P = DoE.spiral_recursive([0, 0], N, np.array([[0, 0]])) 
recursive_crit = DoE.distance_criterion(N, P)
# Get initial criterion
# alpha = np.zeros((N[i]-1,))*0.5
alpha = np.zeros((N-1,))
alpha[0] = 0.5
alpha[1] = 0.5
P1 = DoE.update_points(num_dim, N, alpha)
initial_crit = DoE.distance_criterion(N, P1)
print('N = ',N,' recursive = ', recursive_crit,' initial = ',initial_crit)
    
# Test the vector function
vector = DoE.get_vectors(P)
print(vector[2,:])
