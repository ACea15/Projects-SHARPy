"""
Created on Fri Feb  4 09:32:36 2022
Script to compare the spiral to the structured latin hypercube

@author: pablodfs
"""
import DoE 
import numpy as np


if  (__name__ == '__main__'):
    num_dim = 2
    N = np.array([4,9,16,25])
    for i in range(len(N)):
        points_dim = N[i]
        # get the points
        P0 = DoE.structured_hypercube(points_dim, num_dim) # 2D DoE
        P1 = DoE.spiral_recursive([0]*num_dim, N[i], np.array([[0]*num_dim])) 
        
        # Get the criteria
        crit_structured = DoE.distance_criterion(points_dim, P0)
        crit_spiral     = DoE.distance_criterion(points_dim, P1)
        # Print results
        print('N = ',N[i])
        print('Struct = ',crit_structured,'Spiral = ', crit_spiral)
    
    