import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def spiral_recursive(x0, points_dim, points_list=[]):
    """
    Creates an spiral Latin HyperCube designs of experiments 
    by recursively calling itself

    Args:
        x0 (np.ndarray): initial point coordinates
        points_dim (int): total number of points in the DoE
        points_list (list): initial point in the list of points, to be left
        at [0 for i in range(num_dim)]

    Returns:
        points_list (np.ndarray): DoE of shape(points_dim, num_dimenstions)
    """ 
    num_dim = len(x0)

    cycle_points1 = [i for i in range(1,num_dim+1)]
    cycle_points2 = [i for i in (points_dim-1) - np.arange(num_dim)]
    num_cycle_points = 2*num_dim
    points = np.zeros((num_cycle_points, num_dim))
    for di in range(num_dim):
        points[:,di] = cycle_points1[0:di] + cycle_points2 + cycle_points1[di:]
        points[:,di] += x0[di]
    if points_dim-2*num_dim-1 <= 0:
        points = points[:points_dim-1]
        points_list = np.concatenate((points_list, points))
    else:
        points_list = np.concatenate((points_list, points))
        points_list = spiral_recursive(points[-1], points_dim-2*num_dim, points_list)

    return points_list

def distance_criterion(points_dim, points_list):
    """
    Evaluates the distance criterion for a set of points

    Args:
        points_dim (int): number of dimensions of the problem
        points_list (list): initial point in the list of points, to be left
        at [0 for i in range(num_dim)]
        
    
    Returns:
        distance_eval (float): Floating value of the sum of the reciprocal 
        distance between a pair of points. This is the value that has to be 
        minimised in the optimisation problem.
    """
    d = 0.0 # Predefine a distance
    distance_eval = 0.0 # Predefine the distance criterion
    num_dim = points_list.shape[1]
    for i in range(points_dim):
        for j in range(i+1,points_dim):
            for k in range(num_dim):
                d += (points_list[j,k]-points_list[i,k])**2
                
            distance_eval += 1/d**0.5
            d = 0
        
    return distance_eval

def get_vectors(points_dim, points_list):
    """
    Evaluates a normalized vector to follow in optimisation

    Args:
        points_dim (int): number of dimensions of the problem
        points_list (list): initial point in the list of points, to be left
        at [0 for i in range(num_dim)]
        
    
    Returns:
        vectors (np.narray): Array with normalised components.
    """
    num_dim = points_list.shape[1]
    vectors = np.zeros((points_dim-1,num_dim))
    for i in range(points_dim-1):
        for k in range(num_dim):
            vectors[i,k] = (points_list[i+1,k]-points_list[i,k])            
    return vectors

def update_points(num_dim,points_dim,alpha):
    """
    Updates the points from the starting recursive DoE

    Args:
        num_dim (int): Number of dimensions 
        points_dim (int): Number of points
        alpha (nparray): array of the sliders for each of the lines

    Returns:
        P (nparray): Updated points
    """
    
    # Get initial points, should be able to pass them (Could try global)
    P0 = spiral_recursive([0,0], points_dim, np.array([[0, 0]]))
    P  = P0
    vectors = get_vectors(points_dim, P0)
    for i in range(points_dim-1):
        for k in range(num_dim):
            P[i,k] = P0[i,k]+alpha[i]*vectors[i,k]

    return P

if  (__name__ == '__main__'):
    N = 20 # 50 points 
    P = spiral_recursive([0, 0, 0], N, np.array([[0, 0, 0]])) # 2D DoE
    #P = spiral_recursive([0, 0, 0, 0], N, np.array([[0, 0, 0, 0]])) # 4D DoE etc.

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(P[:,0], P[:,1], marker='.', c='b', s=100)

    plt.plot(P[:,0], P[:,1], 'r--', linewidth=0.5)
 
    major_ticks = np.arange(0, N+1, 1)
    minor_ticks = np.arange(0, N+1, N-1)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=1)
    ax.grid(which='major', alpha=0.2)

    plt.show()
    
    criteria = distance_criterion(4, P)
    vectors  = get_vectors(N, P)
    
    # Only plot in 3D if there are 3 dimensions
    if P.shape[1]==3:
        fig2 = plt.figure()
        ax = plt.axes(projection='3d')
        
        ax.scatter3D(P[:,0], P[:,1], P[:,2], marker='.', c='b', s=100)
        
        ax.plot3D(P[:,0], P[:,1], P[:,2],'r--', linewidth=0.5)
        
        ax.show()
    
        
        