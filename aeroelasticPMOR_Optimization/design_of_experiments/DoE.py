import numpy as np
import matplotlib.pyplot as plt

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


if  (__name__ == '__main__'):
    N = 50 # 50 points 
    P = spiral_recursive([0, 0], N, np.array([[0, 0]])) # 2D DoE
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
    
