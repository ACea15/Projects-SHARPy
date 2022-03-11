# Script to plot a full factorial

import numpy as np
import matplotlib.pyplot as plt

N = 25
total_num = N*N
points = np.zeros((total_num,2))

point_counter = 0
for i in range(N):
    for j in range(N):
        points[point_counter,0] = j
        points[point_counter,1] = i
        point_counter += 1


# Plot the solution
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

plt.scatter(points[:, 0], points[:, 1], marker='.', c='b', s=100)
major_ticks = np.arange(0, N + 1, 2)
minor_ticks = np.arange(0, N + 1, N - 1)

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.set_xlim([-0.1, N])
ax.set_ylim([-0.1, N])
# And a corresponding grid
ax.grid(which='both')

# Or if you want different settings for the grids:
ax.grid(which='minor', alpha=1)
ax.grid(which='major', alpha=0.2)

plt.show()