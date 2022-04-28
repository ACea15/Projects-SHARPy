"""Code to test the linear-regression module
Imports required:
"""
import os
os.chdir('/home/pablodfs/FYP/Projects-SHARPy/aeroelasticPMOR_Optimization/surrogate_model/')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import linear_regression as lr
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import Rbf

# Open pandas data
cwd = os.getcwd()
os.chdir(cwd+'/ar_flutter/')
data_pandas = pd.read_csv('ar_flutter.csv')
data_numpy = data_pandas.to_numpy()
# Store data
AR = data_numpy[:,1]
u_flutter = data_numpy[:,2]

# Now implement a second order polynomial regression fit
k = 2
points_train = {
    'x' : AR[[0,2,4,6]],
    'y' : u_flutter[[0,2,4,6]]
}
points_test = {
    'x': AR[[1,3,5]],
    'y' : u_flutter[[1,3,5]]
}

surr2 = lr.Polynomial(k,points_train,points_test)
surr2.build()

xp = np.linspace(20,44,100)
yp = surr2.eval_surrogate(xp)
# Plot results
fig = plt.figure()
ax  = fig.add_subplot(1,1,1)
plt.plot(AR,u_flutter,'x')
plt.plot(xp,yp,'-b')

ax.grid(True)
ax.set_ylabel('U_flutter (m/s)')
ax.set_xlabel('AR')
plt.show()
