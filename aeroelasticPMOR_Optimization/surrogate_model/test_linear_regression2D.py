"""Code to test the 2D linear-regression module features
Imports required:
"""
import os
os.chdir('/home/pablodfs/FYP/Projects-SHARPy/aeroelasticPMOR_Optimization/surrogate_model/')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import linear_regression as lr

foldername = 'aoa_ar_forces'
cwd = os.getcwd()
os.chdir(cwd+'/'+foldername+'/')
data_pandas = pd.read_csv(foldername+'.csv')
# Go back to the initial folder
os.chdir(cwd)
# Get data from pandas
data_numpy = data_pandas.to_numpy()
AoA_deg = data_numpy[:,2]
AR      = data_numpy[:,3]
lift    = data_numpy[:,4]
drag    = data_numpy[:,5]

num_models = len(data_numpy[:,1])
i_train = np.arange(0,num_models+1,2)
i_test = np.arange(1,num_models,2)
x1 = AoA_deg[i_train]
x1_test = AoA_deg[i_test]
x2 = AR[i_train]
x2_test = AR[i_test]
lift_test = lift[i_test]

# Define the parameters required for the surrogate
points_train = {'x':{'aoa_deg':x1,
                     'ar': x2},
                'y':lift[i_train]
                }
points_test = {'x':{'aoa_deg':x1_test,
                     'ar': x2_test},
                'y':lift[i_test]
                }
degree = {
    'aoa_deg':2,
    'ar':3
}

liftsurr = lr.Polynomial(degree,points_train,points_test)
liftsurr.build()

xp = np.linspace(0,3.6,100)
yp0 = AR[0]*np.ones(xp.shape)
yp1 = AR[7]*np.ones(xp.shape)
yp2 = AR[14]*np.ones(xp.shape)
yp3 = AR[21]*np.ones(xp.shape)
yp4 = AR[28]*np.ones(xp.shape)
yp5 = AR[35]*np.ones(xp.shape)
yp6 = AR[42]*np.ones(xp.shape)

xp0 = np.zeros([2,len(xp)])
xp0[0,:] = xp
xp0[1,:] = yp0
xp1 = np.zeros([2,len(xp)])
xp1[0,:] = xp
xp1[1,:] = yp1
xp2 = np.zeros([2,len(xp)])
xp2[0,:] = xp
xp2[1,:] = yp2
xp3 = np.zeros([2,len(xp)])
xp3[0,:] = xp
xp3[1,:] = yp3
xp4 = np.zeros([2,len(xp)])
xp4[0,:] = xp
xp4[1,:] = yp4
xp5 = np.zeros([2,len(xp)])
xp5[0,:] = xp
xp5[1,:] = yp5
xp6 = np.zeros([2,len(xp)])
xp6[0,:] = xp
xp6[1,:] = yp6


fig, ax = plt.subplots()
ax.plot(xp,liftsurr.eval_surrogate(xp0),'b-')
ax.plot(xp,liftsurr.eval_surrogate(xp2),'r-')
ax.plot(xp,liftsurr.eval_surrogate(xp4),'g-')
ax.plot(xp,liftsurr.eval_surrogate(xp6),'k-')
ax.legend(['AR=20','AR=28','AR=36','AR=44'])
ax.grid(True)
ax.set_ylabel('lift (N)')
ax.set_xlabel('AoA (deg)')

plt.show()

