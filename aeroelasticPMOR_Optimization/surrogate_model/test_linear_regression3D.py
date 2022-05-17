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

foldername = 'aoa_ar_taper_forces'
cwd = os.getcwd()
os.chdir(cwd+'/'+foldername+'/')
data_pandas = pd.read_csv(foldername+'.csv')
# Go back to the initial folder
os.chdir(cwd)
# Get data from pandas
data_numpy = data_pandas.to_numpy()
labels  = data_numpy[:,1]
AoA_deg = data_numpy[:,2]
AR      = data_numpy[:,3]
taper   = data_numpy[:,4]
lift    = data_numpy[:,5]
drag    = data_numpy[:,6]

label_i = np.zeros([len(labels),])
label_j = np.zeros([len(labels),])
label_k = np.zeros([len(labels),])
for i in range(len(labels)):
    label_i[i] = labels[i].split("_")[0]
    label_j[i] = labels[i].split("_")[1]
    label_k[i] = labels[i].split("_")[2]


num_models = len(data_numpy[:,1])
i_train = np.arange(0,num_models+1,2)
i_test = np.arange(1,num_models,2)
x1 = AoA_deg[i_train]
x1_test = AoA_deg[i_test]
x2 = AR[i_train]
x2_test = AR[i_test]
lift_test = lift[i_test]

# Get indices of points of interest
taper_index = np.where(label_k==6.)
ar0_index = np.where(label_j==0.)
aoa_index_ref = np.where(label_i==2.)


lift0_index = np.zeros([7,])
counter = 0
for i in range(49):
    if np.isin(ar0_index[0][i],taper_index):
        lift0_index[counter] = ar0_index[0][i]
        counter +=1
lift0 = lift[np.int_(lift0_index)]
lift_ref = lift0[2]
# Define the parameters required for the surrogate
points_train = {'x':{'aoa_deg':x1,
                     'ar': x2,
                     'taper':taper[i_train]},
                'y':lift[i_train]
                }
points_test = {'x':{'aoa_deg':x1_test,
                     'ar': x2_test,
                    'taper':taper[i_test]},
                'y':lift[i_test]
                }
degree = {
    'aoa_deg':2,
    'ar':3,
    'taper': 4
}


liftsurr = lr.Polynomial(degree,points_train,points_test)
liftsurr.build()
taper_values = np.linspace(0.2,1.0,7)
aoa_deg_values = np.linspace(0,3.6,7)
xp = np.linspace(0,3.6,100)
yp0 = AR[0]*np.ones(xp.shape)
yp1 = AR[7]*np.ones(xp.shape)
yp2 = AR[14]*np.ones(xp.shape)
yp3 = AR[21]*np.ones(xp.shape)
yp4 = AR[28]*np.ones(xp.shape)
yp5 = AR[35]*np.ones(xp.shape)
yp6 = AR[42]*np.ones(xp.shape)

zp0 = taper_values[0]*np.ones(xp.shape)
zp1 = taper_values[1]*np.ones(xp.shape)
zp2 = taper_values[2]*np.ones(xp.shape)
zp3 = taper_values[3]*np.ones(xp.shape)
zp4 = taper_values[4]*np.ones(xp.shape)
zp5 = taper_values[5]*np.ones(xp.shape)
zp6 = taper_values[6]*np.ones(xp.shape)

xp0 = {'aoa_deg':xp,'ar':yp0,'taper':zp6}
xp1 = {'aoa_deg':xp,'ar':yp1,'taper':zp6}
xp2 = {'aoa_deg':xp,'ar':yp2,'taper':zp6}
xp3 = {'aoa_deg':xp,'ar':yp3,'taper':zp6}
xp4 = {'aoa_deg':xp,'ar':yp4,'taper':zp6}
xp5 = {'aoa_deg':xp,'ar':yp5,'taper':zp6}
xp6 = {'aoa_deg':xp,'ar':yp6,'taper':zp6}

fig, ax = plt.subplots()
ax.plot(aoa_deg_values,lift0,'bx',label='_nolegend_')
ax.plot(xp,liftsurr.eval_surrogate(xp0),'b-')
ax.plot(xp,liftsurr.eval_surrogate(xp2),'r-')
ax.plot(xp,liftsurr.eval_surrogate(xp4),'g-')
ax.plot(xp,liftsurr.eval_surrogate(xp6),'k-')
ax.legend(['AR=20','AR=28','AR=36','AR=44'])
ax.grid(True)
ax.set_ylabel('lift (N)')
ax.set_xlabel('AoA (deg)')

plt.show()
# Check saving the surrogate
filename = 'lift3D_test.csv'
filepath = cwd+'/surrogate_built/'+filename
liftsurr.save_parameters(filepath)
