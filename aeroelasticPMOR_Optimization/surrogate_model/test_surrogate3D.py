"""Script to test the surrogate module functionality with a 3D surrogate
Author: Pablo de Felipe
Date: 17/05/22 (less than 3 weeks to the end)"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd

#from scipy.interpolate import RBFInterpolator
from scipy.interpolate import Rbf
import linear_regression as lr
import surrogate as sr
import pickle

filename = 'aoa_ar_taper_forces_v1.csv'
cwd = os.getcwd()

output_name = 'lift'
parameter_names = ['aoa_deg','ar','taper']
file_path = cwd+'/surrogate_data/'+filename
surrogate_type = 'polynomial'
degree = {'aoa_deg':2,
         'ar':3,
         'taper':4}

surr_dict = {
    'output_name': output_name,
    'parameter_names': parameter_names,
    'file_path': file_path,
    'surrogate_type': surrogate_type,
    'degree': degree
}
lift_surr = sr.Surrogate(surr_dict)
data_pandas = lift_surr.get_data()
num_models = lift_surr.num_models
i_train = np.arange(0,num_models+1,2)
i_test = np.arange(1,num_models,2)

lift_surr.sort_data(i_train,i_test)
#lift_surr.plot_doe()
lift_surr.build()
print(lift_surr.surr.theta)

ar_values = np.linspace(20,44,7)
taper_values = np.linspace(0.2,1.0,7)
aoa_deg_values = np.linspace(0,3.6,7)
xp = np.linspace(0,3.6,100)
yp0 = ar_values[0]*np.ones(xp.shape)
yp1 = ar_values[1]*np.ones(xp.shape)
yp2 = ar_values[2]*np.ones(xp.shape)
yp3 = ar_values[3]*np.ones(xp.shape)
yp4 = ar_values[4]*np.ones(xp.shape)
yp5 = ar_values[5]*np.ones(xp.shape)
yp6 = ar_values[6]*np.ones(xp.shape)

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

ax.plot(xp,lift_surr.surr.eval_surrogate(xp0),'b-')
ax.plot(xp,lift_surr.surr.eval_surrogate(xp2),'r-')
ax.plot(xp,lift_surr.surr.eval_surrogate(xp4),'g-')
ax.plot(xp,lift_surr.surr.eval_surrogate(xp6),'k-')
ax.legend(['AR=20','AR=28','AR=36','AR=44'])
ax.grid(True)
ax.set_ylabel('lift (N)')
ax.set_xlabel('AoA (deg)')

plt.show()
