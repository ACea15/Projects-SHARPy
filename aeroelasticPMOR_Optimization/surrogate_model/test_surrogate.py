"""Script to test the functionality of the surrogate class
Date: 04/05/2022
Auhor: Pablo de Felipe
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import Rbf
import surrogate as sr

foldername = 'aoa_flutter'
cwd = os.getcwd()
os.chdir(cwd+'/'+foldername+'/')
data_pandas = pd.read_csv(foldername+'.csv')
# Go back to the initial folder
os.chdir(cwd)
###################DICT DEFFINITIONS###########################################
output_name = 'u_flutter'
parameter_names = ['aoa_deg']
folder_path = cwd+'/'+foldername+'/'
file_path = folder_path + foldername+'.csv'
error_file_path = folder_path+foldername+'_error.csv'
surr_dict = {
    'output_name': output_name,
    'parameter_names': parameter_names,
    'file_path': file_path,
}
flutter_surr = sr.Surrogate(surr_dict)
data_pandas = flutter_surr.get_data()
num_models = flutter_surr.num_models
i_train = np.array([0,2,4])
i_test = np.array([1,3])

flutter_surr.sort_data(i_train,i_test)

# Take reference at AR=32, AoA=1.2deg, Taper = 1.0, Twist=0.0, stiffness=0
u_flutter_ref = 28.     # Reference flutter speed for error evaluation

flutter_surr.test_1Dcases(u_flutter_ref)
flutter_surr.save_1Dcases_erros(error_file_path)

