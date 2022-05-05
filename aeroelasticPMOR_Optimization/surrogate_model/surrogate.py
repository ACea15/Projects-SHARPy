""" Module to build and plot results for a surrogate model of a specific quantity

The module contains a surrogate class which defines the data to be used to train the model and to test as well as the
method used to build it.

    Typical usage example:

    surr = Surrogate(input_dict)
    y = surr.evaluate(x)

    Needs the linear_regression.py module
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import Rbf
import linear_regression as lr

class Surrogate:
    """Implementation of surrogate building functionality for n-dimensions"""

    def __init__(self,input_dict):
        """
        Function to initialise the surrogate

        Args:
            input_dict: Dictionary with parameters for the surrogate

        """

        self.output_name = input_dict["output_name"]
        self.parameter_names = input_dict["parameter_names"]
        self.file_path = input_dict["file_path"]


    def get_data(self):
        """
        Reads the data from a .csv file specified as a file path
        Returns:
            data_pandas: data-frame with data
        """
        file_path = self.file_path
        data_pandas = pd.read_csv(file_path)
        self.data_pandas = data_pandas
        self.keys = list(data_pandas)
        data_numpy = data_pandas.to_numpy()
        self.num_models = len(data_numpy[:, 1])
        self.data_dict = {}

        for vars in self.parameter_names:
            index = self.keys.index(vars)
            self.data_dict[vars] = data_numpy[:,index]
        # Store the data in arrays for manipulation
        index = self.keys.index(self.output_name)
        self.data_dict[self.output_name] = data_numpy[:,index]
        return data_pandas
        # Get the keys from the data i.e AoA, AR, taper, twist, stiffness, delta_e or anything
    def sort_data(self,i_train,i_test):
        self.train_dict = {}
        self.test_dict = {}
        for vars in self.parameter_names:
            self.train_dict[vars] = self.data_dict[vars][i_train]
            self.test_dict[vars] = self.data_dict[vars][i_test]
        # Append output
        self.train_dict[self.output_name] = self.data_dict[self.output_name][i_train]
        self.test_dict[self.output_name] = self.data_dict[self.output_name][i_test]

    def test_1Dcases(self,ref_val):
        """ Function to evaluate the surrogate with polynomials and RBFs to
        determine which is the best for a single parameter dependency

        Args:
            ref_val - Reference value for the parameters in case

        """
        # Build the surrogates
        # Polynomial surrogates on lift
        points_train = {
            'x' : self.train_dict[self.parameter_names[0]],
            'y' : self.train_dict[self.output_name]
        }
        points_test  = {
            'x':self.test_dict[self.parameter_names[0]],
            'y':self.test_dict[self.output_name]
        }
        x = points_train['x']
        y = points_train['y']
        x_test = points_test['x']
        y_test = points_test['y']

        surr1 = lr.Polynomial(1, points_train, points_test)
        surr1.build()
        surr2 = lr.Polynomial(2, points_train, points_test)
        surr2.build()
        surr3 = lr.Polynomial(3, points_train, points_test)
        surr3.build()
        surr4 = lr.Polynomial(4, points_train, points_test)
        surr4.build()


        # Radial Basis functions on lift
        surr_rb_l = Rbf(x, y, function='linear')
        surr_rb_m = Rbf(x, y, function='multiquadric')
        surr_rb_g = Rbf(x, y, function='gaussian')
        surr_rb_c = Rbf(x, y, function='cubic')

        error_1 = surr1.eval_error()
        error_2 = surr2.eval_error()
        error_3 = surr3.eval_error()
        error_4 = surr4.eval_error()

        error_l = 0
        error_m = 0
        error_g = 0
        error_c = 0

        for i in range(len(x_test)):
            error_l += (surr_rb_l(x_test[i]) - y_test[i]) ** 2
            error_m += (surr_rb_m(x_test[i]) - y_test[i]) ** 2
            error_g += (surr_rb_g(x_test[i]) - y_test[i]) ** 2
            error_c += (surr_rb_c(x_test[i]) - y_test[i]) ** 2
        error_l = error_l / len(x_test)
        error_m = error_m / len(x_test)
        error_g = error_g / len(x_test)
        error_c = error_c / len(x_test)
        # check errors
        print('1st =', surr1.eval_error())
        print('2nd =', surr2.eval_error())
        print('3rd =', surr3.eval_error())
        print('4th =', surr4.eval_error())

        print('linear       =', error_l)
        print('Multiquadric =', error_m)
        print('Gaussian     =', error_g)
        print('Cubic        =', error_c)
        surr_type = ['1st','2nd', '3rd', '4th', 'linear', 'Multiquadric', 'Gaussian', 'Cubic']
        abs_err = np.array([error_1, error_2,
                   error_3,error_4, error_l, error_m, error_g, error_c])
        norm_err= abs_err/ref_val
        self.error1D_data = {
            "surr_type": surr_type,
            "norm_error": norm_err
        }
        self.error1D_data

        # Plot the points on the graph
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))


        xp = np.linspace(x[0],x[len(x)-1],100)
        ax[0].plot(x, y, 'bx')
        ax[0].plot(x_test, y_test, 'ro')

        ax[0].plot(xp, surr1.eval_surrogate(xp), 'k-')
        ax[0].plot(xp, surr2.eval_surrogate(xp), 'b-')
        ax[0].plot(xp, surr3.eval_surrogate(xp), 'r-')
        ax[0].plot(xp, surr4.eval_surrogate(xp), 'g-')
        # ax.plot(xp,yp4,'--')
        # ax.plot(xp,ynp2,'x')
        # ax.plot(xp,ynp3,'o')
        # ax.plot(xp,ynp4,'^')

        ax[0].legend(["Training points", "Testing points", "Data-Fit 1st","Data-Fit 2nd", "Data-fit 3rd", "Data-fit 4th"])
        ax[0].grid(True)
        ax[0].set_ylabel(self.output_name)
        ax[0].set_xlabel(self.parameter_names[0])

        ax[1].plot(x, y, 'bx')
        ax[1].plot(x_test, y_test, 'ro')
        ax[1].plot(xp, surr_rb_l(xp), '-')
        ax[1].plot(xp, surr_rb_m(xp), '--')
        ax[1].plot(xp, surr_rb_g(xp), ':')
        ax[1].plot(xp, surr_rb_c(xp), '-')

        ax[1].legend(
            ["Training points", "Testing points", "RB: linear", "RB: multiquadratic", "RB: Gaussian", 'RB: cubic'])
        ax[1].grid(True)
        ax[1].set_ylabel(self.output_name)
        ax[1].set_xlabel(self.parameter_names[0])

        plt.show()

    def save_1Dcases_erros(self,file_path):
        """Save the errors generated from a 1D case"""
        error_data_pandas = pd.DataFrame(self.error1D_data)
        error_data_pandas.to_csv(file_path)


