""" Module implementing different linear regression strategies

The module contains various basis functions which are solved using a linear
regression problem, for now will be using polynomial basis and radial basis functions

    Typical usage example:

    surr = Polynomial(input_dict)
    y = surr.evaluate(x)
"""

import numpy as np

class Polynomial:
    """Implementation of the linear-regression problem with polynomial basis
    functions"""

    def __init__(self,degree, points_train, points_test):
        self.degree = degree
        self.x = points_train['x']
        self.y = points_train['y']
        self.x_test = points_test['x']
        self.y_test = points_test['y']

    def build(self):
        """ Function to build the model
        Args:
            Needs a built model

        Returns:
            Necessary components of the model"""
        x = self.x
        self.b = Polynomial.polynomial_basis(self,x)
        self.theta = Polynomial.eval_parameter(self,self.b)

    def polynomial_basis(self,x):
        """For every training point, evaluates the polynomial basis
        i.e b = [1 x x^2 ... x^k] where k is the order of the polynomial

        Args:
            self.degree: degree polynomial to be used
            x: training points (1D array, does not include
                multiple params)

        Returns:
            b (np.array): 2D array with evaluations of the basis function
                stencil for every training point used
            """
        k = 2#self.degree
        num = x.size
        b = np.zeros([num, k + 1])
        b[:, 0] = 1
        for i in range(num):
            for j in range(k):
                b[i, j + 1] = x[i] ** (j + 1)
        return b

    def eval_parameter(self,b):
        """ Function evaluates a theta matrix which solves a least squares
        problem: [theta]=[B^+]{y} where [B^+] is the Moore-Penrose Pseudo Inverse
        These points can then be used to evaluate the polynomial via:
        f(x) = theta[0]*b[0]+theta[1]*b[1]+...+theta[k+1]*b[k+1]"""

        y     = np.matrix(self.y).T # Use a column vector
        bpinv = np.matrix(np.linalg.pinv(b)) # calculate the pseudo inverse
        theta = bpinv*y
        return theta

    def eval_surrogate(self,x):
        """ Function which evaluates the surrogate function

        Args:
            x (np.array): Points at which surrogate output is seeked

        Returns
            y (np.arrya): Evaluations of the surrogate
        """
        y = np.zeros([len(x), ])
        params = self.theta
        k = self.degree
        for i in range(len(x)):
            btest = Polynomial.polynomial_basis(k, np.array([x[i]]))
            y[i] = np.dot(btest, params)

        return y



