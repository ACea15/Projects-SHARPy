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
        self.x = points_train['x']      # Independent variables
        self.y = points_train['y']      # Dependent variable
        self.x_test = points_test['x']
        self.y_test = points_test['y']
        self.parameter_names = list(points_train['x'].keys())
        self.train_num = len(self.x[self.parameter_names[0]])
        self.test_num = len(self.x_test[self.parameter_names[0]])
        self.variable_num = len(self.parameter_names)
        x_array = np.zeros([self.variable_num,self.train_num])
        x_test_array = np.zeros([self.variable_num,self.test_num])
        # Pass x into an array of points
        for i in range(self.variable_num):
            x_array[i,:]=self.x[self.parameter_names[i]]
            x_test_array[i,:]=self.x_test[self.parameter_names[i]]

    def build(self):
        """ Function to build the model
        Args:
            Needs a built model

        Returns:
            Necessary components of the model"""
        if len(self.parameter_names) == 1:
            x = self.x[self.parameter_names[0]]
            self.b = Polynomial.polynomial_basis(self,x)
            self.theta = Polynomial.eval_parameter(self,self.b)

        elif len(self.parameter_names) == 2:
            x = self.x[self.parameter_names[0]]
            y = self.x[self.parameter_names[1]]
            self.b = Polynomial.polynomial_basis2D(self, x,y)
            self.theta = Polynomial.eval_parameter(self, self.b)
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
        k = self.degree
        num = x.size
        b = np.zeros([num, k + 1])
        b[:, 0] = 1
        for i in range(num):
            for j in range(k):
                b[i, j + 1] = x[i] ** (j + 1)
        return b

    def polynomial_basis2D(self,x, y):
        """Evaluates a vector using a kth degree polynomial in two directions.

        Args:
            x (np.array): Vector of points to be evaluated
            y (np.array): Vector of points to be evaluated
            kx     (int): Degree polynomial in variable x
            ky     (int): Degree polynomial in varibale y
        Returns:
            b (np.array): 2D array with evaluations of x,y for every point

        """
        kx = self.degree[self.parameter_names[0]]
        ky = self.degree[self.parameter_names[1]]
        num = x.size
        num2 = y.size
        if num != num2:
            print('Vectors x and y must be the same size')
        else:
            b = np.zeros([num, (kx + 1) * (ky + 1)])
            for k in range(num):
                counter = 0
                for i in range(kx + 1):
                    for j in range(ky + 1):
                        b[k, counter] = x[k] ** (i) * y[k] ** (j)
                        counter += 1
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

    def eval_surrogate(self,xp):
        """ Function which evaluates the surrogate function

        Args:
            x (np.array): Points at which surrogate output is seeked nxm array where
                         n is the number of dimensions and m the number of points to be evaluated

        Returns
            y (np.array): Evaluations of the surrogate
        """
        [n,m] = xp.shape
        if n != self.variable_num:
            print('Entered number of dimensions does not match dimensions for which surrogate is built for!')
        else:
            if n == 1:
                f = np.zeros([m, ])
                params = self.theta
                for i in range(m):
                    x = xp[0,:]
                    btest = Polynomial.polynomial_basis(self, np.array([x[i]]))
                    f[i] = np.dot(btest, params)

                return f

            elif n==2:
                x = xp[0,:]
                y = xp[1,:]
                f = np.zeros([len(x), ])
                params = self.theta
                for i in range(len(x)):
                    btest = Polynomial.polynomial_basis2D(self, np.array([x[i]]),np.array([y[i]]))
                    f[i] = np.dot(btest, params)
                return f

    def eval_error(self,x_test,y_test):
        """ Function which calculates the Mean Squared Error MSE
            This function will not be used to evaluate surrogate errors but is kept for compatibility
        Args:
            self - with attributes testing_points
        Returns:
            error - MSE error
        """

        error = 0;
        for i in range(len(x_test)):
            ys = Polynomial.eval_surrogate(self,np.array([[x_test[i],]]))
            error += (ys - y_test[i]) ** 2
        error = error / len(x_test)
        return float(error)


