#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 09:53:45 2023

@author: qijingzhao
"""

# FisherMatrix.py
import numpy as np
from scipy.misc import derivative
from scipy.linalg import inv
from getdist import plots
from getdist.gaussian_mixtures import GaussianND
import matplotlib.pyplot as plt

class FisherMatrix:
    def __init__(self, function=None, df=None, params=None, fisher_matrix=None):
        """
        Initialize the FisherMatrix class.

        Parameters:
        - function: A function that predicts observables given the parameters. 
                    The last parameter of this function should be the redshift 'z'.
                    This is required if a predefined Fisher matrix is not provided.
        - df: The uncertainty of the function value(s), fixed for all parameters.
              Required if a predefined Fisher matrix is not provided.
        - params: A dictionary of the parameters and their values, excluding the redshift 'z'.
                  Required if a predefined Fisher matrix is not provided.
        - fisher_matrix: An optional predefined Fisher matrix. If this is provided,
                         the function, df, and params are not required, and the provided
                         Fisher matrix will be used for subsequent calculations.

        Note: The function provided should be defined such that it accepts the parameters 
              from 'params' followed by the redshift 'z' as its arguments. If a predefined
              Fisher matrix is used, ensure it is consistent with the calculations and analyses
              intended to be performed with this class.
        """
        if function is not None and df is not None and params is not None:
            self.function = function
            self.df = df
            self.params = params
            self.original_params = self.params.copy()
            self.fisher_matrix = None  # To be computed later
        elif fisher_matrix is not None:
            self.fisher_matrix = fisher_matrix
        else:
            raise ValueError("Must provide either a function with uncertainties and parameters, or a predefined Fisher matrix.")

  
    def compute_fisher_matrix_cosmolgoy(self, z, **kwargs):
        """
        Compute or retrieve the Fisher matrix.
        """
        if self.fisher_matrix is not None:
            self.original_fisher_matrix = self.fisher_matrix
            return self.fisher_matrix
        else:
            if not isinstance(z, np.ndarray):
                return self._calculate_fisher_matrix_at_z(z, **kwargs)
            else:
                fisher_matrix = 0
                if len(z) != len(self.df):
                    raise ValueError("The length of 'df' must match the length of 'z'.")
                for i, zi in enumerate(z):
                    fisher_matrix += self._calculate_fisher_matrix_at_z(zi, df=self.df[i], **kwargs)
                self.fisher_matrix=fisher_matrix
                self.original_fisher_matrix = self.fisher_matrix


    def _calculate_fisher_matrix_at_z(self, z, df, **kwargs):
        num_params = len(self.params)
        fisher_matrix = np.zeros((num_params, num_params))

        for i in range(num_params):
            for j in range(i, num_params):
                params_i = list(self.params.values())
                params_j = list(self.params.values())
                derivative_i = self.partial_derivative(self.function, i, params_i + [z], **kwargs)
                derivative_j = self.partial_derivative(self.function, j, params_j + [z], **kwargs)
                fisher_matrix[i, j] = fisher_matrix[j, i] = derivative_i * derivative_j / df**2
        self.original_fisher_matrix = fisher_matrix
        return fisher_matrix

    def partial_derivative(self, func, var=0, point=[], **kwargs):
        """
        Compute the partial derivative of a multivariate function.

        Parameters:
        - func: The function for which the derivative is to be computed.
        - var: The index of the variable with respect to which the derivative is taken. Defaults to the first variable.
        - point: The coordinates at which the derivative is computed, provided as a list.
        - **kwargs: Additional keyword arguments, e.g., step size 'dx' for numerical differentiation.

        Returns:
        - The approximate value of the partial derivative.
        """
        args = point[:]  # Copy the parameter list to avoid modifying the original parameters

        def wraps(x):
            args[var] = x
            return func(*args)

        dx = kwargs.get('dx', 1e-6)  # Default step size
        return derivative(wraps, point[var], dx=dx)

    
    def transform_fisher_matrix(self, z, equa, param):
        """
        Transform the Fisher matrix using a set of equations and update the internal Fisher matrix.

        Parameters:
        - z: Redshift or array of redshifts.
        - equa: List of function names.
        - param: List of parameter values.
        """
        if self.fisher_matrix is None:
            raise ValueError("Fisher matrix not computed. Call compute_fisher_matrix first.")

        if isinstance(z, list):
            z = np.asarray(z)
        if not isinstance(z, np.ndarray):
            transform_matrix = self.get_transformation_matrix(z, equa, param)
            self.fisher_matrix = transform_matrix.T @ self.fisher_matrix @ transform_matrix
        else:
            transformed_fisher = 0
            for i in z:
                transform_matrix = self.get_transformation_matrix(i, equa, param)
                transformed_fisher += transform_matrix.T @ self.fisher_matrix @ transform_matrix
            self.fisher_matrix = transformed_fisher

    def get_transformation_matrix(self, z, equa, param):
        """
        Calculate the transformation matrix for a given redshift and set of equations.

        Parameters:
        - z: Redshift.
        - equa: List of function names.
        - param: List of parameter values.

        Returns:
        - Transformation matrix.
        """
        num_equa = len(equa)
        num_params = len(param)
        matrix = np.zeros((num_equa, num_params))
        for i, func in enumerate(equa):
            for j in range(num_params):
                point = list(param)
                matrix[i, j] = self.partial_derivative(func, j, point + [z])
        return np.matrix(matrix)


    def delete_matrix_elements(self, matrix, indices):
        """
        Delete specified rows and columns from a matrix.

        Parameters:
        - matrix: The matrix to modify.
        - indices: Indices of rows/columns to delete.

        Returns:
        - Modified matrix.
        """
        return np.delete(np.delete(matrix, indices, axis=1), indices, axis=0)
    
    @property
    def get_parameter_errors(self):
        """
        Calculate the parameter errors from the internal Fisher matrix.

        Returns:
        - Parameter errors.
        """
        if self.fisher_matrix is None:
            raise ValueError("Fisher matrix not computed. Call compute_fisher_matrix first.")

        if self.fisher_matrix.ndim == 2:
            cov_matrix = inv(self.fisher_matrix)
            errors = np.sqrt(np.diagonal(cov_matrix))
        elif self.fisher_matrix.ndim == 3:
            num_matrices = self.fisher_matrix.shape[0]
            errors = np.zeros((self.fisher_matrix.shape[1], num_matrices))
            for i in range(num_matrices):
                cov_matrix = inv(self.fisher_matrix[i])
                errors[:, i] = np.sqrt(np.diagonal(cov_matrix))
        else:
            raise ValueError("Fisher matrix must be 2 or 3 dimensions.")
        return errors.tolist()

    def add_prior(self, var_index, error):
        """
        Add a prior to a parameter in the Fisher matrix.

        Parameters:
        - var_index: Index of the variable to add the prior to.
        - error: Uncertainty of the variable (standard deviation of the prior).
        """
        if self.fisher_matrix is None:
            raise ValueError("Fisher matrix not computed. Call compute_fisher_matrix first.")

        # Update the Fisher matrix to include the prior
        self.fisher_matrix[var_index, var_index] += 1 / error**2

    def marginalize_over_parameter(self, var_index):
        """
        Marginalize over a parameter in a Fisher matrix.

        Parameters:
        - fisher_matrix: Fisher matrix.
        - var_index: Index of the variable to marginalize over.

        Returns:
        - Marginalized Fisher matrix.
        """
        if self.fisher_matrix is None:
            raise ValueError("Fisher matrix not computed. Call compute_fisher_matrix first.")

        covariance_matrix = inv(self.fisher_matrix)
        new_covariance = self.delete_matrix_elements(covariance_matrix, var_index)
        self.fisher_matrix = inv(new_covariance)
        remove_index = list(self.params.keys())[var_index]
        self.params.pop(remove_index)
        return inv(new_covariance)

    def plot_triangle(self, params=None, title='', nsample=1000000, labels=None, **kwargs):
        """
        Plot a triangle plot (contour diagram) for selected parameters.

        Parameters:
        - params: List of parameter indices (int) or names (str) to include in the plot.
                  If None, all parameters will be plotted.
        - title: Title of the plot.
        - nsample: Number of samples to generate for the contour plot.
        - labels: Labels for the parameters, corresponding to the order in 'params'.
        - **kwargs: Additional keyword arguments for plotting.
        """
        if self.fisher_matrix is None:
            raise ValueError("Fisher matrix not computed. Call compute_fisher_matrix first.")

        param_names = list(self.params.keys())

        # Convert indices to names if necessary
        if params is not None:
            selected_params = [param_names[param] if isinstance(param, int) else param for param in params]
        else:
            selected_params = param_names  # All parameters

        # Get the indices of the selected parameters
        selected_indices = [param_names.index(param) for param in selected_params]

        # Extract the relevant part of the covariance matrix
        cov_matrix = inv(self.fisher_matrix)[np.ix_(selected_indices, selected_indices)]
        means = [self.params[param] for param in selected_params]

        # If no custom labels are provided, use the selected parameter names
        if not labels:
            labels = selected_params

        gauss = GaussianND(means, cov_matrix, names=selected_params, labels=labels)
        samples = gauss.MCSamples(nsample)

        g = plots.get_subplot_plotter()
        g.triangle_plot(samples, filled=True, **kwargs)
        plt.suptitle(title)

    def plot_2d_contour(self, param1, param2, title='', nsample=10000, **kwargs):
        """
        Plot a 2D contour diagram for two specific parameters.

        Parameters:
        - param1: Name of the first parameter.
        - param2: Name of the second parameter.
        - title: Title of the plot.
        - nsample: Number of samples to generate for the contour plot.
        - **kwargs: Additional keyword arguments for plotting.
        """
        if self.fisher_matrix is None:
            raise ValueError("Fisher matrix not computed. Call compute_fisher_matrix first.")
        
        param_names = list(self.params.keys())
        if isinstance(param1, int):
            param1 = param_names[param1]
        if isinstance(param2, int):
            param2 = param_names[param2]
        # Indices of the parameters
        idx1 = list(self.params.keys()).index(param1)
        idx2 = list(self.params.keys()).index(param2)

        cov_matrix = inv(self.fisher_matrix)[[idx1, idx2], :][:, [idx1, idx2]]
        means = [self.params[param1], self.params[param2]]

        gauss = GaussianND(means, cov_matrix, names=[param1, param2], labels=[param1, param2])
        samples = gauss.MCSamples(nsample)

        g = plots.get_single_plotter()
        g.plot_2d(samples, param1, param2, filled=True, **kwargs)
        plt.suptitle(title)