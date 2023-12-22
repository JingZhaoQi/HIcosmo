#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 22:18:29 2023

@author: qijingzhao
"""

import emcee
import numpy as np
import h5py
import scipy.optimize as opt
from .Utilities import timer
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import os

chains_dir='./chains/'
if not os.path.exists(chains_dir):
    os.makedirs(chains_dir)
class MCMC:
    """
    Class to handle MCMC simulation using emcee.
    """
    def __init__(self, params_info, log_prob_function,filename="mcmc_results", restart_simulation_config=False, nwalkers=100):
        """
        Initializes the MCMC simulation.

        Args:
            params_info: Dictionary of parameters with their initial values and bounds.
                         For example: {'H0': (70, 50, 100), 'Omega_m': (0.3, 0, 1)}
            log_prob_function: The log probability function to use.
            nwalkers: The number of walkers.
            filename: Filename to save MCMC results.
            The restart_simulation_config parameter in the MCMC class is used to control the behavior of the simulation when an existing MCMC chain file is detected. 
            Specifically, it determines whether to restart the simulation from scratch or to continue from where it left off in the existing file.

            If restart_simulation_config is set to True, the simulation will disregard any previously completed steps and start anew, effectively resetting the MCMC process.
            If restart_simulation_config is set to False, the simulation will continue from the last recorded state in the existing file, preserving the progress made so far.
            This parameter is particularly useful for managing long-running simulations or when interruptions occur, allowing for either a fresh restart or a continuation based on the user's requirements.
        """
        self.params_info = params_info
        self.log_prob = log_prob_function
        self.restart_simulation_config = restart_simulation_config
        self.nwalkers = nwalkers
        self.ndim = len(params_info)
        self.p0 = [np.array([np.random.uniform(params_info[key][1], params_info[key][2]) for key in params_info]) for _ in range(nwalkers)]
        self.filename = chains_dir+filename+'.h5'
        self._check_existing_file()
        

    def log_probability(self, theta):
        """
        Wrapper for the log probability function.

        Args:
            theta: Parameter vector.

        Returns:
            Log probability value.
        """
        for i, key in enumerate(self.params_info):
            if not (self.params_info[key][1] < theta[i] < self.params_info[key][2]):
                return -np.inf
        return self.log_prob(theta)

    def _optimize_initial_params(self, ntest=10):
        """
        Optimize initial parameters for MCMC.

        Args:
            ntest: Number of test samples for optimization.

        Returns:
            Optimized initial positions for the walkers.
        """
        # Test samples
        test_samples = [np.array([np.random.uniform(self.params_info[key][1], self.params_info[key][2]) 
                                  for key in self.params_info]) for _ in range(ntest)]

        # Find best optimization result
        best_sample = None
        best_val = float('inf')
        for sample in test_samples:
            result = opt.minimize(lambda x: -self.log_probability(x), sample, method='Nelder-Mead')
            if result.fun < best_val:
                best_val = result.fun
                best_sample = result.x

        # Validate the optimization
        self._validate_optimization(sample, best_sample)

        # Generate initial positions for walkers
        nc = 1e-4
        self.p0 = [best_sample + nc * np.random.randn(self.ndim) for _ in range(self.nwalkers)]



    def _check_existing_file(self):
        """
        Check if an existing MCMC result file exists and decides to continue or restart based on configuration.
        """
        file_exists = os.path.exists(self.filename)
        print(f"Checking for existing file: {self.filename}")
        print(f"File exists: {file_exists}")
        print(f"Restart simulation config: {self.restart_simulation_config}")
        if file_exists:
            if self.restart_simulation_config:
                print("Restarting simulation as per configuration.")
                # Reset the backend and overwrite the existing file with new initial parameters
                with h5py.File(self.filename, "w") as f:  # 'w' mode will overwrite the file
                    f.attrs['params_info'] = str(self.params_info)
                    f.attrs['chi2_min'] = 0
                self._optimize_initial_params()
                self.completed_steps = 0
                self.backend = emcee.backends.HDFBackend(self.filename)
                self.backend.reset(self.nwalkers, self.ndim)
                self._optimize_initial_params()
                return True  # Indicate that we are restarting the simulation
            else:
                # Continue with existing simulation
                self.backend = emcee.backends.HDFBackend(self.filename)
                try:
                    self.completed_steps = self.backend.get_chain().shape[0]
                    print(f"Completed steps: {self.completed_steps}")
                except IndexError:
                    # 如果获取链的形状失败，尝试获取链的最后一个样本
                    try:
                        self.p0 = self.backend.get_chain()[-1]
                        self.completed_steps = len(self.backend.get_chain())
                    except IndexError:
                        # 如果还是失败，则重置后端
                        self.backend.reset(self.nwalkers, self.ndim)
                        self.completed_steps = 0
                # self.completed_steps = self.backend.get_chain().shape[0]
                print(f"Completed steps: {self.completed_steps}")
                return False  # Indicate that we are continuing the simulation
        else:
            print("No existing file found. Starting a new simulation.")
            # If file does not exist, create a new file and initialize parameters
            with h5py.File(self.filename, "a") as f:  # 'a' mode will create the file if it doesn't exist
                f.attrs['params_info'] = str(self.params_info)
                f.attrs['chi2_min'] = 0
            self.backend = emcee.backends.HDFBackend(self.filename)
            self.backend.reset(self.nwalkers, self.ndim)
            self._optimize_initial_params()
            self.completed_steps = 0
            return True  # Indicate that we are starting a new simulation

    # def _check_existing_file(self):
    #     """
    #     Check if an existing MCMC result file exists and decides to continue, restart, or stop based on configuration and step count.
    #     The target number of steps is assumed to be stored in self.target_steps.
    
    #     Returns:
    #         Boolean: True if the simulation should proceed, False otherwise.
    #     """
    #     file_exists = os.path.exists(self.filename)
    #     print(f"Checking for existing file: {self.filename}")
    #     print(f"File exists: {file_exists}")
    #     print(f"Restart simulation config: {self.restart_simulation_config}")
    
    #     if file_exists:
    #         self.backend = emcee.backends.HDFBackend(self.filename)
    #         try:
    #             self.completed_steps = self.backend.get_chain().shape[0]
    #         except IndexError:
    #             # Handle errors in reading the chain
    #             self.backend.reset(self.nwalkers, self.ndim)
    #             self.completed_steps = 0
    
    #         print(f"Completed steps: {self.completed_steps}")
    
    #         if self.completed_steps >= self.target_steps:
    #             print(f"Simulation has already completed {self.completed_steps} steps, which meets or exceeds the target of {self.target_steps} steps.")
    #             return False  # Indicate no need to continue running
    
    #         if self.restart_simulation_config:
    #             print("Restarting simulation as per configuration.")
    #             with h5py.File(self.filename, "w") as f:
    #                 f.attrs['params_info'] = str(self.params_info)
    #                 f.attrs['chi2_min'] = 0
    #             self._optimize_initial_params()
    #             self.backend.reset(self.nwalkers, self.ndim)
    #             self.completed_steps = 0
    #             return True  # Indicate starting a new simulation
    
    #         return True  # Indicate continuing the existing simulation
    
        # else:
        #     print("No existing file found. Starting a new simulation.")
        #     with h5py.File(self.filename, "a") as f:
        #         f.attrs['params_info'] = str(self.params_info)
        #         f.attrs['chi2_min'] = 0
        #     self._optimize_initial_params()
        #     self.backend.reset(self.nwalkers, self.ndim)
        #     self.completed_steps = 0
        #     return True  # Indicate starting a new simulation


        
        
    def _validate_optimization(self, original, optimized):
        """
        Validate the optimization result.

        Args:
            original: Original parameter values.
            optimized: Optimized parameter values.
        """
        if np.allclose(original, optimized, atol=1e-3):
            print("Warning: Optimized parameters are very close to the initial guess. This might indicate an issue with the likelihood function.")
            print(f"Original: {original}, Optimized: {optimized}")
            print("If you believe this is incorrect, please check your likelihood function.")
    
    @timer
    def run_mcmc(self, nsteps=20000, convergence_interval=100, check_convergence=True,**kwargs):
        """
        Runs the MCMC simulation with convergence check.

        Args:
            nsteps: The number of steps to run.
            check_convergence: If True, check for convergence.
            convergence_interval: Number of steps to wait before checking convergence.

        Returns:
            The emcee sampler object.
        """
        # Check if the simulation has already reached or exceeded the desired number of steps
        if self.completed_steps >= nsteps:
            print(f"Simulation already completed {self.completed_steps} steps, which meets or exceeds the target of {nsteps} steps. No further steps will be run.")
            return None  # You might want to return the current state of the sampler instead

        remaining_steps = nsteps - self.completed_steps  # Calculate the remaining steps
        
        print(f"Starting MCMC with a total of {remaining_steps} remaining steps. ")

        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_probability, backend=self.backend,**kwargs)
        
        if check_convergence:
            print("The simulation will stop early if convergence is reached.")
            for _ in range(0, nsteps, convergence_interval):
                steps_to_run = min(convergence_interval, remaining_steps - sampler.iteration)
                sampler.run_mcmc(self.p0, steps_to_run, progress=True, store=True)
                self.p0 = sampler.get_last_sample()
    
                if check_convergence and sampler.iteration >= convergence_interval:
                    try:
                        tau = sampler.get_autocorr_time(tol=0)
                        converged = np.all(tau * 100 < sampler.iteration)
                        if converged:
                            print(f"\n Convergence reached after {sampler.iteration} steps.")
                            break
                    except emcee.autocorr.AutocorrError:
                        pass
        else:
            sampler.run_mcmc(self.p0, nsteps, progress=True, store=True)

        chi2_min = -2 * np.max(sampler.get_log_prob())

        with h5py.File(self.filename, "a") as f:
            f.attrs['chi2_min'] = chi2_min

        return sampler


    def get_chain(self, discard=0, thin=1, flat=False):
        """
        Retrieves the MCMC chain from the HDF5 file.

        Args:
            discard: The number of steps to discard from the beginning.
            thin: The thinning factor for the chain.
            flat: If True, the chain is flattened; otherwise, the shape is (nwalkers, nsteps, ndim).

        Returns:
            The MCMC chain.
        """
        reader = emcee.backends.HDFBackend(self.filename, read_only=True)
        return reader.get_chain(discard=discard, thin=thin, flat=flat)

outdir='./results/'

class MCplot(object):
    def __init__(self, filenames_info):
        """
        Initialize the MCplot object.

        Args:
            filenames_info (list of tuple): A list of tuples, each containing the filename 
                                            of the MCMC result file (HDF5 format) 
                                            and its corresponding label.
        """
        self.filenames_info = filenames_info
        self.samples = []
        self.load_samples()

    def load_samples(self):
        """
        Load samples from multiple HDF5 files.
        """
        self.MCMC_name = []
        for filename, label in self.filenames_info:
            reader = emcee.backends.HDFBackend(chains_dir+filename+'.h5', read_only=True)
            samples = reader.get_chain(flat=True)
            self.MCMC_name.append(filename)
            # Load parameter names and ranges from file attributes if available
            with h5py.File(chains_dir+filename+'.h5', 'r') as f:
                if 'params_info' in f.attrs:
                    params_info = eval(f.attrs['params_info'])
                    param_names = list(params_info.keys())
                    self.ranges = {key: (min_val, max_val) for key, (_, min_val, max_val) in params_info.items()}
                else:
                    param_names = ['param_{}'.format(i) for i in range(samples.shape[1])]
                    self.ranges = None
                self.chi2_min = f.attrs['chi2_min']
            formatted_samples = MCSamples(samples=samples, names=param_names, labels=param_names, label=label, ranges=self.ranges,settings={'ignore_rows':0.3})
            self.param_names = param_names
            self.samples.append((formatted_samples, label))


    def reset_params_name(self, new_names):
        """
        Update the parameter names stored in the HDF5 files.

        Args:
            new_names (list): A list of new names for the parameters.
        """
        if not isinstance(new_names, list):
            raise ValueError("new_names must be a list of new parameter names.")

        new_samples = []
        for sample, label in self.samples:
            raw_samples = sample.samples
            if len(new_names) != raw_samples.shape[1]:
                raise ValueError("The length of new_names must match the number of parameters in the samples.")

            # Preserve the parameter ranges
            old_names = list(sample.getParamNames().list())
            new_sample = MCSamples(samples=raw_samples, names=new_names, labels=new_names, label=label, ranges=self.ranges)
            new_samples.append((new_sample, label))

        self.samples = new_samples
        self.param_names = new_names
        for filename, _ in self.filenames_info:
            with h5py.File(chains_dir+filename + '.h5', 'r+') as f:
                if 'params_info' in f.attrs:
                    # Update the parameter names
                    param_names = eval(f.attrs['params_info'])
                    new_params = {new_names[i]: param_names[old_names[i]] for i in range(len(new_names))}
                    f.attrs['params_info'] = str(new_params)
                else:
                    raise KeyError(f"params_info not found in {chains_dir+filename}")
        
        
    def plot1D(self, param_index_or_name, **kwargs):
        """
        Plot the 1D marginalized distribution for åa parameter.

        Args:
            param_index_or_name (int or str): Index or name of the parameter to plot.
            **kwargs: Additional keyword arguments for the plot.
        """
        g = plots.get_single_plotter()
        samples_list = [sample for sample, _ in self.samples]
        label_list = [label for _, label in self.samples]
        formatted_params = self.param_names[param_index_or_name] if isinstance(param_index_or_name, int) else param_index_or_name
        g.plot_1d(samples_list, formatted_params, **kwargs)
        g.add_legend(label_list, colored_text=True)
        plt.show()
        if 'fig_name' in kwargs:
            g.export(os.path.join(outdir,'%s.pdf'%kwargs['fig_name']))
        else:
            g.export(os.path.join(outdir+''.join(self.MCMC_name)+formatted_params.replace('\\','')+'_1D.pdf'))
        return g


    def plot2D(self, param_index_or_name, **kwargs):
        """
        Plot the 2D contour plot for specified parameters.

        Args:
            param_index_or_name (int or str): Index or name of the parameter to plot.
            **kwargs: Additional keyword arguments for the plot.
        """
        g = plots.getSinglePlotter(width_inch=8,ratio=1)
        samples_list = [sample for sample, _ in self.samples]
        label_list = [label for _, label in self.samples]
        formatted_params = [self.param_names[param] if isinstance(param, int) else param for param in param_index_or_name]
        # for sample, label in self.samples:
        g.plot_2d(samples_list, formatted_params[0], formatted_params[1], filled=True, **kwargs)
        # g.add_legend([label for _, label in self.samples], colored_text=True)
        g.add_legend(label_list, colored_text=True)
        plt.show()
        if 'fig_name' in kwargs:
            g.export(os.path.join(outdir,'%s.pdf'%kwargs['fig_name']))
        else:
            g.export(os.path.join(outdir+''.join(self.MCMC_name)+'_2D.pdf'))
        return g

    def plot_triangle(self, param_index_or_name= None, **kwargs):
        """
        Plot a triangle plot for specified parameters.

        Args:
            param_index_or_name (int or str): Index or name of the parameter to plot.
            **kwargs: Additional keyword arguments for the plot.
        """
        if param_index_or_name == None:
            param_index_or_name=self.param_names
        g = plots.get_subplot_plotter(width_inch=9)
        g.settings.legend_fontsize = 20
        g.settings.axes_fontsize = 14
        g.settings.lab_fontsize = 18
        g.settings.figure_legend_frame = False
        g.settings.alpha_filled_add=0.8
        samples_to_plot = [sample for sample, _ in self.samples]
        formatted_params = [self.param_names[param] if isinstance(param, int) else param for param in param_index_or_name]
        g.triangle_plot(samples_to_plot, formatted_params, filled=True, **kwargs)
        # g.add_legend([label for _, label in self.samples], colored_text=True,legend_loc=0)
        plt.show()
        if 'fig_name' in kwargs:
            g.export(os.path.join(outdir,'%s.pdf'%kwargs['fig_name']))
        else:
            g.export(os.path.join(outdir+''.join(self.MCMC_name)+'_triangle.pdf'))
        return g

    def calculate_aic_bic(self, n_data_points=None):
        """
        Calculate AIC and BIC.

        Args:
            n_data_points (int, optional): Number of data points for BIC calculation. If None, BIC is not calculated.

        Returns:
            tuple: (AIC, BIC) values. BIC will be None if n_data_points is not provided.
        """
        aic = bic = None

        n_params = len(self.param_names)

        aic = self.chi2_min + 2 * n_params
        if n_data_points is not None:
            bic = self.chi2_min + n_params * np.log(n_data_points)

        return aic, bic
    

    @property
    def results(self, n_data_points=None):
        total_chains = len(self.samples)
        max_params = max(len(sample.getParamNames().list()) for sample, _ in self.samples)
    
        fig, axs = plt.subplots(1, total_chains, figsize=(5 * total_chains, 6 + (max_params - 1)), dpi=90)
        if total_chains == 1:
            axs = [axs]
        for chain_index, (sample, label) in enumerate(self.samples):
            param_names, param_latex = sample.getLatex()
            aic, bic = self.calculate_aic_bic(n_data_points)
    
            axs[chain_index].axis('off')
            axs[chain_index].text(0.1, 0.9, f'Results for "{label}":', fontsize=18)
    
            for i, (name, latex) in enumerate(zip(param_names, param_latex)):
                axs[chain_index].text(0.1, 0.75 - i * 0.12, f"${name} = {latex}$", fontsize=15)
    
            if aic is not None:
                axs[chain_index].text(0.1, 0.1, f"AIC: {aic:.3f}", fontsize=15)
            if bic is not None:
                axs[chain_index].text(0.1, 0.05, f"BIC: {bic:.3f}", fontsize=15)
    
        plt.savefig(outdir + 'combined_results.png', dpi=300)
