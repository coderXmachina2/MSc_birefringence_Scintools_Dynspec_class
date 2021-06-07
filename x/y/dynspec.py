#!/usr/bin/env python

"""
dynspec.py
----------------------------------
Dynamic spectrum class
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import time
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import random
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import split
from copy import deepcopy as cp
from scint_models import scint_acf_model, scint_acf_model_2d_approx,\
                         scint_acf_model_2d, tau_acf_model, dnu_acf_model,\
                         fit_parabola, fit_log_parabola
from scint_utils import is_valid, svd_model, interp_nan_2d
from scipy.interpolate import griddata, interp1d, RectBivariateSpline, interp2d
from scipy.integrate import quad
from scipy.signal import convolve2d, medfilt, savgol_filter
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.stats import logistic
from scipy.special import expit

def sigmoid_func(x):
    return 1 / (1 + math.exp(-x))

class Dynspec:

    def __init__(self, filename=None, dyn=None, verbose=True, process=True,
                 lamsteps=False):
        """"
        Initialise a dynamic spectrum object by either reading from file
            or from existing object
        """

        if filename:
            self.load_file(filename, verbose=verbose, process=process,
                           lamsteps=lamsteps)
        elif dyn:
            self.load_dyn_obj(dyn, verbose=verbose, process=process,
                              lamsteps=lamsteps)
        else:
            print("Error: No dynamic spectrum file or object")

    def __add__(self, other):
        """
        Defines dynamic spectra addition, which is concatination in time,
            with the gaps filled
        """
        print("Now adding {} ...".format(other.name))

        if self.freq != other.freq \
                or self.bw != other.bw or self.df != other.df:
            print("WARNING: Code does not yet account for different \
                  frequency properties")

        # Set constant properties
        bw = self.bw
        df = self.df
        freqs = self.freqs
        freq = self.freq
        nchan = self.nchan
        dt = self.dt

        # Calculate properties for the gap
        timegap = round((other.mjd - self.mjd)*86400
                        - self.tobs, 1)  # time between two dynspecs
        extratimes = np.arange(self.dt/2, timegap, dt)
        if timegap < dt:
            extratimes = [0]
            nextra = 0
        else:
            nextra = len(extratimes)
        dyngap = np.zeros([np.shape(self.dyn)[0], nextra])

        # Set changed properties
        name = self.name.split('.')[0] + "+" + other.name.split('.')[0] \
            + ".dynspec"
        header = self.header + other.header
        # times = np.concatenate((self.times, self.times[-1] + extratimes,
        #                       self.times[-1] + extratimes[-1] + other.times))
        nsub = self.nsub + nextra + other.nsub
        tobs = self.tobs + timegap + other.tobs
        # Note: need to check "times" attribute for added dynspec
        times = np.linspace(0, tobs, nsub)
        mjd = np.min([self.mjd, other.mjd])  # mjd for earliest dynspec
        newdyn = np.concatenate((self.dyn, dyngap, other.dyn), axis=1)

        # Get new dynspec object with these properties
        newdyn = BasicDyn(newdyn, name=name, header=header, times=times,
                          freqs=freqs, nchan=nchan, nsub=nsub, bw=bw,
                          df=df, freq=freq, tobs=tobs, dt=dt, mjd=mjd)

        return Dynspec(dyn=newdyn, verbose=False, process=False)

    def load_file(self, filename, verbose=True, process=True, lamsteps=False):
        """
        Load a dynamic spectrum from psrflux-format file
        """

        start = time.time()
        # Import all data from filename
        if verbose:
            print("\nLOADING Dynspec Now! {0}...".format(filename))
        head = []
        with open(filename, "r") as file:
            for line in file:
                if line.startswith("#"):
                    headline = str.strip(line[1:])
                    head.append(headline)
                    if str.split(headline)[0] == 'MJD0:':
                        # MJD of start of obs
                        self.mjd = float(str.split(headline)[1])
        self.name = os.path.basename(filename)
        self.header = head

        rawdata = np.loadtxt(filename).transpose()  # read file
        #print("\nLen raw_data: " , " Len raw data: ", len(rawdata))
        #print("isub: ", rawdata[0], " Len isub: ", len(rawdata[0]))
        #print("ichan: ", rawdata[1], " Len ichan: ", len(rawdata[1]))
        #print("time(min): ", rawdata[2] , " Len time: ", len(rawdata[2]))
        #print("freq(MHz): ", rawdata[3], " Len freq: ",len(rawdata[3]))
        #print("flux: ", rawdata[4], " Len flux: ",len(rawdata[4]))
        #print("flux_err: ", rawdata[5], " Len flux_err: ", len(rawdata[5]))

        #So it declares a bunch of these from the raw data
        self.times = np.unique(rawdata[2]*60)  # time since obs start (secs)
        self.freqs = rawdata[3]  # Observing frequency in MHz.
        fluxes = rawdata[4]  # fluxes
        fluxerrs = rawdata[5]  # flux errors
        self.nchan = int(np.unique(rawdata[1])[-1])  # number of channels
        self.bw = self.freqs[-1] - self.freqs[0]  # obs bw
        self.df = round(self.bw/self.nchan, 5)  # channel bw
        self.bw = round(self.bw + self.df, 2)  # correct bw
        self.nchan += 1  # correct nchan
        self.nsub = int(np.unique(rawdata[0])[-1]) + 1
        self.tobs = self.times[-1]+self.times[0]  # initial estimate of tobs
        
        self.dt = self.tobs/self.nsub
        if self.dt > 1:
            self.dt = round(self.dt)
        else:
            self.times = np.linspace(self.times[0], self.times[-1], self.nsub)
        
        self.tobs = self.dt * self.nsub  # recalculated tobs
        # Now reshape flux arrays into a 2D matrix
        self.freqs = np.unique(self.freqs)
        self.freq = round(np.mean(self.freqs), 2)
        
        fluxes = fluxes.reshape([self.nsub, self.nchan]).transpose()
        fluxerrs = fluxerrs.reshape([self.nsub, self.nchan]).transpose()
        
        if self.df < 0:  # flip things
            self.df = -self.df
            self.bw = -self.bw
            # Flip flux matricies since self.freqs is now in ascending order
            fluxes = np.flip(fluxes, 0)
            fluxerrs = np.flip(fluxerrs, 0)
        # Finished reading, now setup dynamic spectrum
        
        self.dyn = fluxes  # initialise dynamic spectrum
        self.lamsteps = lamsteps
        
        if process:
            self.default_processing(lamsteps=lamsteps)  # do default processing
        end = time.time()
        if verbose:
            print("...LOADED in {0} seconds\n".format(round(end-start, 2)))
            self.info()

        #print("\nDyn is just fluxes: ", type(self.dyn))
        #print(self.dyn)
        #print("Prove it! ", len(self.dyn), " ", len(self.dyn[0])) #return this to notebook.

    def load_dyn_obj(self, dyn, verbose=True, process=True, lamsteps=False):
        """
        Load in a dynamic spectrum object of different type.
        """

        start = time.time()
        # Import all data from filename
        if verbose:
            print("LOADING DYNSPEC OBJECT {0}...".format(dyn.name))
        self.name = dyn.name
        self.header = dyn.header
        self.times = dyn.times  # time since obs start (secs)
        self.freqs = dyn.freqs  # Observing frequency in MHz.
        self.nchan = dyn.nchan  # number of channels
        self.nsub = dyn.nsub
        self.bw = dyn.bw  # obs bw
        self.df = dyn.df  # channel bw
        self.freq = dyn.freq
        self.tobs = dyn.tobs  # initial estimate of tobs
        self.dt = dyn.dt
        self.mjd = dyn.mjd
        self.dyn = dyn.dyn
        self.lamsteps = lamsteps
        #self.sec_spec_num = []

        if process:
            self.default_processing(lamsteps=lamsteps)  # do default processing
        end = time.time()
        if verbose:
            print("...LOADED in {0} seconds\n".format(round(end-start, 2)))
            self.info()

    def default_processing(self, lamsteps=False):
        """
        Default processing of a Dynspec object
        """
        #this
        self.trim_edges()  # remove zeros on band edges
        self.refill()  # refill with linear interpolation
        self.correct_dyn()
        self.calc_acf()  # calculate the ACF
        
        if lamsteps:
            self.scale_dyn()
        
        self.calc_sspec(lamsteps=lamsteps)  # Calculates secondary spectrum. IF you calc_sspec later again down the line

    def plot_dyn(self, 
                 lamsteps=False, 
                 input_dyn=None, 
                 filename=None,
                 input_x=None, 
                 input_y=None, 
                 trap=False, 
                 dpi_adj = 60,
                 display=True, 
                 title_size = 22.5,
                 title=None, 
                 man_blank_arr=[]):
        """
        Plot the dynamic spectrum.
        """
        plt.figure(1, figsize=(12, 6))
        if input_dyn is None:
            if lamsteps:
                if not hasattr(self, 'lamdyn'):
                    self.scale_dyn()
                dyn = self.lamdyn
            elif trap:
                if not hasattr(self, 'trapdyn'):
                    self.scale_dyn(scale='trapezoid')
                dyn = self.trapdyn
            else:
                dyn = self.dyn
        else:
            dyn = input_dyn
            
        medval = np.median(dyn[is_valid(dyn)*np.array(np.abs(is_valid(dyn)) > 0)])
        
        minval = np.min(dyn[is_valid(dyn)*np.array(np.abs(is_valid(dyn)) > 0)])
        
        # standard deviation
        
        std = np.std(dyn[is_valid(dyn)*np.array(np.abs(
                                                is_valid(dyn)) > 0)])
        vmin = minval
        vmax = medval+5*std        
        #plt.plot(self.freqs) #expect a linear line
        #plt.show()
        
        #What you need is a protocol that matches the blanking value to a self.freqs value.
        
        if input_dyn is None:
            if lamsteps:
                plt.pcolormesh(self.times/60, self.lam, dyn,
                               vmin=vmin, vmax=vmax)
                plt.ylabel('Wavelength (m)', fontsize = 20)
                
            else:
                plt.pcolormesh(self.times/60, self.freqs, dyn,
                               vmin=vmin, vmax=vmax)
                plt.ylabel('Frequency (MHz)', fontsize = 20)
                plt.tick_params(axis='both', which='major', labelsize=20)
                
                for lines in man_blank_arr:
                    plt.axhline(y = lines, alpha = 0.8, color='red')
                
            plt.xlabel('Time (mins)', fontsize = 19)

            cbar=plt.colorbar()
            cbar.ax.tick_params(labelsize=18) 
        else:
            plt.pcolormesh(input_x, input_y, dyn, vmin=vmin, vmax=vmax)
            
        if title:
            plt.title(title, fontsize=title_size)

        if filename is not None:
            plt.savefig(filename, 
                        dpi=dpi_adj, 
                        papertype='a4', 
                        bbox_inches='tight',
                        pad_inches=0.1)
            plt.close()
        elif input_dyn is None and display:
            plt.show()

    def plot_V_pol_ccf(self, method='acf1d', alpha=5.3, contour=False, filename=None,
                 input_acf=None, input_t=None, input_f=None, fit=True,
                 mcmc=False, display=True):
        """
        Plot the XCF. Autocorrelation function
        """

        if not hasattr(self, 'acf'):
            self.calc_acf()
        if not hasattr(self, 'tau') and input_acf is None and fit:
            self.get_scint_params(method=method, alpha=alpha, mcmc=mcmc)
        if input_acf is None:
            arr = self.acf
            tspan = self.tobs
            fspan = self.bw
        else:
            arr = input_acf
            tspan = max(input_t) - min(input_t)
            fspan = max(input_f) - min(input_f)
        arr = np.fft.ifftshift(arr)
        wn = arr[0][0] - arr[0][1]  # subtract the white noise spike
        arr[0][0] = arr[0][0] - wn  # Set the noise spike to zero for plotting
        arr = np.fft.fftshift(arr)
        t_delays = np.linspace(-tspan/60, tspan/60, np.shape(arr)[1])
        f_shifts = np.linspace(-fspan, fspan, np.shape(arr)[0])

        if input_acf is None:  # Also plot scintillation scales axes
            fig, ax1 = plt.subplots()
            if contour:
                im = ax1.contourf(t_delays, f_shifts, arr)
            else:
                im = ax1.pcolormesh(t_delays, f_shifts, arr)
            ax1.set_ylabel('Frequency lag (MHz)')
            ax1.set_xlabel('Time lag (mins)')
            miny, maxy = ax1.get_ylim()
            if fit:
                ax2 = ax1.twinx()
                ax2.set_ylim(miny/self.dnu, maxy/self.dnu)
                ax2.set_ylabel('Frequency lag / (dnu_d = {0})'.
                               format(round(self.dnu, 2)))
                ax3 = ax1.twiny()
                minx, maxx = ax1.get_xlim()
                ax3.set_xlim(minx/(self.tau/60), maxx/(self.tau/60))
                ax3.set_xlabel('Time lag/(tau_d={0})'.format(round(
                                                             self.tau/60, 2)))
            fig.colorbar(im, pad=0.15)
        else:  # just plot acf without scales
            if contour:
                plt.contourf(t_delays, f_shifts, arr)
            else:
                plt.pcolormesh(t_delays, f_shifts, arr)
            plt.ylabel('Frequency lag (MHz)')
            plt.xlabel('Time lag (mins)')

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
            plt.close()
        elif input_acf is None and display:
            plt.show()
            
            
    def plot_acf(self, method='acf1d', alpha=5.3, contour=False, filename=None,
                 input_acf=None, input_t=None, input_f=None, fit=True,
                 mcmc=False, display=True):
        """
        Plot the ACF. Autocorrelation function
        """

        if not hasattr(self, 'acf'):
            self.calc_acf()
        if not hasattr(self, 'tau') and input_acf is None and fit:
            self.get_scint_params(method=method, alpha=alpha, mcmc=mcmc)
        if input_acf is None:
            arr = self.acf
            tspan = self.tobs
            fspan = self.bw
        else:
            arr = input_acf
            tspan = max(input_t) - min(input_t)
            fspan = max(input_f) - min(input_f)
        arr = np.fft.ifftshift(arr) 
        wn = arr[0][0] - arr[0][1]  # subtract the white noise spike
        arr[0][0] = arr[0][0] - wn  # Set the noise spike to zero for plotting
        arr = np.fft.fftshift(arr)
        t_delays = np.linspace(-tspan/60, tspan/60, np.shape(arr)[1])
        f_shifts = np.linspace(-fspan, fspan, np.shape(arr)[0])

        if input_acf is None:  # Also plot scintillation scales axes
            fig, ax1 = plt.subplots()
            if contour:
                im = ax1.contourf(t_delays, f_shifts, arr)
            else:
                im = ax1.pcolormesh(t_delays, f_shifts, arr)
            ax1.set_ylabel('Frequency lag (MHz)')
            ax1.set_xlabel('Time lag (mins)')
            miny, maxy = ax1.get_ylim()
            if fit:
                ax2 = ax1.twinx()
                ax2.set_ylim(miny/self.dnu, maxy/self.dnu)
                ax2.set_ylabel('Frequency lag / (dnu_d = {0})'.
                               format(round(self.dnu, 2)))
                ax3 = ax1.twiny()
                minx, maxx = ax1.get_xlim()
                ax3.set_xlim(minx/(self.tau/60), maxx/(self.tau/60))
                ax3.set_xlabel('Time lag/(tau_d={0})'.format(round(
                                                             self.tau/60, 2)))
            fig.colorbar(im, pad=0.15)
        else:  # just plot acf without scales
            if contour:
                plt.contourf(t_delays, f_shifts, arr)
            else:
                plt.pcolormesh(t_delays, f_shifts, arr)
            plt.ylabel('Frequency lag (MHz)')
            plt.xlabel('Time lag (mins)')

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
            plt.close()
        elif input_acf is None and display:
            plt.show()

    def plot_sspec_birefinge(self, lamsteps=False, input_sspec=None, filename=None, plotlims=False,
                   input_x=None, input_y=None, trap=False, prewhite=True,
                   plotarc=False, maxfdop=np.inf, delmax=None, ref_freq=1400,
                   cutmid=0, startbin=0, display=True, colorbar=True,
                   title=None):
        """
        Plot the secondary spectrum
        """

        print("\nPlotting Sec From Birefringence...")

        if input_sspec is None:
            if lamsteps:
                if not hasattr(self, 'lamsspec'):
                    print("Recalculating SSpec")
                    self.calc_sspec(lamsteps=lamsteps, prewhite=prewhite)
                sspec = cp(self.lamsspec)
            elif trap:
                if not hasattr(self, 'trapsspec'):
                    self.calc_sspec(trap=trap, prewhite=prewhite)
                sspec = cp(self.trapsspec)
            else:
                if not hasattr(self, 'sspec'):
                    self.calc_sspec(lamsteps=lamsteps, prewhite=prewhite)
                sspec = cp(self.sspec)
            xplot = cp(self.fdop)
        else:
            sspec = input_sspec
            xplot = input_x
            
        medval = np.median(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        # std = np.std(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        maxval = np.max(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        vmin = medval - 3
        vmax = maxval - 3
        
        indicies = np.argwhere(np.abs(xplot) < maxfdop)
        xplot = xplot[indicies].squeeze()
        sspec = sspec[:, indicies].squeeze()
        nr, nc = np.shape(sspec)
        sspec[:, int(nc/2-np.floor(cutmid/2)):int(nc/2 +
                                                  np.ceil(cutmid/2))] = np.nan
        sspec[:startbin, :] = np.nan
        # Maximum value delay axis (us @ ref_freq)
        delmax = np.max(self.tdel) if delmax is None else delmax
        delmax = delmax*(ref_freq/self.freq)**2
        ind = np.argmin(abs(self.tdel-delmax))
        
        self.xplot_sspec = xplot
        self.plt_sspec_ind = ind
        self.plt_sspec_vmin= vmin
        self.plt_sspec_vmax= vmax
        self.plt_sspec = sspec
        
        if input_sspec is None:
            if lamsteps:
                plt.pcolormesh(xplot, self.beta[:ind], sspec[:ind, :],
                               vmin=vmin, vmax=vmax)
                plt.ylabel(r'$f_\lambda$ (m$^{-1}$)')
            else:
                plt.pcolormesh(xplot, self.tdel[:ind], sspec[:ind, :],
                               vmin=vmin, vmax=vmax)
                plt.ylabel(r'$f_\nu$ ($\mu$s)')
            plt.xlabel(r'$f_t$ (mHz)')
            
            if plotlims: #I introduced plotlims...
                plt.axvline(x=-2.5, color = 'red')
                plt.axvline(x=-2.5, color = 'red')
                plt.axvline(x=5, color = 'green')
                plt.axvline(x=-5, color = 'green')
                plt.axvline(x=10, color = 'blue')
                plt.axvline(x=-10, color = 'blue')
                
            bottom, top = plt.ylim()
            if plotarc:
                if lamsteps:
                    eta = self.betaeta #betaeta is your object, fitted from fir arc
                    
                else:
                    eta = self.eta #eta is your object
                    
                plt.plot(xplot, eta*np.power(xplot, 2), #This is literally the arc...
                         'r--', alpha=0.68) #Sample the points in which the arc intersects. That is your signal
            plt.ylim(bottom, top)

        else:
            plt.pcolormesh(xplot, input_y, sspec, vmin=vmin, vmax=vmax)
        if colorbar:
            plt.colorbar()
            
        if title:
            plt.title(title, fontsize=19)

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1) #you want to win you get this to work
            plt.close()
        elif input_sspec is None and display:
            plt.show()
    
    def plot_sspec_stripped(self, lamsteps=False, input_sspec=None, 
                            filename=None, plotlims=False,
                            input_x=None, input_y=None, 
                            trap=False, prewhite=True,
                            plotarc=False, maxfdop=np.inf, 
                            delmax=None, ref_freq=1400, 
                            plotarc_alpha=0.5,
                            cutmid=0, startbin=0, dpi_adj = 30 ,display=True, colorbar=True,
                   title=None):
        """
        Plot the secondary spectrum
        """

        print("\nPlotting Sec Spectrum:")

        if input_sspec is None:
            if lamsteps:
                if not hasattr(self, 'lamsspec'):
                    #So it recalculates the secondary spectrum every time it plots...
                    self.calc_sspec(lamsteps=lamsteps, prewhite=prewhite)
                sspec = cp(self.lamsspec)
            elif trap:
                if not hasattr(self, 'trapsspec'):
                    self.calc_sspec(trap=trap, prewhite=prewhite)
                sspec = cp(self.trapsspec)
            else:
                if not hasattr(self, 'sspec'):
                    self.calc_sspec(lamsteps=lamsteps, prewhite=prewhite)
                sspec = cp(self.sspec)
            xplot = cp(self.fdop)
        else:
            sspec = input_sspec
            xplot = input_x
            
        medval = np.median(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        # std = np.std(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        maxval = np.max(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        vmin = medval - 3
        vmax = maxval - 3

        # Get fdop plotting range
        # Range for x axis...
        
        indicies = np.argwhere(np.abs(xplot) < maxfdop)
        xplot = xplot[indicies].squeeze()
        sspec = sspec[:, indicies].squeeze()
        nr, nc = np.shape(sspec)
        sspec[:, int(nc/2-np.floor(cutmid/2)):int(nc/2 +
                                                  np.ceil(cutmid/2))] = np.nan
        sspec[:startbin, :] = np.nan
        # Maximum value delay axis (us @ ref_freq)
        delmax = np.max(self.tdel) if delmax is None else delmax
        delmax = delmax*(ref_freq/self.freq)**2
        ind = np.argmin(abs(self.tdel-delmax))
        
        self.xplot_sspec = xplot
        self.plt_sspec_ind = ind
        self.plt_sspec_vmin= vmin
        self.plt_sspec_vmax= vmax
        self.plt_sspec = sspec
        
        print("This is what it wanted:", self.xplot_sspec)
        
        """
        if input_sspec is None:
            if lamsteps:
                plt.pcolormesh(xplot, self.beta[:ind], sspec[:ind, :],
                               vmin=vmin, vmax=vmax)
                plt.ylabel(r'$f_\lambda$ (m$^{-1}$)', fontsize = 37)
                plt.tick_params(axis='both', which='major', labelsize=37)
            else:
                plt.pcolormesh(xplot, self.tdel[:ind], sspec[:ind, :],
                               vmin=vmin, vmax=vmax)
                plt.ylabel(r'$f_\nu$ ($\mu$s)', fontsize = 37)
                plt.tick_params(axis='both', which='major', labelsize=37)
            plt.xlabel(r'$f_t$ (mHz)', fontsize = 37)
            
            if plotlims: #I introduced plotlims...
                plt.axvline(x=-2.5, color = 'red', alpha=0.48, linestyle='dashed')
                plt.axvline(x=2.5, color = 'red', alpha=0.48, linestyle='dashed')
                plt.axvline(x=5, color = 'green', alpha=0.58, linestyle='dashed')
                plt.axvline(x=-5, color = 'green', alpha=0.58, linestyle='dashed')
                plt.axvline(x=10, color = 'blue', alpha=0.28, linestyle='dashed')
                plt.axvline(x=-10, color = 'blue', alpha=0.28, linestyle='dashed')
                
            bottom, top = plt.ylim()
            self.arc_top = top
            if plotarc:
                if lamsteps:
                    eta = self.betaeta #betaeta is your object, fitted from fir arc
                    
                else:
                    eta = self.eta #eta is you object
                    
                plt.plot(xplot, eta*np.power(xplot, 2), #This is literally the arc...
                         'r--', alpha=plotarc_alpha) #Sample the points in which the arc intersects. That is your signal
                
            plt.ylim(bottom, top)

        else:
            plt.pcolormesh(xplot, input_y, sspec, vmin=vmin, vmax=vmax)
        if colorbar:
            cbar=plt.colorbar()
            cbar.ax.tick_params(labelsize=37)
        if title:
            plt.title(title, fontsize=39)

        if filename is not None:
            plt.savefig(filename, dpi=dpi_adj, papertype='a4',bbox_inches='tight', pad_inches=0.1) #you want to win you get this to work
            plt.close()
        elif input_sspec is None and display:
            plt.show()
        """
             
    def plot_sspec(self, 
                   lamsteps=False,      #lamstep argument
                   input_sspec=None,    #we don't use this. It becomes something.
                   filename=None,       #for saving of plot
                   plotlims=False,      #I put this in
                   input_x=None,        #we don't use this
                   input_y=None,        #we don't use this
                   trap=False,          #we never use this
                   prewhite=True,       #preprocessing
                   plotarc=False,       #overlays plotted arc
                   
                   maxfdop=np.inf,      #maxfdop,  #document this
                   delmax=None,         #delmax    #document this
                   ref_freq=1400,       #reference frequency #document this
                   title_size = 22.5,
                   line_w = 4.0,
                   plotarc_alpha=0.5,   #I put this in
                   cutmid=0,            #This does something #perhaps we should document this
                   startbin=0,          #This does something #perhaps we should document this
                   dpi_adj = 30,        #I put this in
                   vmax_off = 3,        #I put this in
                   vmin_off= 3 ,        #I put this in
                   display=True,        #Is always true, its just for suppression.
                   colorbar=True,       #
                   title=None):         #title
        """
        Plot the secondary spectrum
        
        #vmax_off
        
        #vin_off
        
        """

        print("\nPlotting Sec Spectrum:")

        if input_sspec is None:
            if lamsteps:
                if not hasattr(self, 'lamsspec'):
                    #So it recalculates the secondary spectrum every time it plots...
                    self.calc_sspec(lamsteps=lamsteps, prewhite=prewhite)
                sspec = cp(self.lamsspec)
            elif trap:
                if not hasattr(self, 'trapsspec'):
                    self.calc_sspec(trap=trap, prewhite=prewhite)
                sspec = cp(self.trapsspec)
            else:
                if not hasattr(self, 'sspec'):
                    self.calc_sspec(lamsteps=lamsteps, prewhite=prewhite)
                sspec = cp(self.sspec)
            xplot = cp(self.fdop)
        else:
            sspec = input_sspec #now there is an input_sspec
            xplot = input_x
            
        medval = np.median(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        # std = np.std(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        maxval = np.max(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        
        #Thats what is controller.
        #suppose. vmax bounds the top
        #vmin bounds the bottom
        
        print("medval:",medval, "maxval:", maxval)
        
        #subtracting
        print("vmin, vmax offsets:", "vmin_off:" , vmin_off, "/" , "vmax_off:" , vmax_off)
        
        #vmin was once medval - 3
        #vmax was once maxval - 3
        
        vmin = medval - vmin_off
        vmax = maxval - vmax_off
        
        print("vmin:", vmin, "vmax:", vmax)

        # Get fdop plotting range
        # Range for x axis...
        
        indicies = np.argwhere(np.abs(xplot) < maxfdop)
        xplot = xplot[indicies].squeeze()
        sspec = sspec[:, indicies].squeeze()
              
        nr, nc = np.shape(sspec)
        sspec[:, int(nc/2-np.floor(cutmid/2)):int(nc/2 +
                                                  np.ceil(cutmid/2))] = np.nan
        sspec[:startbin, :] = np.nan
        # Maximum value delay axis (us @ ref_freq)
        delmax = np.max(self.tdel) if delmax is None else delmax
        delmax = delmax*(ref_freq/self.freq)**2
        ind = np.argmin(abs(self.tdel-delmax))
        
        self.xplot_sspec = xplot
        self.plt_sspec_ind = ind
        self.plt_sspec_vmin= vmin
        self.plt_sspec_vmax= vmax
        self.plt_sspec = sspec 
        
        #the reason why it has always been plotting the RR* / SQLD Fitted arc is because I always call plot_sspec before calling
        #arc_analytics
        
        #also, the reason for the ind is because this thing has ind...
        
        #xplot we know. its the fdop axis
        #self.beta is the lamstep axis
              
        #sspec is the 2 d array
              
        if input_sspec is None:
            if lamsteps:
                plt.pcolormesh(xplot, self.beta[:ind], sspec[:ind, :],
                               vmin=vmin, vmax=vmax)
                plt.ylabel(r'$f_\lambda$ (m$^{-1}$)', fontsize = 37)
                plt.tick_params(axis='both', which='major', labelsize=37)
            else:
                plt.pcolormesh(xplot, self.tdel[:ind], sspec[:ind, :],
                               vmin=vmin, vmax=vmax)
                plt.ylabel(r'$f_\nu$ ($\mu$s)', fontsize = 37)
                plt.tick_params(axis='both', which='major', labelsize=37)
            plt.xlabel(r'$f_t$ (mHz)', fontsize = 37)
            
            if plotlims: #I introduced plotlims...
                plt.axvline(x=-2.5, color = 'red', alpha=0.48, linestyle='dashed')
                plt.axvline(x=2.5, color = 'red', alpha=0.48, linestyle='dashed')
                plt.axvline(x=5, color = 'green', alpha=0.58, linestyle='dashed')
                plt.axvline(x=-5, color = 'green', alpha=0.58, linestyle='dashed')
                plt.axvline(x=10, color = 'blue', alpha=0.28, linestyle='dashed')
                plt.axvline(x=-10, color = 'blue', alpha=0.28, linestyle='dashed')
                
            bottom, top = plt.ylim()
            self.arc_top = top
            print("Arc top:", self.arc_top)
            if plotarc:
                if lamsteps:
                    eta = self.betaeta #betaeta is your object, fitted from fir arc
                    
                else:
                    eta = self.eta #eta is you object
                    
                plt.plot(xplot, eta*np.power(xplot, 2), #This is literally the arc...
                         'r--', alpha=plotarc_alpha, linewidth=line_w) #Sample the points in which the arc intersects. That is your signal. Fuckin amateur
                
            plt.ylim(bottom, top)

        else:
            plt.pcolormesh(xplot, input_y, sspec, vmin=vmin, vmax=vmax)
        if colorbar:
            cbar=plt.colorbar()
            cbar.ax.tick_params(labelsize=37)
        if title:
            plt.title(title, fontsize=title_size)

        if filename is not None:
            plt.savefig(filename, dpi=dpi_adj, papertype='a4',bbox_inches='tight', pad_inches=0.1) #you want to win you get this to work
            plt.close()
        elif input_sspec is None and display:
            plt.show()

    def plot_scat_im(self, display=True, plot_log=True, colorbar=True,
                     title=None, input_scat_im=None, input_fdop=None,
                     lamsteps=False, trap=False, clean=True, use_angle=False,
                     use_spatial=False, s=None, veff=None, d=None,
                     filename=None):
        """
        Plot the scattered image
        """
        c = 299792458.0  # m/s
        if input_scat_im is None:
            if not hasattr(self, 'scat_im'):
                if lamsteps:
                    input_sspec = self.lamsspec
                elif trap:
                    input_sspec = self.trapsspec
                else:
                    input_sspec = self.sspec
                self.calc_scat_im(input_sspec=input_sspec,
                                  lamsteps=lamsteps, trap=trap,
                                  clean=clean)
            else:
                scat_im = self.scat_im
                xyaxes = self.scat_im_ax

        else:
            scat_im = input_scat_im
            xyaxes = input_fdop

        if use_angle:
            # plot in on-sky angle
            thetarad = (xyaxes / (1e9 * self.freq)) * \
                       (c * s / (veff * 1000))
            thetaas = (thetarad * 180 / np.pi) * 3600
            xyaxes = thetaas
        elif use_spatial:
            # plot in spatial coordinates
            thetarad = (xyaxes / (1e9 * self.freq)) * \
                       (c * s / (veff * 1000))
            thetaas = (thetarad * 180 / np.pi) * 3600
            xyaxes = thetaas * (1 - s) * d * 1000

        if plot_log:
            scat_im -= np.min(scat_im)
            scat_im += 1e-10
            scat_im = 10 * np.log10(scat_im)
        medval = np.median(scat_im[is_valid(scat_im) *
                                   np.array(np.abs(scat_im) > 0)])
        maxval = np.max(scat_im[is_valid(scat_im) *
                                np.array(np.abs(scat_im) > 0)])
        vmin = medval - 3
        vmax = maxval - 3

        plt.pcolormesh(xyaxes, xyaxes, scat_im, vmin=vmin, vmax=vmax)
        plt.title('Scattered image')
        if use_angle:
            plt.xlabel('Angle parallel to velocity (as)')
            plt.ylabel('Angle perpendicular to velocity (as)')
        elif use_spatial:
            plt.xlabel('Distance parallel to velocity (AU)')
            plt.ylabel('Distance perpendicular to velocity (AU)')
        else:
            plt.xlabel('Angle parallel to velocity')
            plt.ylabel('Angle perpendicular to velocity')
        if title:
            plt.title(title)
        if colorbar:
            plt.colorbar()
        if filename is not None:
            plt.savefig(filename)
        if display:
            plt.show()
        else:
            plt.close()

        return xyaxes

    def plot_all(self, dyn=1, sspec=3, acf=2, norm_sspec=4, colorbar=True,
                 lamsteps=False, filename=None, display=True):
        """
        Plots multiple figures in one
        """

        # Dynamic Spectrum
        plt.subplot(2, 2, dyn)
        self.plot_dyn(lamsteps=lamsteps)
        plt.title("Dynamic Spectrum")

        # Autocovariance Function
        plt.subplot(2, 2, acf)
        self.plot_acf(subplot=True)
        plt.title("Autocovariance")

        # Secondary Spectrum
        plt.subplot(2, 2, sspec)
        plt.title("Secondary Spectrum")

        # Normalised Secondary Spectrum
        plt.subplot(2, 2, norm_sspec)
        self.norm_sspec(plot=True, scrunched=False, lamsteps=lamsteps,
                        plot_fit=False)
        plt.title("Normalised fdop secondary spectrum")

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
            plt.close()
        elif display:
            plt.show()

    def fit_arc(self, method='norm_sspec', 
                asymm=False, 
                plot=False,
                delmax=None, 
                numsteps=1e4, 
                startbin=3, 
                cutmid=3, 
                lamsteps=True,
                etamax=None, 
                etamin=None, 
                low_power_diff=-3,
                high_power_diff=-1.5,
                ref_freq=1400, 
                constraint=[0, np.inf],
                nsmooth=5, filename=None, 
                noise_error=True, display=True,
                log_parabola=False):
        """
        Find the arc curvature with maximum power along it

            constraint: Only search for peaks between constraint[0] and
                constraint[1]
        """

        if not hasattr(self, 'tdel'):
            self.calc_sspec()
            
        delmax = np.max(self.tdel) if delmax is None else delmax
        delmax = delmax*(ref_freq/self.freq)**2  # adjust for frequency

        if lamsteps:
            print("It is lamsteps")
            if not hasattr(self, 'lamsspec'):
                self.calc_sspec(lamsteps=lamsteps)
            sspec = np.array(cp(self.lamsspec))
            yaxis = cp(self.beta)
            ind = np.argmin(abs(self.tdel-delmax))
            self.ind_sv = ind
            ymax = self.beta[ind]  #cut beta at equivalent value to delmax #yes thats it.
                                   #what is the 
            self.ymax_sv = ymax
        else:
            print("It is not lamsteps")
            if not hasattr(self, 'sspec'): #Why is it wobblying here?
                self.calc_sspec()
            sspec = np.array(cp(self.sspec))
            yaxis = cp(self.tdel)
            ymax = delmax

        nr, nc = np.shape(sspec)
        # Estimate noise in secondary spectrum
        a = np.array(sspec[int(nr/2):,
                           int(nc/2 + np.ceil(cutmid/2)):].ravel())
        b = np.array(sspec[int(nr/2):, 0:int(nc/2 -
                                             np.floor(cutmid/2))].ravel())
        noise = np.std(np.concatenate((a, b)))

        # Adjust secondary spectrum
        ind = np.argmin(abs(self.tdel-delmax))
        sspec[0:startbin, :] = np.nan  # mask first N delay bins
        
        # mask middle N Doppler bins
        sspec[:, int(nc/2 - np.floor(cutmid/2)):int(nc/2 +
              np.ceil(cutmid/2))] = np.nan
        
        sspec = sspec[0:ind, :]  # cut at delmax
        yaxis = yaxis[0:ind]

        # noise of mean out to delmax.
        noise = np.sqrt(np.sum(np.power(noise, 2)))/np.sqrt(len(yaxis)*2)

        if etamax is None:
            etamax = ymax/((self.fdop[1]-self.fdop[0])*cutmid)**2
        if etamin is None:
            etamin = (yaxis[1]-yaxis[0])*startbin/(max(self.fdop))**2

        try:
            len(etamin)
            etamin_array = np.array(etamin).squeeze()
            etamax_array = np.array(etamax).squeeze()
        except TypeError:
            # Force to be arrays for iteration
            etamin_array = np.array([etamin])
            etamax_array = np.array([etamax])

        # At 1mHz for 1400MHz obs, the maximum arc terminates at delmax
        max_sqrt_eta = np.sqrt(np.max(etamax_array))
        min_sqrt_eta = np.sqrt(np.min(etamin_array))
        
        # Create an array with equal steps in sqrt(curvature)
        sqrt_eta_all = np.linspace(min_sqrt_eta, max_sqrt_eta, numsteps)

        for iarc in range(0, len(etamin_array)):
            if len(etamin_array) == 1:
                etamin = etamin
                etamax = etamax
            else:
                etamin = etamin_array.squeeze()[iarc]
                etamax = etamax_array.squeeze()[iarc]

            if not lamsteps:
                c = 299792458.0  # m/s
                beta_to_eta = c*1e6/((ref_freq*10**6)**2)
                etamax = etamax/(self.freq/ref_freq)**2  # correct for freq
                etamax = etamax*beta_to_eta
                etamin = etamin/(self.freq/ref_freq)**2
                etamin = etamin*beta_to_eta
                constraint = constraint/(self.freq/ref_freq)**2
                constraint = constraint*beta_to_eta

            sqrt_eta = sqrt_eta_all[(sqrt_eta_all <= np.sqrt(etamax)) *
                                    (sqrt_eta_all >= np.sqrt(etamin))]
            numsteps_new = len(sqrt_eta)

            # initiate
            etaArray = []

            if method == 'norm_sspec':
                # Get the normalised secondary spectrum, set for minimum eta as
                #   normalisation. Then calculate peak as
                self.norm_sspec(eta=etamin, delmax=delmax, plot=False,
                                startbin=startbin, maxnormfac=1, cutmid=cutmid,
                                lamsteps=lamsteps, scrunched=True,
                                plot_fit=False, numsteps=numsteps_new)
                
                norm_sspec = self.normsspecavg.squeeze()
                etafrac_array = np.linspace(-1, 1, len(norm_sspec))
                ind1 = np.argwhere(etafrac_array > 1/(2*len(norm_sspec)))
                ind2 = np.argwhere(etafrac_array < -1/(2*len(norm_sspec)))

                norm_sspec_avg = np.add(norm_sspec[ind1],
                                        np.flip(norm_sspec[ind2], axis=0))/2
                norm_sspec_avg = norm_sspec_avg.squeeze()
                etafrac_array_avg = 1/etafrac_array[ind1].squeeze()
                # Make sure is valid
                filt_ind = is_valid(norm_sspec_avg)
                norm_sspec_avg = np.flip(norm_sspec_avg[filt_ind], axis=0)
                etafrac_array_avg = np.flip(etafrac_array_avg[filt_ind],
                                            axis=0)

                # Form eta array and cut at maximum
                etaArray = etamin*etafrac_array_avg**2
                ind = np.argwhere(etaArray < etamax)
                etaArray = etaArray[ind].squeeze()
                norm_sspec_avg = norm_sspec_avg[ind].squeeze()

                # Smooth data
                norm_sspec_avg_filt = \
                    savgol_filter(norm_sspec_avg, nsmooth, 1)

                #print("What are norm_sspec_avg and norm_sspec_avg_filt?")
                #print("Norm sspec avg", norm_sspec_avg)                
                #print("Norm sspec avg filt", norm_sspec_avg_filt)
                # search for peaks within constraint range
                indrange = np.argwhere((etaArray > constraint[0]) *
                                       (etaArray < constraint[1]))

                sumpow_inrange = norm_sspec_avg_filt[indrange]
                ind = np.argmin(np.abs(norm_sspec_avg_filt -
                                       np.max(sumpow_inrange)))

                # Now find eta and estimate error by fitting parabola
                #   Data from -3db on low curvature side to -1.5db on high side
                max_power = norm_sspec_avg_filt[ind]
                power = max_power
                ind1 = 1
                while (power > max_power + low_power_diff and
                       ind + ind1 < len(norm_sspec_avg_filt)-1):  # -3db
                    ind1 += 1
                    power = norm_sspec_avg_filt[ind - ind1]
                power = max_power
                ind2 = 1
                while (power > max_power + high_power_diff and
                       ind + ind2 < len(norm_sspec_avg_filt)-1):  # -1db power
                    ind2 += 1
                    power = norm_sspec_avg_filt[ind + ind2]
                # Now select this region of data for fitting
                xdata = etaArray[int(ind-ind1):int(ind+ind2)]
                ydata = norm_sspec_avg[int(ind-ind1):int(ind+ind2)]

                # Do the fit
                # yfit, eta, etaerr = fit_parabola(xdata, ydata)
                if log_parabola:
                    yfit, eta, etaerr = fit_log_parabola(xdata, ydata)
                else:
                    yfit, eta, etaerr = fit_parabola(xdata, ydata)

                if np.mean(np.gradient(np.diff(yfit))) > 0:
                    raise ValueError('Fit returned a forward parabola.')
                eta = eta

                if noise_error:
                    # Now get error from the noise in secondary spectra instead
                    etaerr2 = etaerr  # error from parabola fit
                    power = max_power
                    ind1 = 1
                    while (power > (max_power - noise) and (ind - ind1 > 1)):
                        power = norm_sspec_avg_filt[ind - ind1]
                        ind1 += 1
                    power = max_power
                    ind2 = 1
                    while (power > (max_power - noise) and
                           (ind + ind2 < len(norm_sspec_avg_filt) - 1)):
                        ind2 += 1
                        power = norm_sspec_avg_filt[ind + ind2]

                    etaerr = np.abs(etaArray[int(ind-ind1)] -
                                    etaArray[int(ind+ind2)])/2

                if plot and iarc == 0:
                    
                    print("What if I just do the max of the stuff. Would I get something reasonable?")
                    print(np.max(etaArray))
                    
                    print(np.max(norm_sspec_avg))
                    print(np.max(norm_sspec_avg_filt))
                    print(np.max(yfit))
                    
                    plt.plot(etaArray, norm_sspec_avg)
                    plt.plot(etaArray, norm_sspec_avg_filt)
                    plt.plot(xdata, yfit)
                    
                    #plt.axvline(x=np.max(norm_sspec_avg), color = 'red')
                    #plt.axvline(x=np.max(norm_sspec_avg_filt), color = 'blue')
                    #plt.axvline(x=np.max(yfit), color = 'green')
                    
                    
                    self.norm_sspec_avg_samp =norm_sspec_avg
                    self.norm_sspec_avg_filt_samp =norm_sspec_avg_filt
                    self.etaArray_samp = etaArray
                    
                    self.xdata_samp=xdata
                    self.yfit_samp = yfit
                    
                    plt.tick_params(axis='both', which='major', labelsize=37)
                    plt.axvspan(xmin=eta-etaerr, xmax=eta+etaerr,
                                facecolor='C2', alpha=0.5)
                    plt.xscale('log')
                    if lamsteps:
                        plt.xlabel(r'Arc curvature, '
                                   r'$\eta$ (${\rm m}^{-1}\,{\rm mHz}^{-2}$)', fontsize = 32)
                    else:
                        plt.xlabel('eta (tdel)', fontsize = 32)
                    plt.ylabel(r'Mean Power, ' r'$P_{arc}$ (dB)', fontsize = 32)
                elif plot:
                    plt.plot(xdata, yfit,
                             color='C{0}'.format(str(int(3+iarc))))
                    plt.tick_params(axis='both', which='major', labelsize=37)
                    

                    
                    plt.axvspan(xmin=eta-etaerr, xmax=eta+etaerr,
                                facecolor='C{0}'.format(str(int(3+iarc))),
                                alpha=0.3)
                if plot and iarc == len(etamin_array)-1:
                    if filename is not None:
                        plt.title(r'Mean Power versus trial curvature, ' r'$P_{arc}(\eta)$ (dB)', fontsize = 32)
                        plt.grid(alpha = 0.62)
                        plt.savefig(filename, figsize=(6, 6), dpi=140,
                                    bbox_inches='tight', pad_inches=0.1)
                        plt.close()
                    elif display:
                        plt.show()

            else:
                raise ValueError('Unknown arc fitting method. Please choose \
                                 from gidmax or norm_sspec')

            if iarc == 0:  # save primary
                if lamsteps:
                    self.betaeta = eta
                    self.betaetaerr = etaerr / np.sqrt(2)
                    self.betaetaerr2 = etaerr2 / np.sqrt(2)
                else:
                    self.eta = eta
                    self.etaerr = etaerr / np.sqrt(2)
                    self.etaerr2 = etaerr2 / np.sqrt(2)

    def fit_arclets(self, lamsteps=False):

        if not hasattr(self, 'betaeta') and not hasattr(self, 'eta'):
            self.fit_arc(plot=False, log_parabola=True, low_power_diff=-0.5,
                         high_power_diff=-0.5, delmax=0.8)

        medval = np.median(self.sspec[is_valid(self.sspec) *
                                      np.array(np.abs(self.sspec) > 0)])
        maxval = np.max(self.sspec[is_valid(self.sspec) *
                                   np.array(np.abs(self.sspec) > 0)])
        vmin = medval - 3
        vmax = maxval - 3
        plt.pcolormesh(self.fdop, self.tdel, self.sspec, vmin=vmin,
                       vmax=vmax)
        plt.ylim((0, max(self.tdel)))

        if lamsteps:
            yaxis = self.beta
            sspec = 10**(self.lamsspec / 10)
            eta = self.betaeta
        else:
            yaxis = self.tdel
            sspec = 10**(self.sspec / 10)
            eta = self.eta
        g = RectBivariateSpline(yaxis, self.fdop, sspec)

        avg_power_f = []
        avg_power_err_f = []
        arc_len = []
        fdop = []
        for f in self.fdop:
            if abs(2 * f) < max(self.fdop) and (eta * f**2 < max(yaxis)) \
               and f != 0:

                def integrand(fd):
                    return g(fd, eta * (f ** 2 - (fd - f) ** 2))

                fmin = 0
                fmax = 2 * f
                x = np.linspace(fmin, fmax, 20)
                plt.plot(x, eta * (f ** 2 - (x - f) ** 2))
                power, power_err = quad(integrand, fmin, fmax, limit=1000)

                arc_factor = 2 / ((fmax - fmin) / f)
                arc_length = (arc_factor**2 * np.arcsinh((2 / arc_factor) *
                                                         eta * f) + 2 *
                              eta * f * np.sqrt(arc_factor**2 + 4 *
                                                (eta * f)**2)) / \
                             (2 * arc_factor**2 * eta)

                avg_power = power / arc_length
                avg_power_err = power_err / arc_length

                avg_power_f.append(avg_power)
                avg_power_err_f.append(avg_power_err)
                arc_len.append(arc_length)
                fdop.append(f)

        plt.show()

        power_f = np.array(avg_power_f)
        power_err_f = np.array(avg_power_err_f)
        fdop = np.array(fdop)

        plt.plot(fdop, power_f)
        plt.fill_between(fdop, power_f + power_err_f, power_f - power_err_f,
                         color='r', alpha=0.5)
        plt.xlabel(r'$f_t$ (mHz)')
        plt.ylabel(r'Power')
        plt.ylim((0, 5e2))
        plt.show()

        # plt.plot(fdop, arc_len)
        # plt.show()

    def norm_sspec(self, eta=None, delmax=None, plot=False, startbin=1,
                   maxnormfac=5, minnormfac=0, cutmid=3, lamsteps=False,
                   scrunched=True, plot_fit=True, ref_freq=1400, numsteps=None,
                   filename=None, display=True, unscrunched=True,
                   powerspec=False, interp_nan=False):
        """
        Normalise fdop axis using arc curvature
        """

        # Maximum value delay axis (us @ ref_freq)
        delmax = np.max(self.tdel) if delmax is None else delmax
        delmax = delmax*(ref_freq/self.freq)**2

        if lamsteps:
            if not hasattr(self, 'lamsspec'):
                self.calc_sspec(lamsteps=lamsteps)
            sspec = cp(self.lamsspec)
            yaxis = cp(self.beta)
            if not hasattr(self, 'betaeta') and eta is None:
                self.fit_arc(lamsteps=lamsteps, delmax=delmax, plot=plot,
                             startbin=startbin)
        else:
            if not hasattr(self, 'sspec'):
                self.calc_sspec()
            sspec = cp(self.sspec)
            yaxis = cp(self.tdel)
            if not hasattr(self, 'eta') and eta is None:
                self.fit_arc(lamsteps=lamsteps, delmax=delmax, plot=plot,
                             startbin=startbin)
        if eta is None:
            if lamsteps:
                eta = self.betaeta
            else:
                eta = self.eta
        else:  # convert to beta
            if not lamsteps:
                c = 299792458.0  # m/s
                beta_to_eta = c*1e6/((ref_freq*10**6)**2)
                eta = eta/(self.freq/ref_freq)**2  # correct for frequency
                eta = eta*beta_to_eta

        medval = np.median(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        maxval = np.max(sspec[is_valid(sspec)*np.array(np.abs(sspec) > 0)])
        vmin = medval - 3
        vmax = maxval - 3

        ind = np.argmin(abs(self.tdel-delmax))
        sspec = sspec[startbin:ind, :]  # cut first N delay bins and at delmax
        # sspec[0:startbin] = np.nan
        nr, nc = np.shape(sspec)
        # mask out centre bins
        sspec[:, int(nc/2 - np.floor(cutmid/2)):int(nc/2 +
              np.floor(cutmid/2))] = np.nan
        tdel = yaxis[startbin:ind]
        # tdel = yaxis[:ind]
        fdop = self.fdop
        maxfdop = maxnormfac*np.sqrt(tdel[-1]/eta)  # Maximum fdop for plot
        if maxfdop > max(fdop):
            maxfdop = max(fdop)
        # Number of fdop bins to use. Oversample by factor of 2
        nfdop = 2*len(fdop[abs(fdop) <=
                           maxfdop]) if numsteps is None else numsteps
        fdopnew = np.linspace(-maxnormfac, maxnormfac,
                              nfdop)  # norm fdop
        if minnormfac > 0:
            unscrunched = False  # Cannot plot 2D function
            inds = np.argwhere(np.abs(fdopnew) > minnormfac)
            fdopnew = fdopnew[inds]
        normSspec = []
        isspectot = np.zeros(np.shape(fdopnew))
        for ii in range(0, len(tdel)):
            itdel = tdel[ii]
            imaxfdop = maxnormfac*np.sqrt(itdel/eta)
            ifdop = fdop[abs(fdop) <= imaxfdop]/np.sqrt(itdel/eta)
            isspec = sspec[ii, abs(fdop) <= imaxfdop]  # take the iith row
            ind = np.argmin(abs(fdopnew))
            normline = np.interp(fdopnew, ifdop, isspec)
            normSspec.append(normline)
            isspectot = np.add(isspectot, normline)
        normSspec = np.array(normSspec).squeeze()
        if interp_nan:
            # interpolate NaN values
            normSspec = interp_nan_2d(normSspec)
        isspecavg = np.mean(normSspec, axis=0)  # make average
        powerspectrum = np.nanmean(np.power(10, normSspec/10), axis=1)
        ind1 = np.argmin(abs(fdopnew-1)-2)
        if isspecavg[ind1] < 0:
            isspecavg = isspecavg + 2  # make 1 instead of -1
        if plot:
            # Plot delay-scrunched "power profile"
            if scrunched:
                plt.plot(fdopnew, isspecavg)
                bottom, top = plt.ylim()
                plt.xlabel("Normalised $f_t$")
                plt.ylabel("Mean power (dB)")
                if plot_fit:
                    plt.plot([1, 1], [bottom*0.9, top*1.1], 'r--', alpha=0.5)
                    plt.plot([-1, -1], [bottom*0.9, top*1.1], 'r--', alpha=0.5)
                plt.ylim(bottom*0.9, top*1.1)  # always plot from zero!
                plt.xlim(-maxnormfac, maxnormfac)
                if filename is not None:
                    filename_name = filename.split('.')[0]
                    filename_extension = filename.split('.')[1]
                    plt.savefig(filename_name + '_1d.' + filename_extension,
                                bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                elif display:
                    plt.show()
            # Plot 2D normalised secondary spectrum
            if unscrunched:
                plt.pcolormesh(fdopnew, tdel, normSspec, vmin=vmin, vmax=vmax)
                if lamsteps:
                    plt.ylabel(r'$f_\lambda$ (m$^{-1}$)')
                else:
                    plt.ylabel(r'$f_\nu$ ($\mu$s)')
                bottom, top = plt.ylim()
                plt.xlabel("Normalised $f_t$")
                if plot_fit:
                    plt.plot([1, 1], [bottom, top], 'r--', alpha=0.5)
                    plt.plot([-1, -1], [bottom, top], 'r--', alpha=0.5)
                plt.ylim(bottom, top)
                plt.colorbar()
                if filename is not None:
                    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                elif display:
                    plt.show()
            # plot power spectrum
            if powerspec:
                plt.loglog(np.sqrt(tdel), powerspectrum)
                # Overlay theory
                kf = np.argwhere(np.sqrt(tdel) <= 10)
                amp = np.mean(powerspectrum[kf]*(np.sqrt(tdel[kf]))**3.67)
                plt.loglog(np.sqrt(tdel), amp*(np.sqrt(tdel))**(-3.67))
                if lamsteps:
                    plt.xlabel(r'$f_\lambda^{1/2}$ (m$^{-1/2}$)')
                else:
                    plt.xlabel(r'$f_\nu^{1/2}$ ($\mu$s$^{1/2}$)')
                plt.ylabel("Mean PSD")
                plt.grid(which='both', axis='both')
                if filename is not None:
                    filename_name = filename.split('.')[0]
                    filename_extension = filename.split('.')[1]
                    plt.savefig(filename_name + '_power.' + filename_extension,
                                bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                elif display:
                    plt.show()

        self.normsspecavg = isspecavg
        self.normsspec = normSspec
        self.normsspec_tdel = tdel
        return

    def get_acf_tilt(self, plot=False, tmax=None, fmax=None):
        """
        Estimates the tilt in the ACF, which is proportional to the phase
            gradient parallel to Veff
        """
        if not hasattr(self, 'acf'):
            self.calc_acf()
        if not hasattr(self, 'dnu'):
            self.get_scint_params()

        if tmax is None:
            tmax = self.tau/60/3
        else:
            tmax = tmax
        if fmax is None:
            fmax = self.dnu/5
        else:
            fmax = fmax

        acf = cp(self.acf)
        nr, nc = np.shape(acf)
        t_delays = np.linspace(-self.tobs/60, self.tobs/60, np.shape(acf)[1])
        f_shifts = np.linspace(-self.bw, self.bw, np.shape(acf)[0])

        # just the peak
        xdata_inds = np.argwhere(abs(t_delays) <= tmax)
        xdata = np.array(t_delays[xdata_inds]).squeeze()

        inds = np.argwhere(abs(f_shifts) <= fmax)
        peak_array = []
        peakerr_array = []
        y_array = []

        # Fit parabolas to find the peak in each frequency-slice
        for ii in inds:
            f_shift = f_shifts[ii]
            ind = np.argwhere(f_shifts == f_shift)
            ydata = np.array(acf[ind, xdata_inds]).squeeze()
            yfit, peak, peakerr = fit_parabola(xdata, ydata)
            peak_array.append(peak)
            peakerr_array.append(peakerr)
            y_array.append(f_shift)

        # Now do a weighted fit of a straight line to the peaks
        params, pcov = np.polyfit(peak_array, y_array, 1, cov=True,
                                  w=1/np.array(peakerr_array).squeeze())
        yfit = params[0]*peak_array + params[1]  # y values

        # Get parameter errors
        errors = []
        for i in range(len(params)):  # for each parameter
            errors.append(np.absolute(pcov[i][i])**0.5)

        self.acf_tilt = float(params[0].squeeze())
        self.acf_tilt_err = float(errors[0].squeeze())

        if plot:
            plt.errorbar(peak_array, y_array,
                         xerr=np.array(peakerr_array).squeeze(),
                         marker='.', alpha=0.3)
            plt.plot(peak_array, yfit, alpha=0.5)
            plt.ylabel('Frequency lag (MHz)')
            plt.xlabel('Time lag (mins)')
            plt.title('Peak measurements, and weighted fit')
            plt.show()

            plt.pcolormesh(t_delays, f_shifts, acf)
            plt.plot(peak_array, y_array, 'r', alpha=0.2)
            plt.plot(peak_array, yfit, 'k', alpha=0.2)
            plt.ylabel('Frequency lag (MHz)')
            plt.xlabel('Time lag (mins)')
            plt.title(r'Tilt = {0} $\pm$ {1} (MHz/min)'.format(
                    round(self.acf_tilt, 2), round(self.acf_tilt_err, 1)))
            plt.show()

        return

    def get_scint_params(self, method="acf1d", plot=False, alpha=5/3,
                         mcmc=False, full_frame=False, display=True,
                         nscale=4, nitr=1):
        """
        Measure the scintillation timescale
            Method:
                acf1d - takes a 1D cut through the centre of the ACF for
                sspec - measures timescale from the secondary spectrum
                acf2d - uses an analytic approximation to the ACF including
                    a phase gradient (a shear to the ACF)
        """

        #from lmfit import Minimizer, Parameters
        from lmfit import minimizer, parameter
        import corner

        if not hasattr(self, 'acf'):
            self.calc_acf()
        if not hasattr(self, 'sspec'):
            self.calc_sspec()

        ydata_f = self.acf[int(self.nchan):, int(self.nsub)]
        xdata_f = self.df * np.linspace(0, len(ydata_f), len(ydata_f))
        ydata_t = self.acf[int(self.nchan), int(self.nsub):]
        xdata_t = self.dt * np.linspace(0, len(ydata_t), len(ydata_t))

        nt = len(xdata_t)  # number of t-lag samples
        nf = len(xdata_f)

        # concatenate x and y arrays
        xdata = np.array(np.concatenate((xdata_t, xdata_f)))
        ydata = np.array(np.concatenate((ydata_t, ydata_f)))

        weights = np.ones(np.shape(ydata))

        # Get initial parameter values from 1d fit
        # Estimate amp and white noise level
        wn = max([ydata_f[0]-ydata_f[1], ydata_t[0]-ydata_t[1]])
        amp = max([ydata_f[1], ydata_t[1]])
        # Estimate tau for initial guess. Closest index to 1/e power
        tau = xdata_t[np.argmin(abs(ydata_t - amp/np.e))]
        # Estimate dnu for initial guess. Closest index to 1/2 power
        dnu = xdata_f[np.argmin(abs(ydata_f - amp/2))]

        # Define fit parameters
        print("This is whats up!")
        #params = Parameters()
        params = parameter.Parameters()
        #params = parameter.Parameter('my_param')
        params.add('tau', value=tau, vary=True, min=0.0, max=np.inf)
        params.add('dnu', value=dnu, vary=True, min=0.0, max=np.inf)
        params.add('amp', value=amp, vary=True, min=0.0, max=np.inf)
        params.add('wn', value=wn, vary=True, min=0.0, max=np.inf)
        params.add('nt', value=nt, vary=False)
        params.add('nf', value=nf, vary=False)
        if alpha is None:
            params.add('alpha', value=5/3, vary=True,
                       min=1, max=3)
        else:
            params.add('alpha', value=alpha, vary=False)

        def fitter(scint_model, params, args, mcmc=mcmc, pos=None):
            # Do fit
            func = minimizer.Minimizer(scint_model, params, fcn_args=args)
            results = func.minimize()
            if mcmc:
                func = Minimizer(scint_model, results.params, fcn_args=args)
                print('Doing mcmc posterior sample')
                mcmc_results = func.emcee(nwalkers=nitr, steps=2000,
                                          burn=500, pos=pos, is_weighted=False)
                results = mcmc_results

            return results

        if method == 'acf1d':
            results = fitter(scint_acf_model, params, (xdata, ydata, weights), mcmc=mcmc)

        if method == 'acf2d_approx' or method == 'acf2d':
            results = fitter(scint_acf_model, params, (xdata, ydata, weights), mcmc=False)
            
            params = results.params

            dnu = params['dnu']
            tau = params['tau']

            ntau = nscale
            ndnu = nscale
            while ntau > (self.tobs / tau):
                ntau -= 0.5
                print('Warning: nscale too large for number of sub ints.' +
                      'Decreasing.')
            while ndnu > (self.bw / dnu):
                ndnu -= 0.5
                print('Warning: nscale too large for number of channels. ' +
                      'Decreasing.')

            fmin = int(self.nchan - ndnu * (dnu / self.df))
            fmax = int(self.nchan + ndnu * (dnu / self.df))
            tmin = int(self.nsub - ntau * (tau / self.dt))
            tmax = int(self.nsub + ntau * (tau / self.dt))

            ydata_2d = self.acf[fmin-1:fmax, tmin-1:tmax]
            tticks = np.linspace(-self.tobs, self.tobs, len(self.acf[0, :]))
            tdata = tticks[tmin-1:tmax]
            fticks = np.linspace(-self.bw, self.bw, len(self.acf[:, 0]))
            fdata = fticks[fmin-1:fmax]

            weights_2d = np.ones(np.shape(ydata_2d))

            if method == 'acf2d_approx':

                params.add('tobs', value=self.tobs, vary=False)
                params.add('freq', value=self.freq, vary=False)
                params.add('phasegrad', value=1e-10, vary=True,
                           min=-np.Inf, max=np.Inf)

                results = fitter(scint_acf_model_2d_approx, params,
                                 (tdata, fdata, ydata_2d, weights_2d),
                                 mcmc=mcmc)

            elif method == 'acf2d':

                params.add('tobs', value=np.max(tdata), vary=False)
                params.add('bw', value=np.max(fdata), vary=False)
                params.add('nscale', value=nscale, vary=False)
                params.add('ar', value=1,
                           vary=True, min=1, max=10)
                params.add('phasegrad_x', value=0.01, vary=True,
                           min=-5, max=5)
                params.add('phasegrad_y', value=0.01, vary=True,
                           min=-5, max=5)
                params.add('v_x', value=np.random.normal(loc=0, scale=1),
                           vary=True, min=-10, max=10)
                params.add('v_y', value=np.random.normal(loc=0, scale=1),
                           vary=True, min=-10, max=10)
                #params.add('psi', value=np.random.uniform(low=0, high=180),
                #           vary=True, min=0, max=180)

                plt.imshow(ydata_2d)
                plt.show()

                if mcmc:
                    pos_array = []
                    for itr in range(0, nitr):
                        pos_i = [np.random.normal(loc=params['tau'].value,
                                                  scale=0.2*params['tau'].value),  # tau
                                 np.random.normal(loc=params['dnu'].value,
                                                  scale=0.2*params['dnu'].value),  # dnu
                                 np.random.normal(loc=params['amp'].value,
                                                  scale=0.1*params['amp'].value),  # amp
                                 np.random.normal(loc=params['wn'].value,
                                                  scale=0.1*params['wn'].value),  # wn
                                 # ar
                                 1 + np.abs(np.random.normal(loc=0, scale=2)),
                                 np.random.normal(loc=0, scale=1),  # phs_x
                                 np.random.normal(loc=0, scale=1),  # phs_y
                                 np.random.normal(loc=0, scale=0.1),  # v_ra
                                 np.random.normal(loc=0, scale=0.1),  # v_dec
                                # np.random.uniform(low=0, high=180),  # psi
                                 np.random.uniform(low=0, high=10),  # lnsigma
                                 ]
                        pos_array.append(pos_i)

                    pos = np.array(pos_array)

                results = fitter(scint_acf_model_2d, params,
                                 (ydata_2d, weights_2d), mcmc=mcmc, pos=pos)

        elif method == 'sspec':
            '''
            sspec method
            '''
            print("This method doesn't work yet, do something else")
            # fdyn = np.fft.fft2(self.dyn, (2 * nf, 2 * nt))
            # fdynsq = fdyn * np.conjugate(fdyn)
            #
            # secspec = np.real(fdynsq)
            # #secspec = np.fft.fftshift(fdynsq)
            # #secspec = secspec[nf:2*nf, :]
            # #secspec = np.real(secspec)
            #
            # rowsum = np.sum(secspec[:, :nt], axis=0)
            # ydata_t = rowsum / (2*nf)
            # colsum = np.sum(secspec[:nf, :], axis=1)
            # ydata_f = colsum / (2 * nt)
            #
            # # concatenate x and y arrays
            # xdata = np.array(np.concatenate((xdata_t, xdata_f)))
            # ydata = np.concatenate((ydata_t, ydata_f))
            #
            # weights = np.ones(np.shape(ydata))
            #
            # params = results.params
            # results = fitter(scint_sspec_model, params,
            #                  (xdata, ydata, weights))

        # Done fitting - now define results
        self.tau = results.params['tau'].value
        self.tauerr = results.params['tau'].stderr
        self.dnu = results.params['dnu'].value
        self.dnuerr = results.params['dnu'].stderr
        if method == 'acf2d_approx':
            self.phasegrad = results.params['phasegrad'].value
            self.phasegraderr = results.params['phasegrad'].stderr
        elif method == 'acf2d':
            self.ar = results.params['ar'].value
            self.arerr = results.params['ar'].stderr
            self.phasegrad_x = results.params['phasegrad_x'].value
            self.phasegrad_xerr = results.params['phasegrad_x'].stderr
            self.phasegrad_y = results.params['phasegrad_y'].value
            self.phasegrad_yerr = results.params['phasegrad_y'].stderr
            self.v_x = results.params['v_x'].value
            self.v_xerr = results.params['v_x'].stderr
            self.v_y = results.params['v_y'].value
            self.v_yerr = results.params['v_y'].stderr
            # self.psi = results.params['psi'].value
            # self.psierr = results.params['psi'].stderr
        if alpha is None:
            self.talpha = results.params['alpha'].value
            self.talphaerr = results.params['alpha'].stderr
        else:
            self.talpha = alpha
            self.talphaerr = 0

        print("\t ACF FIT PARAMETERS\n")
        print("tau:\t\t\t{val} +/- {err} s".format(val=self.tau,
              err=self.tauerr))
        print("dnu:\t\t\t{val} +/- {err} MHz".format(val=self.dnu,
              err=self.dnuerr))
        print("alpha:\t\t\t{val} +/- {err}".format(val=self.talpha,
              err=self.talphaerr))
        if method == 'acf2d_approx':
            print("phase grad:\t\t{val} +/- {err}".format(val=self.phasegrad,
                  err=self.phasegraderr))
        elif method == 'acf2d':
            print("ar:\t\t{val} +/- {err}".format(val=self.ar,
                  err=self.arerr))
            print("phase grad x:\t\t{val} +/- {err}".format(
                    val=self.phasegrad_x, err=self.phasegrad_xerr))
            print("phase grad y:\t\t{val} +/- {err}".format(
                    val=self.phasegrad_y, err=self.phasegrad_yerr))
            print("v_ra:\t\t{val} +/- {err}".format(val=self.v_ra,
                  err=self.v_raerr))
            print("v_dec:\t\t{val} +/- {err}".format(val=self.v_dec,
                  err=self.v_decerr))
            print("psi:\t\t{val} +/- {err}".format(val=self.psi,
                  err=self.psierr))

        if plot:
            # get models:
            if method == 'acf1d':
                # Get tau model
                t_residuals = tau_acf_model(results.params, xdata_t, ydata_t,
                                            weights[:nt])
                tmodel = ydata_t - t_residuals/weights[:nt]
                # Get dnu model
                f_residuals = dnu_acf_model(results.params, xdata_f, ydata_f,
                                            weights[nt:])
                fmodel = ydata_f - f_residuals/weights[nt:]

                plt.subplot(2, 1, 1)
                plt.plot(xdata_t, ydata_t)
                plt.plot(xdata_t, tmodel)
                plt.xlabel('Time lag (s)')
                plt.subplot(2, 1, 2)
                plt.plot(xdata_f, ydata_f)
                plt.plot(xdata_f, fmodel)
                plt.xlabel('Frequency lag (MHz)')
                if display:
                    plt.show()

            elif method == 'acf2d_approx' or method == 'acf2d':
                # Get tau model
                if full_frame:
                    ydata = self.acf
                    tdata = tticks
                    fdata = fticks
                    weights = np.ones(np.shape(ydata))
                    if method == 'acf2d_approx':
                        residuals = scint_acf_model_2d_approx(results.params,
                                                              tdata, fdata,
                                                              ydata, weights)
                    else:
                        residuals = scint_acf_model_2d(results.params, ydata,
                                                       weights)

                    model = (ydata - residuals) / weights

                else:
                    ydata = ydata_2d
                    if method == 'acf2d_approx':
                        residuals = scint_acf_model_2d_approx(results.params,
                                                              tdata, fdata,
                                                              ydata,
                                                              weights_2d)
                    else:
                        residuals = scint_acf_model_2d(results.params, ydata,
                                                       weights_2d)

                    model = (ydata - residuals) / weights_2d

                data = [(ydata, 'data'), (model, 'model'),
                        (residuals, 'residuals')]
                for d in data:
                    plt.pcolormesh(tdata/60, fdata, d[0])
                    plt.title(d[1])
                    plt.xlabel('Time lag (mins)')
                    plt.ylabel('Frequency lag (MHz)')
                    if display:
                        plt.show()

            elif method == 'sspec':
                '''
                sspec plotting routine
                '''

            if mcmc:
                corner.corner(results.flatchain,
                              labels=results.var_names,
                              truths=list(results.params.valuesdict().
                                          values()))
                if display:
                    plt.show()

        return results.params

    def cut_dyn(self, tcuts=0, fcuts=0, plot=False, filename=None,
                lamsteps=False, maxfdop=np.inf, figsize=(8, 13), display=True):
        """
        Cuts the dynamic spectrum into tcuts+1 segments in time and
                fcuts+1 segments in frequency
        """

        if filename is not None:
            plt.ioff()  # turn off interactive plotting
        nchan = len(self.freqs)  # re-define in case of trimming
        nsub = len(self.times)
        fnum = np.floor(nchan/(fcuts + 1))
        tnum = np.floor(nsub/(tcuts + 1))
        cutdyn = np.empty(shape=(fcuts+1, tcuts+1, int(fnum), int(tnum)))
        # find the right fft lengths for rows and columns
        nrfft = int(2**(np.ceil(np.log2(int(fnum)))+1)/2)
        ncfft = int(2**(np.ceil(np.log2(int(tnum)))+1))
        cutsspec = np.empty(shape=(fcuts+1, tcuts+1, nrfft, ncfft))
        cutacf = np.empty(shape=(fcuts+1, tcuts+1, 2*int(fnum), 2*int(tnum)))
        plotnum = 1
        for ii in reversed(range(0, fcuts+1)):  # plot from high to low
            for jj in range(0, tcuts+1):
                cutdyn[int(ii)][int(jj)][:][:] =\
                    self.dyn[int(ii*fnum):int((ii+1)*fnum),
                             int(jj*tnum):int((jj+1)*tnum)]
                input_dyn_x = self.times[int(jj*tnum):int((jj+1)*tnum)]
                input_dyn_y = self.freqs[int(ii*fnum):int((ii+1)*fnum)]
                input_sspec_x, input_sspec_y, cutsspec[int(ii)][int(jj)][:][:]\
                    = self.calc_sspec(input_dyn=cutdyn[int(ii)][int(jj)][:][:],
                                      lamsteps=lamsteps)
                cutacf[int(ii)][int(jj)][:][:] \
                    = self.calc_acf(input_dyn=cutdyn[int(ii)][int(jj)][:][:])
                if plot:
                    # Plot dynamic spectra
                    plt.figure(1, figsize=figsize)
                    plt.subplot(fcuts+1, tcuts+1, plotnum)
                    self.plot_dyn(input_dyn=cutdyn[int(ii)][int(jj)][:][:],
                                  input_x=input_dyn_x/60, input_y=input_dyn_y)
                    plt.xlabel('t (mins)')
                    plt.ylabel('f (MHz)')

                    # Plot acf
                    plt.figure(2, figsize=figsize)
                    plt.subplot(fcuts+1, tcuts+1, plotnum)
                    self.plot_acf(input_acf=cutacf[int(ii)][int(jj)][:][:],
                                  input_t=input_dyn_x,
                                  input_f=input_dyn_y)
                    plt.xlabel('t lag (mins)')
                    plt.ylabel('f lag ')

                    # Plot secondary spectra
                    plt.figure(3, figsize=figsize)
                    plt.subplot(fcuts+1, tcuts+1, plotnum)
                    self.plot_sspec(input_sspec=cutsspec[int(ii)]
                                                        [int(jj)][:][:],
                                    input_x=input_sspec_x,
                                    input_y=input_sspec_y, lamsteps=lamsteps,
                                    maxfdop=maxfdop)
                    plt.xlabel(r'$f_t$ (mHz)')
                    if lamsteps:
                        plt.ylabel(r'$f_\lambda$ (m$^{-1}$)')
                    else:
                        plt.ylabel(r'$f_\nu$ ($\mu$s)')
                    plotnum += 1
        if plot:
            plt.figure(1)
            if filename is not None:
                filename_name = filename.split('.')[0]
                filename_extension = filename.split('.')[1]
                plt.savefig(filename_name + '_dynspec.' + filename_extension,
                            figsize=(6, 10), dpi=150, bbox_inches='tight',
                            pad_inches=0.1)
                plt.close()
            elif display:
                plt.show()
            plt.figure(2)
            if filename is not None:
                plt.savefig(filename_name + '_acf.' + filename_extension,
                            figsize=(6, 10), dpi=150, bbox_inches='tight',
                            pad_inches=0.1)
                plt.close()
            elif display:
                plt.show()
            plt.figure(3)
            if filename is not None:
                plt.savefig(filename_name + '_sspec.' + filename_extension,
                            figsize=(6, 10), dpi=150, bbox_inches='tight',
                            pad_inches=0.1)
                plt.close()
            elif display:
                plt.show()
        self.cutdyn = cutdyn
        self.cutsspec = cutsspec

    def trim_edges(self):
        """
        Find and remove the band edges
        """

        rowsum = sum(abs(self.dyn[0][:]))
        # Trim bottom
        while rowsum == 0 or np.isnan(rowsum):
            self.dyn = np.delete(self.dyn, (0), axis=0)
            self.freqs = np.delete(self.freqs, (0))
            rowsum = sum(abs(self.dyn[0][:]))
        rowsum = sum(abs(self.dyn[-1][:]))
        # Trim top
        while rowsum == 0 or np.isnan(rowsum):
            self.dyn = np.delete(self.dyn, (-1), axis=0)
            self.freqs = np.delete(self.freqs, (-1))
            rowsum = sum(abs(self.dyn[-1][:]))
        # Trim left
        colsum = sum(abs(self.dyn[:][0]))
        while colsum == 0 or np.isnan(rowsum):
            self.dyn = np.delete(self.dyn, (0), axis=1)
            self.times = np.delete(self.times, (0))
            colsum = sum(abs(self.dyn[:][0]))
        colsum = sum(abs(self.dyn[:][-1]))
        # Trim right
        while colsum == 0 or np.isnan(rowsum):
            self.dyn = np.delete(self.dyn, (-1), axis=1)
            self.times = np.delete(self.times, (-1))
            colsum = sum(abs(self.dyn[:][-1]))
        self.nchan = len(self.freqs)
        self.bw = round(max(self.freqs) - min(self.freqs) + self.df, 2)
        self.freq = round(np.mean(self.freqs), 2)
        self.nsub = len(self.times)
        self.tobs = round(max(self.times) - min(self.times) + self.dt, 2)
        self.mjd = self.mjd + self.times[0]/86400

    def refill(self, linear=True, zeros=True):
        """
        Replaces the nan values in array. Also replaces zeros by default
        """
        print("Is it that it is mot beimg Refilled?. Looks like it is working.")
        if zeros:
            self.dyn[self.dyn == 0] = np.nan

        if linear:  # do linear interpolation
            array = cp(self.dyn)
            self.dyn = interp_nan_2d(array)

        # Fill remainder with the mean
        meanval = np.mean(self.dyn[is_valid(self.dyn)])
        self.dyn[np.isnan(self.dyn)] = meanval

    def correct_dyn(self, svd=True, nmodes=1, frequency=False, time=True,
                    lamsteps=False, nsmooth=5):
        """
        Correct for gain variations in time and frequency
        """

        if lamsteps:
            if not self.lamsteps:
                self.scale_dyn()
            dyn = self.lamdyn
        else:
            dyn = self.dyn
        dyn[np.isnan(dyn)] = 0

        if svd:
            dyn, model = svd_model(dyn, nmodes=nmodes)
        else:
            if frequency:
                self.bandpass = np.mean(dyn, axis=1)
                # Make sure there are no zeros
                self.bandpass[self.bandpass == 0] = np.mean(self.bandpass)
                if nsmooth is not None:
                    bandpass = savgol_filter(self.bandpass, nsmooth, 1)
                else:
                    bandpass = self.bandpass
                dyn = np.divide(dyn, np.reshape(bandpass,
                                                [len(bandpass), 1]))

            if time:
                timestructure = np.mean(dyn, axis=0)
                # Make sure there are no zeros
                timestructure[timestructure == 0] = np.mean(timestructure)
                if nsmooth is not None:
                    timestructure = savgol_filter(timestructure, nsmooth, 1)
                dyn = np.divide(dyn, np.reshape(timestructure,
                                                [1, len(timestructure)]))

        if lamsteps:
            self.lamdyn = dyn
        else:
            self.dyn = dyn

    def calc_scat_im(self, input_sspec=None, input_eta=None, input_fdop=None,
                     input_tdel=None, sampling=64, lamsteps=False, trap=False,
                     ref_freq=1400, clean=True, s=None, veff=None, d=None,
                     fit_arc=True, plotarc=False, plot_fit=False, plot=False,
                     plot_log=True, use_angle=False, use_spatial=False):
        """
        Calculate the scattered image
        """

        if input_sspec is None:
            if lamsteps:
                if not hasattr(self, 'lamsspec'):
                    self.calc_sspec(lamsteps=lamsteps)
                sspec = cp(self.lamsspec)
            elif trap:
                if not hasattr(self, 'trapsspec'):
                    self.calc_sspec(trap=trap)
                sspec = cp(self.trapsspec)
            else:
                if not hasattr(self, 'sspec'):
                    self.calc_sspec(lamsteps=lamsteps)
                sspec = cp(self.sspec)
            fdop = cp(self.fdop)
            tdel = cp(self.tdel)
        else:
            sspec = input_sspec
            fdop = input_fdop
            tdel = input_tdel

        nf = len(fdop)
        nt = len(tdel)

        sspec = 10**(sspec / 10)

        if input_eta is None and fit_arc:
            if not hasattr(self, 'betaeta') and not hasattr(self, 'eta'):
                self.fit_arc(lamsteps=lamsteps,
                             log_parabola=True, plot=plot_fit)
            if lamsteps:
                c = 299792458.0  # m/s
                beta_to_eta = c * 1e6 / ((ref_freq * 1e6)**2)
                # correct for freq
                eta = self.betaeta / (self.freq / ref_freq)**2
                eta = eta*beta_to_eta
                eta = eta
            else:
                eta = self.eta
        else:
            if input_eta is None:
                eta = tdel[nt-1] / fdop[nf-1]**2
            else:
                eta = input_eta

        if plotarc:
            self.plot_sspec(lamsteps=lamsteps, plotarc=plotarc)

        # crop sspec to desired region
        flim = next(i for i, delay in enumerate(eta * fdop**2) if
                    delay < np.max(tdel))
        if flim == 0:
            tlim = next(i for i, delay in enumerate(tdel) if
                        delay > eta * fdop[0] ** 2)
            sspec = sspec[:tlim, :]
            tdel = fdop[:tlim]
        else:

            sspec = sspec[:, flim-int(0.02*nf):nf-flim+int(0.02*nf)]
            fdop = fdop[flim-int(0.02*nf):nf-flim+int(0.02*nf)]

        if clean:
            # fill infs and extremely small pixel values
            array = cp(sspec)
            x = np.arange(0, array.shape[1])
            y = np.arange(0, array.shape[0])

            # mask invalid values
            array = np.ma.masked_where((array < 1e-22), array)
            xx, yy = np.meshgrid(x, y)

            # get only the valid values
            x1 = xx[~array.mask]
            y1 = yy[~array.mask]
            newarr = np.ravel(array[~array.mask])

            sspec = griddata((x1, y1), newarr, (xx, yy), method='linear')

            # fill nans with the mean
            meanval = np.mean(sspec[is_valid(sspec)])
            sspec[np.isnan(sspec)] = meanval

        max_fd = max(fdop)
        fd_step = max_fd / sampling

        fdop_x = np.arange(-max_fd, max_fd, fd_step)
        fdop_x = np.append(fdop_x, max_fd)
        nx = len(fdop_x)

        fdop_y = np.arange(0, max_fd, fd_step)
        fdop_y = np.append(fdop_y, max_fd)
        ny = len(fdop_y)

        fdop_x_est, fdop_y_est = np.meshgrid(fdop_x, fdop_y)
        fdop_est = fdop_x_est
        tdel_est = (fdop_x_est**2 + fdop_y_est**2) * eta
        fd, td = np.meshgrid(fdop, tdel)

        # 2D interpolation
        interp = RectBivariateSpline(td[:, 0], fd[0], sspec)
        image = interp.ev(tdel_est, fdop_est)

        plt.pcolor(10*np.log10(image))
        plt.show()

        image = image * fdop_y_est
        scat_im = np.zeros((nx, nx))
        scat_im[ny-1:nx, :] = image
        scat_im[0:ny, :] = np.flipud(image[0:ny, :])

        if plot or plot_log:
            self.plot_scat_im(input_scat_im=scat_im, input_fdop=fdop_x,
                              s=s, veff=veff, d=d, use_angle=use_angle,
                              use_spatial=use_spatial, display=True,
                              plot_log=plot_log)

        self.scat_im = scat_im
        self.scat_im_ax = fdop_x

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

#please set prewhite = true
    def calc_Xspec_reconst_v2(self, cal_rcp_dyn, cal_lcp_dyn,
                   prewhite=False, halve=True, log_or_nah = True,
                   phasor='cos',
                   plot=False, real_arg = False, lamsteps=False,
                   input_dyn=None, 
                   input_x=None, 
                   input_y=None, trap=False,
                   window='hamming', window_frac=0.1):
        """
        Calculate Cross Spectrum. add hoc, design from the ground up the cross spectra.Now I will replicate in my original state.
        Do exactly what 
        
        FFT Sizing
        
        Subtract Mean
        
        Windowing
        
        prewhitening
        
        FFT
        
        Multiply
        
        Imag Cross Spectra

        post darkening
        """
        #preprocessing:
        
        #my proof, my truth
        #modify this when you are ready to go.
        #It will change memory kernel crash
               
        l_dyn = cp(cal_lcp_dyn.lamdyn)
        r_dyn = cp(cal_rcp_dyn.lamdyn)

        nf = np.shape(cal_rcp_dyn.lamdyn)[0]     #get shapes. Same size
        nt = np.shape(cal_rcp_dyn.lamdyn)[1]

        l_dyn = l_dyn - np.mean(l_dyn)  # subtract mean. Change this to lamdyn
        r_dyn = r_dyn - np.mean(r_dyn) 

        cw = np.hamming(np.floor(window_frac*nt))  #windowing
        sw = np.hamming(np.floor(window_frac*nf))
        chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),np.ones([nt-len(cw)]))
        subint_window = np.insert(sw, int(np.ceil(len(sw)/2)),np.ones([nf-len(sw)]))
        l_dyn = np.multiply(chan_window, l_dyn)
        l_dyn  = np.transpose(np.multiply(subint_window,np.transpose(l_dyn )))

        cw = np.hamming(np.floor(window_frac*nt))  #windowing
        sw = np.hamming(np.floor(window_frac*nf))
        chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),np.ones([nt-len(cw)]))
        subint_window = np.insert(sw, int(np.ceil(len(sw)/2)),np.ones([nf-len(sw)]))
        r_dyn = np.multiply(chan_window, r_dyn)
        r_dyn = np.transpose(np.multiply(subint_window,np.transpose(r_dyn)))

        nrfft = int(2**(np.ceil(np.log2(nf))+1))   #n rows and n columns fft
        ncfft = int(2**(np.ceil(np.log2(nt))+1))

        l_dyn = l_dyn - np.mean(l_dyn)  # subtract mean
        r_dyn = r_dyn - np.mean(r_dyn)  # subtract mean
        
        
        if prewhite: #prewhite is always true, prewhite before fft
            simpw = convolve2d([[1, -1], [-1, 1]], l_dyn, mode='valid')
        else:
            simpw = l_dyn
            
        if prewhite: #prewhite is always true, prewhite before fft
            simpw = convolve2d([[1, -1], [-1, 1]], r_dyn, mode='valid')
        else:
            simpw = r_dyn
        
        
        rcp_ft = np.fft.fft2(r_dyn, s=[nrfft, ncfft])
        lcp_ft = np.fft.fft2(l_dyn, s=[nrfft, ncfft])
        
        lcp_ft = np.conjugate(lcp_ft)
        

        #How we sampled lam sspec 3. Its the same as this
        sec_lcp_ft = np.fft.fftshift(lcp_ft)  
        sec_lcp_ft = sec_lcp_ft[int(nrfft/2):][:]  # crop #       
        
        sec_rcp_ft = np.fft.fftshift(rcp_ft)  
        sec_rcp_ft = sec_rcp_ft[int(nrfft/2):][:]  # crop #    
        

        if halve:
            td = np.array(list(range(0, int(nrfft/2))))
        else:
            td = np.array(list(range(0, int(nrfft))))

        fd = np.array(list(range(int(-ncfft/2), int(ncfft/2))))
        fdop = np.reshape(np.multiply(fd, 1e3/(ncfft*self.dt)),[len(fd)])  # in mHz
        tdel = np.reshape(np.divide(td, (nrfft*self.df)),[len(td)])  # in us

        if lamsteps:
            beta = np.divide(td, (nrfft*self.dlam))  # in m^-1   
        
        #now we multiply
        
        cross_prod = np.multiply(sec_lcp_ft , sec_rcp_ft ) 
        
        print("cross_p is here. It is complex.")
        print(cross_prod)
        
        if(real_arg):
            print("Taking Real")
            sec = cross_prod.real
        else:
            print("Taking Imag")
            sec = cross_prod.imag
        
        #this is post darken
        vec1 = np.reshape(np.power(np.sin(np.multiply(sc.pi/ncfft, fd)), 2),
                                  [ncfft, 1])
        vec2 = np.reshape(np.power(np.sin(np.multiply(sc.pi/nrfft, td)), 2),
                                  [1, int(nrfft/2)])
        postdark = np.transpose(vec1*vec2)
        postdark[:, int(ncfft/2)] = 1
        postdark[0, :] = 1
        sec = np.divide(sec, postdark)
        
        self.pre_log_cross_sspec_the_truth = sec
        
        #sec = log_multi*np.log10(sec)
        
        #RR = np.real(np.multiply(rcp_ft,np.conj(rcp_ft)))
        #LL = np.real(np.multiply(lcp_ft,np.conj(lcp_ft)))
        
        #rcp_ft = np.fft.fft2 (cal_rcp_dyn.lamdyn)
        #lcp_ft = np.fft.fft2 (cal_lcp_dyn.lamdyn)

        #shift and drop delays
        #sec_LL = np.fft.fftshift(LL)
        #sec_LL = sec_LL[int(nrfft/2):][:]

        #sec_RR = np.fft.fftshift(RR)
        #sec_RR = sec_RR[int(nrfft/2):][:]

        
        #sec_cross_p = np.fft.fftshift(cross_p)
        #sec_cross_p = sec_cross_p[int(nrfft/2):][:]

        #cross_p = np.multiply(lcp_ft,np.conj(rcp_ft))
        #LL = np.real(np.multiply(lcp_ft,np.conj(lcp_ft)))
        #RR = np.real(np.multiply(rcp_ft,np.conj(rcp_ft)))
        #imag_cross_p = np.imag(sec_cross_p)
        
        #norm = np.sqrt(np.multiply(sec_LL,sec_RR))
        #cross_cos = np.divide(np.real(sec_cross_p),norm)
        #cross_sin = np.divide(np.imag(sec_cross_p),norm)
        
                         
            
        print("These things are necessary")
        print("halve:", halve)
        print("fdop:",fdop)
        print("tdel:",tdel)
        print("beta:",beta)

        if log_or_nah:
            print("Make log!")
            sec  = 10*np.log(sec)            
            #sec = 10*np.log10(sec)
            
        #please set prewhite to true...
        #norm = np.sqrt (np.multiply(LL,RR))
        #cross_cos = np.divide(np.real(cross_p),norm)
        #cross_sin = np.divide(np.imag(cross_p),norm)
        #print("Lamdyn Expectation verified")
        """
        plt.matshow(cross_cos)
        plt.title("Real Part")
        plt.colorbar()
        plt.savefig('./sanity_check/Real_enhanced.png')
        plt.show()

        plt.matshow(cross_sin)
        plt.title("Imag Part")
        plt.colorbar()
        plt.savefig('./sanity_check/Imag_enhanced.png')
        plt.show()
        
        print("Real part Mean:", np.mean(cross_cos), "median: ", np.median(cross_cos), "\nvariance",  np.var(cross_cos),
              "standard deviation:",np.std(cross_cos), "Shape:", cross_cos.shape)

        print("\nImag part Mean:", np.mean(cross_sin), "median: ", np.median(cross_sin), "\nvariance",  np.var(cross_sin),
              "standard deviation:",np.std(cross_sin), "Shape:", cross_sin.shape)
        """
        
        #what is sec?
        if (phasor == 'cos'):
            print('selecting cos as phasor')
            sec = cross_cos
        elif (phasor == 'sin'):
            print('selecting sin as phasor')
            sec = cross_sin
        elif (phasor == 'cross'): #send the arguement cross
            print('selecting imag cross spec')
        else:
            print("Error. No correct phasor.")
        #sec = 
        #lamstep sampling lamstep sampling does change the size, but does not change the statistics
        #Windowing
        #Normalization
        #prewhitening
        #Fourier transform
        #Crop post darken
        #log scale
        
        #somewhere along the way, the statistics get screwed up
        
        #print("\nCalculating Secondary Spectrum")
        
        #what needs to get sent out?
        #sec
        #fdop
        #tdel
        #beta
        
        #This saves the attribute
        if input_dyn is None:
            if lamsteps:
                self.lamsspec = sec
            elif trap:
                self.trapsspec = sec
            else:
                self.sspec = sec

            self.fdop = fdop
            self.tdel = tdel
            if lamsteps:
                self.beta = beta
            if plot:
                self.plot_sspec(lamsteps=lamsteps, trap=trap)
        else:#this does not matter as much
            if lamsteps:
                beta = np.divide(td, (nrfft*self.dlam))  # in m^-1
                yaxis = beta
            else:
                yaxis = tdel
            return fdop, yaxis, sec   



    def calc_Xspec_reconst_v1(self, cal_rcp_dyn, cal_lcp_dyn,
                   prewhite=False, halve=True, log_or_nah = True,
                   phasor='cos',
                   plot=False, lamsteps=False,
                   input_dyn=None, 
                   input_x=None, 
                   input_y=None, trap=False,
                   window='hamming', window_frac=0.1):
        """
        Calculate Cross Spectrum. add hoc, design from the ground up the cross spectra. I tried to recreate my results with the knowledge I have right now. Rengineered.
        
        FFT Sizing
        
        Subtract Mean
        
        Windowing
        
        prewhitening
        
        FFT
        
        Multiply
        
        Imag Cross Spectra

        post darkening
        """
        #preprocessing:
        
        #my proof, my truth
        #modify this when you are ready to go.
        #It will change memory kernel crash
               
        l_dyn = cp(cal_lcp_dyn.lamdyn)
        r_dyn = cp(cal_rcp_dyn.lamdyn)

        nf = np.shape(cal_rcp_dyn.lamdyn)[0]     #get shapes. Same size
        nt = np.shape(cal_rcp_dyn.lamdyn)[1]

        l_dyn = l_dyn - np.mean(l_dyn)  # subtract mean. Change this to lamdyn
        r_dyn = r_dyn - np.mean(r_dyn) 

        cw = np.hamming(np.floor(window_frac*nt))  #windowing
        sw = np.hamming(np.floor(window_frac*nf))
        chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),np.ones([nt-len(cw)]))
        subint_window = np.insert(sw, int(np.ceil(len(sw)/2)),np.ones([nf-len(sw)]))
        l_dyn = np.multiply(chan_window, l_dyn)
        l_dyn  = np.transpose(np.multiply(subint_window,np.transpose(l_dyn )))

        cw = np.hamming(np.floor(window_frac*nt))  #windowing
        sw = np.hamming(np.floor(window_frac*nf))
        chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),np.ones([nt-len(cw)]))
        subint_window = np.insert(sw, int(np.ceil(len(sw)/2)),np.ones([nf-len(sw)]))
        r_dyn = np.multiply(chan_window, r_dyn)
        r_dyn = np.transpose(np.multiply(subint_window,np.transpose(r_dyn)))

        nrfft = int(2**(np.ceil(np.log2(nf))+1))   #n rows and n columns fft
        ncfft = int(2**(np.ceil(np.log2(nt))+1))

        l_dyn = l_dyn - np.mean(l_dyn)  # subtract mean
        r_dyn = r_dyn - np.mean(r_dyn)  # subtract mean
        
        
        if prewhite: #prewhite is always true, prewhite before fft
            simpw = convolve2d([[1, -1], [-1, 1]], l_dyn, mode='valid')
        else:
            simpw = l_dyn
            
        if prewhite: #prewhite is always true, prewhite before fft
            simpw = convolve2d([[1, -1], [-1, 1]], r_dyn, mode='valid')
        else:
            simpw = r_dyn
        
        
        rcp_ft = np.fft.fft2(r_dyn, s=[nrfft, ncfft])
        lcp_ft = np.fft.fft2(l_dyn, s=[nrfft, ncfft])

        cross_p = np.multiply(lcp_ft,np.conj(rcp_ft))
        
        print("cross_p is here. It is complex.")
        print(cross_p)
        
        #RR = np.real(np.multiply(rcp_ft,np.conj(rcp_ft)))
        #LL = np.real(np.multiply(lcp_ft,np.conj(lcp_ft)))
        #rcp_ft = np.fft.fft2 (cal_rcp_dyn.lamdyn)
        #lcp_ft = np.fft.fft2 (cal_lcp_dyn.lamdyn)

        #shift and drop delays
        #sec_LL = np.fft.fftshift(LL)
        #sec_LL = sec_LL[int(nrfft/2):][:]

        #sec_RR = np.fft.fftshift(RR)
        #sec_RR = sec_RR[int(nrfft/2):][:]

        
        sec_cross_p = np.fft.fftshift(cross_p)
        sec_cross_p = sec_cross_p[int(nrfft/2):][:]

        #cross_p = np.multiply(lcp_ft,np.conj(rcp_ft))
        #LL = np.real(np.multiply(lcp_ft,np.conj(lcp_ft)))
        #RR = np.real(np.multiply(rcp_ft,np.conj(rcp_ft)))
        imag_cross_p = np.imag(sec_cross_p)
        
        #norm = np.sqrt(np.multiply(sec_LL,sec_RR))
        #cross_cos = np.divide(np.real(sec_cross_p),norm)
        #cross_sin = np.divide(np.imag(sec_cross_p),norm)
        
        if halve:
            td = np.array(list(range(0, int(nrfft/2))))
        else:
            td = np.array(list(range(0, int(nrfft))))

        fd = np.array(list(range(int(-ncfft/2), int(ncfft/2))))
        fdop = np.reshape(np.multiply(fd, 1e3/(ncfft*self.dt)),[len(fd)])  # in mHz
        tdel = np.reshape(np.divide(td, (nrfft*self.df)),[len(td)])  # in us

        if lamsteps:
            beta = np.divide(td, (nrfft*self.dlam))  # in m^-1      
            
        if prewhite:  # Now post-darken. after beta
            if halve:
                vec1 = np.reshape(np.power(np.sin(
                                  np.multiply(sc.pi/ncfft, fd)), 2),
                                  [ncfft, 1])
                vec2 = np.reshape(np.power(np.sin(
                                  np.multiply(sc.pi/nrfft, td)), 2),
                                  [1, int(nrfft/2)])
                postdark = np.transpose(vec1*vec2)
                postdark[:, int(ncfft/2)] = 1
                postdark[0, :] = 1
                
                imag_cross_p = np.divide(imag_cross_p, postdark)
            else:
                raise RuntimeError('Cannot apply prewhite to full frame')

        print("These things are necessary")
        print("halve:", halve)
        print("fdop:",fdop)
        print("tdel:",tdel)
        print("beta:",beta)

        if log_or_nah:
            print("Make log!")
            imag_cross_p = 10*np.log(imag_cross_p)            
            #sec = 10*np.log10(sec)
            
        #please set prewhite to true...
        #norm = np.sqrt (np.multiply(LL,RR))
        #cross_cos = np.divide(np.real(cross_p),norm)
        #cross_sin = np.divide(np.imag(cross_p),norm)
        #print("Lamdyn Expectation verified")
        """
        plt.matshow(cross_cos)
        plt.title("Real Part")
        plt.colorbar()
        plt.savefig('./sanity_check/Real_enhanced.png')
        plt.show()

        plt.matshow(cross_sin)
        plt.title("Imag Part")
        plt.colorbar()
        plt.savefig('./sanity_check/Imag_enhanced.png')
        plt.show()
        
        print("Real part Mean:", np.mean(cross_cos), "median: ", np.median(cross_cos), "\nvariance",  np.var(cross_cos),
              "standard deviation:",np.std(cross_cos), "Shape:", cross_cos.shape)

        print("\nImag part Mean:", np.mean(cross_sin), "median: ", np.median(cross_sin), "\nvariance",  np.var(cross_sin),
              "standard deviation:",np.std(cross_sin), "Shape:", cross_sin.shape)
        """
        
        #what is sec?
        if (phasor == 'cos'):
            print('selecting cos as phasor')
            sec = cross_cos
        elif (phasor == 'sin'):
            print('selecting sin as phasor')
            sec = cross_sin
        elif (phasor == 'cross'): #send the arguement cross
            print('selecting imag cross spec')
            sec = imag_cross_p
        else:
            print("Error. No correct phasor.")
        #sec = 
        #lamstep sampling lamstep sampling does change the size, but does not change the statistics
        #Windowing
        #Normalization
        #prewhitening
        #Fourier transform
        #Crop post darken
        #log scale
        
        #somewhere along the way, the statistics get screwed up
        
        #print("\nCalculating Secondary Spectrum")
        
        #what needs to get sent out?
        #sec
        #fdop
        #tdel
        #beta
        
        #This saves the attribute
        if input_dyn is None:
            if lamsteps:
                self.lamsspec = sec
            elif trap:
                self.trapsspec = sec
            else:
                self.sspec = sec

            self.fdop = fdop
            self.tdel = tdel
            if lamsteps:
                self.beta = beta
            if plot:
                self.plot_sspec(lamsteps=lamsteps, trap=trap)
        else:#this does not matter as much
            if lamsteps:
                beta = np.divide(td, (nrfft*self.dlam))  # in m^-1
                yaxis = beta
            else:
                yaxis = tdel
            return fdop, yaxis, sec   

    def calc_Xspec_finale(self, cal_rcp_dyn, cal_lcp_dyn,
                   prewhite=False, halve=True,
                   phasor='cos',
                   plot=False, lamsteps=False,
                   input_dyn=None, 
                   input_x=None, 
                   input_y=None, trap=False,
                   window='hamming', window_frac=0.1):
        """
        Calculate Cross Spectrum
        
        FFT Sizing
        
        Subtract Mean
        
        Windowing
        
        FFT
        
        Multiply
        
        Imag Cross Spectra
        """
        #preprocessing:
        
        #my proof, my truth
        #modify this when you are ready to go.
        #It will change memory kernel crash
               
        l_dyn = cp(cal_lcp_dyn.lamdyn)
        r_dyn = cp(cal_rcp_dyn.lamdyn)

        nf = np.shape(cal_rcp_dyn.lamdyn)[0]     #get shapes. Same size
        nt = np.shape(cal_rcp_dyn.lamdyn)[1]

        l_dyn = l_dyn - np.mean(l_dyn)  # subtract mean. Change this to lamdyn
        r_dyn = r_dyn - np.mean(r_dyn) 

        cw = np.hamming(np.floor(window_frac*nt))  #windowing
        sw = np.hamming(np.floor(window_frac*nf))
        chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),np.ones([nt-len(cw)]))
        subint_window = np.insert(sw, int(np.ceil(len(sw)/2)),np.ones([nf-len(sw)]))
        l_dyn = np.multiply(chan_window, l_dyn)
        l_dyn  = np.transpose(np.multiply(subint_window,np.transpose(l_dyn )))

        cw = np.hamming(np.floor(window_frac*nt))  #windowing
        sw = np.hamming(np.floor(window_frac*nf))
        chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),np.ones([nt-len(cw)]))
        subint_window = np.insert(sw, int(np.ceil(len(sw)/2)),np.ones([nf-len(sw)]))
        r_dyn = np.multiply(chan_window, r_dyn)
        r_dyn = np.transpose(np.multiply(subint_window,np.transpose(r_dyn)))

        nrfft = int(2**(np.ceil(np.log2(nf))+1))   #n rows and n columns fft
        ncfft = int(2**(np.ceil(np.log2(nt))+1))

        l_dyn = l_dyn - np.mean(l_dyn)  # subtract mean
        r_dyn = r_dyn - np.mean(r_dyn)  # subtract mean
        
        rcp_ft = np.fft.fft2(r_dyn, s=[nrfft, ncfft])
        lcp_ft = np.fft.fft2(l_dyn, s=[nrfft, ncfft])

        cross_p = np.multiply(lcp_ft,np.conj(rcp_ft))
        
        print("cross_p is here. It is complex.")
        print(cross_p)
        
        RR = np.real(np.multiply(rcp_ft,np.conj(rcp_ft)))
        LL = np.real(np.multiply(lcp_ft,np.conj(lcp_ft)))
        #rcp_ft = np.fft.fft2 (cal_rcp_dyn.lamdyn)
        #lcp_ft = np.fft.fft2 (cal_lcp_dyn.lamdyn)

        #shift and drop delays
        sec_LL = np.fft.fftshift(LL)
        sec_LL = sec_LL[int(nrfft/2):][:]

        sec_RR = np.fft.fftshift(RR)
        sec_RR = sec_RR[int(nrfft/2):][:]

        sec_cross_p = np.fft.fftshift(cross_p)
        sec_cross_p = sec_cross_p[int(nrfft/2):][:]

        #cross_p = np.multiply(lcp_ft,np.conj(rcp_ft))
        #LL = np.real(np.multiply(lcp_ft,np.conj(lcp_ft)))
        #RR = np.real(np.multiply(rcp_ft,np.conj(rcp_ft)))
        imag_cross_p = np.imag(sec_cross_p)
        
        #norm = np.sqrt(np.multiply(sec_LL,sec_RR))
        #cross_cos = np.divide(np.real(sec_cross_p),norm)
        #cross_sin = np.divide(np.imag(sec_cross_p),norm)
        
        if halve:
            td = np.array(list(range(0, int(nrfft/2))))
        else:
            td = np.array(list(range(0, int(nrfft))))

        fd = np.array(list(range(int(-ncfft/2), int(ncfft/2))))
        fdop = np.reshape(np.multiply(fd, 1e3/(ncfft*self.dt)),[len(fd)])  # in mHz
        tdel = np.reshape(np.divide(td, (nrfft*self.df)),[len(td)])  # in us

        if lamsteps:
            beta = np.divide(td, (nrfft*self.dlam))  # in m^-1       

        print("These things are necessary")
        print("halve:", halve)
        print("fdop:",fdop)
        print("tdel:",tdel)
        print("beta:",beta)
            
        #norm = np.sqrt (np.multiply(LL,RR))
        #cross_cos = np.divide(np.real(cross_p),norm)
        #cross_sin = np.divide(np.imag(cross_p),norm)
        #print("Lamdyn Expectation verified")
        """
        plt.matshow(cross_cos)
        plt.title("Real Part")
        plt.colorbar()
        plt.savefig('./sanity_check/Real_enhanced.png')
        plt.show()

        plt.matshow(cross_sin)
        plt.title("Imag Part")
        plt.colorbar()
        plt.savefig('./sanity_check/Imag_enhanced.png')
        plt.show()
        
        print("Real part Mean:", np.mean(cross_cos), "median: ", np.median(cross_cos), "\nvariance",  np.var(cross_cos),
              "standard deviation:",np.std(cross_cos), "Shape:", cross_cos.shape)

        print("\nImag part Mean:", np.mean(cross_sin), "median: ", np.median(cross_sin), "\nvariance",  np.var(cross_sin),
              "standard deviation:",np.std(cross_sin), "Shape:", cross_sin.shape)
        """
        
        #what is sec?
        if (phasor == 'cos'):
            print('selecting cos as phasor')
            sec = cross_cos
        elif (phasor == 'sin'):
            print('selecting sin as phasor')
            sec = cross_sin
        elif (phasor == 'cross'): #send the arguement cross
            print('selecting imag cross spec')
            sec = imag_cross_p
        else:
            print("Error. No correct phasor.")
        #sec = 
        #lamstep sampling lamstep sampling does change the size, but does not change the statistics
        #Windowing
        #Normalization
        #prewhitening
        #Fourier transform
        #Crop post darken
        #log scale
        
        #somewhere along the way, the statistics get screwed up
        
        #print("\nCalculating Secondary Spectrum")
        
        #what needs to get sent out?
        #sec
        #fdop
        #tdel
        #beta
        
        #This saves the attribute
        if input_dyn is None:
            if lamsteps:
                self.lamsspec = sec
            elif trap:
                self.trapsspec = sec
            else:
                self.sspec = sec

            self.fdop = fdop
            self.tdel = tdel
            if lamsteps:
                self.beta = beta
            if plot:
                self.plot_sspec(lamsteps=lamsteps, trap=trap)
        else:#this does not matter as much
            if lamsteps:
                beta = np.divide(td, (nrfft*self.dlam))  # in m^-1
                yaxis = beta
            else:
                yaxis = tdel
            return fdop, yaxis, sec
        
#The final move, is to rebuild your technology. To prove that it works in one go. and that it produces the same result

    def calc_Xspec_finale_raw(self, cal_rcp_dyn, cal_lcp_dyn,
                   prewhite=False, halve=True,
                   phasor='cos',
                   plot=False, lamsteps=False,
                   input_dyn=None, 
                   input_x=None, 
                   input_y=None, trap=False,
                   window='hamming', window_frac=0.1):
        """
        Calculate secondary cross spectrum, no frills at all
        
        Sizing nf, nt, nrfft, ncfft
        
        FFT
        
        Multiply
        
        Imag Cross Spectra
        """
        #preprocessing:
        
        #my proof, my truth
        #modify this when you are ready to go.
        #It will change memory kernel crash
               
        l_dyn = cp(cal_lcp_dyn.lamdyn)
        r_dyn = cp(cal_rcp_dyn.lamdyn)

        nf = np.shape(cal_rcp_dyn.lamdyn)[0]     #get shapes. Same size
        nt = np.shape(cal_rcp_dyn.lamdyn)[1]
        
        '''
        l_dyn = l_dyn - np.mean(l_dyn)  # subtract mean. Change this to lamdyn
        r_dyn = r_dyn - np.mean(r_dyn) 

        cw = np.hamming(np.floor(window_frac*nt))  #windowing
        sw = np.hamming(np.floor(window_frac*nf))
        chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),np.ones([nt-len(cw)]))
        subint_window = np.insert(sw, int(np.ceil(len(sw)/2)),np.ones([nf-len(sw)]))
        l_dyn = np.multiply(chan_window, l_dyn)
        l_dyn  = np.transpose(np.multiply(subint_window,np.transpose(l_dyn )))

        cw = np.hamming(np.floor(window_frac*nt))  #windowing
        sw = np.hamming(np.floor(window_frac*nf))
        chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),np.ones([nt-len(cw)]))
        subint_window = np.insert(sw, int(np.ceil(len(sw)/2)),np.ones([nf-len(sw)]))
        r_dyn = np.multiply(chan_window, r_dyn)
        r_dyn = np.transpose(np.multiply(subint_window,np.transpose(r_dyn)))
        '''

        nrfft = int(2**(np.ceil(np.log2(nf))+1))   #n rows and n columns fft
        ncfft = int(2**(np.ceil(np.log2(nt))+1))

        #l_dyn = l_dyn - np.mean(l_dyn)  # subtract mean
        #r_dyn = r_dyn - np.mean(r_dyn)  # subtract mean
        
        rcp_ft = np.fft.fft2(r_dyn, s=[nrfft, ncfft])
        lcp_ft = np.fft.fft2(l_dyn, s=[nrfft, ncfft])

        cross_p = np.multiply(lcp_ft,np.conj(rcp_ft))
        
        print("cross_p is here. It is complex.")
        print(cross_p)
        
        RR = np.real(np.multiply(rcp_ft,np.conj(rcp_ft)))
        LL = np.real(np.multiply(lcp_ft,np.conj(lcp_ft)))
        #rcp_ft = np.fft.fft2 (cal_rcp_dyn.lamdyn)
        #lcp_ft = np.fft.fft2 (cal_lcp_dyn.lamdyn)

        #shift and drop delays
        sec_LL = np.fft.fftshift(LL)
        sec_LL = sec_LL[int(nrfft/2):][:]

        sec_RR = np.fft.fftshift(RR)
        sec_RR = sec_RR[int(nrfft/2):][:]

        sec_cross_p = np.fft.fftshift(cross_p)
        sec_cross_p = sec_cross_p[int(nrfft/2):][:]

        #cross_p = np.multiply(lcp_ft,np.conj(rcp_ft))
        #LL = np.real(np.multiply(lcp_ft,np.conj(lcp_ft)))
        #RR = np.real(np.multiply(rcp_ft,np.conj(rcp_ft)))
        imag_cross_p_raw = np.imag(sec_cross_p)
        
        #norm = np.sqrt(np.multiply(sec_LL,sec_RR))
        #cross_cos = np.divide(np.real(sec_cross_p),norm)
        #cross_sin = np.divide(np.imag(sec_cross_p),norm)
        
        if halve:
            td = np.array(list(range(0, int(nrfft/2))))
        else:
            td = np.array(list(range(0, int(nrfft))))

        fd = np.array(list(range(int(-ncfft/2), int(ncfft/2))))
        fdop = np.reshape(np.multiply(fd, 1e3/(ncfft*self.dt)),[len(fd)])  # in mHz
        tdel = np.reshape(np.divide(td, (nrfft*self.df)),[len(td)])  # in us

        if lamsteps:
            beta = np.divide(td, (nrfft*self.dlam))  # in m^-1       

        print("These things are necessary")
        print("halve:", halve)
        print("fdop:",fdop)
        print("tdel:",tdel)
        print("beta:",beta)
            
        #norm = np.sqrt (np.multiply(LL,RR))
        #cross_cos = np.divide(np.real(cross_p),norm)
        #cross_sin = np.divide(np.imag(cross_p),norm)
        #print("Lamdyn Expectation verified")

        plt.matshow(imag_cross_p_raw)
        plt.title("Real Part")
        plt.colorbar()
        plt.savefig('./sanity_check/raw_cross_prod.png')
        plt.show()
        
        """
        plt.matshow(cross_sin)
        plt.title("Imag Part")
        plt.colorbar()
        plt.savefig('./sanity_check/Imag_enhanced.png')
        plt.show()
        
        print("Real part Mean:", np.mean(cross_cos), "median: ", np.median(cross_cos), "\nvariance",  np.var(cross_cos),
              "standard deviation:",np.std(cross_cos), "Shape:", cross_cos.shape)

        print("\nImag part Mean:", np.mean(cross_sin), "median: ", np.median(cross_sin), "\nvariance",  np.var(cross_sin),
              "standard deviation:",np.std(cross_sin), "Shape:", cross_sin.shape)
        """
        
        #what is sec?
        if (phasor == 'cos'):
            print('selecting cos as phasor')
            sec = cross_cos
        elif (phasor == 'sin'):
            print('selecting sin as phasor')
            sec = cross_sin
        elif (phasor == 'cross'): #send the arguement cross
            print('selecting imag cross spec')
            sec = imag_cross_p_raw
        else:
            print("Error. No correct phasor.")
        #sec = 
        #lamstep sampling lamstep sampling does change the size, but does not change the statistics
        #Windowing
        #Normalization
        #prewhitening
        #Fourier transform
        #Crop post darken
        #log scale
        
        #somewhere along the way, the statistics get screwed up
        
        #print("\nCalculating Secondary Spectrum")
        
        #what needs to get sent out?
        #sec
        #fdop
        #tdel
        #beta
        
        #This saves the attribute
        if input_dyn is None:
            if lamsteps:
                self.lamsspec = sec
            elif trap:
                self.trapsspec = sec
            else:
                self.sspec = sec

            self.fdop = fdop
            self.tdel = tdel
            if lamsteps:
                self.beta = beta
            if plot:
                self.plot_sspec(lamsteps=lamsteps, trap=trap)
        else:#this does not matter as much
            if lamsteps:
                beta = np.divide(td, (nrfft*self.dlam))  # in m^-1
                yaxis = beta
            else:
                yaxis = tdel
            return fdop, yaxis, sec
        
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

    def calc_UL_sspec_enhanced_prewhite_postdark(self, cal_rcp_dyn, cal_lcp_dyn,
                   prewhite=False, halve=True,
                   phasor='cos',
                   plot=False, lamsteps=False,
                   input_dyn=None, 
                   input_x=None, 
                   input_y=None, trap=False,
                   window='hamming', window_frac=0.1):
        """
        Calculate secondary spectrum
        """
        #preprocessing:
        
        #my proof, my truth
        #modify this when you are ready to go.
        #It will change memory kernel crash
               
        l_dyn = cp(cal_lcp_dyn.lamdyn)
        r_dyn = cp(cal_rcp_dyn.lamdyn)

        nf = np.shape(cal_rcp_dyn.lamdyn)[0]     #get shapes. Same size
        nt = np.shape(cal_rcp_dyn.lamdyn)[1]

        l_dyn = l_dyn - np.mean(l_dyn)  # subtract mean. Change this to lamdyn
        r_dyn = r_dyn - np.mean(r_dyn) 

        cw = np.hamming(np.floor(0.15*nt))  #windowing
        sw = np.hamming(np.floor(0.15*nf))
        chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),np.ones([nt-len(cw)]))
        subint_window = np.insert(sw, int(np.ceil(len(sw)/2)),np.ones([nf-len(sw)]))
        l_dyn = np.multiply(chan_window, l_dyn)
        l_dyn  = np.transpose(np.multiply(subint_window,np.transpose(l_dyn )))

        cw = np.hamming(np.floor(0.15*nt))  #windowing
        sw = np.hamming(np.floor(0.15*nf))
        chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),np.ones([nt-len(cw)]))
        subint_window = np.insert(sw, int(np.ceil(len(sw)/2)),np.ones([nf-len(sw)]))
        r_dyn = np.multiply(chan_window, r_dyn)
        r_dyn = np.transpose(np.multiply(subint_window,np.transpose(r_dyn)))

        nrfft = int(2**(np.ceil(np.log2(nf))+1))   #n rows and n columns fft
        ncfft = int(2**(np.ceil(np.log2(nt))+1))

        l_dyn = l_dyn - np.mean(l_dyn)  # subtract mean
        r_dyn = r_dyn - np.mean(r_dyn)  # subtract mean

        #Prewhiten l_dyn
        if prewhite: #prewhite
            print("Prewhitening")
            l_dyn = convolve2d([[1, -1], [-1, 1]], l_dyn, mode='valid')
            r_dyn = convolve2d([[1, -1], [-1, 1]], r_dyn, mode='valid')
        else:
            simpw = dyn
        #Prewhiten r_dyn
        
        rcp_ft = np.fft.fft2(r_dyn, s=[nrfft, ncfft])
        lcp_ft = np.fft.fft2(l_dyn, s=[nrfft, ncfft])

        cross_p = np.multiply(lcp_ft,np.conj(rcp_ft))
        RR = np.real(np.multiply(rcp_ft,np.conj(rcp_ft)))
        LL = np.real(np.multiply(lcp_ft,np.conj(lcp_ft)))
        #rcp_ft = np.fft.fft2 (cal_rcp_dyn.lamdyn)
        #lcp_ft = np.fft.fft2 (cal_lcp_dyn.lamdyn)

        #shift and drop delays
        sec_LL = np.fft.fftshift(LL)
        sec_LL = sec_LL[int(nrfft/2):][:]

        sec_RR = np.fft.fftshift(RR)
        sec_RR = sec_RR[int(nrfft/2):][:]

        sec_cross_p = np.fft.fftshift(cross_p)
        sec_cross_p = sec_cross_p[int(nrfft/2):][:]

        #cross_p = np.multiply(lcp_ft,np.conj(rcp_ft))
        #LL = np.real(np.multiply(lcp_ft,np.conj(lcp_ft)))
        #RR = np.real(np.multiply(rcp_ft,np.conj(rcp_ft)))

        norm = np.sqrt(np.multiply(sec_LL,sec_RR))
        cross_cos = np.divide(np.real(sec_cross_p),norm)
        cross_sin = np.divide(np.imag(sec_cross_p),norm)
        
        if halve:
            td = np.array(list(range(0, int(nrfft/2))))
        else:
            td = np.array(list(range(0, int(nrfft))))

        fd = np.array(list(range(int(-ncfft/2), int(ncfft/2))))
        fdop = np.reshape(np.multiply(fd, 1e3/(ncfft*self.dt)),[len(fd)])  # in mHz
        tdel = np.reshape(np.divide(td, (nrfft*self.df)),[len(td)])  # in us

        if lamsteps:
            beta = np.divide(td, (nrfft*self.dlam))  # in m^-1       

        print("These things are necessary")
        print("halve:", halve)
        print("fdop:",fdop)
        print("tdel:",tdel)
        print("beta:",beta)
            
        #norm = np.sqrt (np.multiply(LL,RR))
        #cross_cos = np.divide(np.real(cross_p),norm)
        #cross_sin = np.divide(np.imag(cross_p),norm)
        #print("Lamdyn Expectation verified")

        plt.matshow(cross_cos)
        plt.title("Real Part")
        plt.colorbar()
        plt.savefig('./sanity_check/Real_enhanced.png')
        plt.show()

        plt.matshow(cross_sin)
        plt.title("Imag Part")
        plt.colorbar()
        plt.savefig('./sanity_check/Imag_enhanced.png')
        plt.show()
        
        print("Real part Mean:", np.mean(cross_cos), "median: ", np.median(cross_cos), "\nvariance",  np.var(cross_cos),
              "standard deviation:",np.std(cross_cos), "Shape:", cross_cos.shape)

        print("\nImag part Mean:", np.mean(cross_sin), "median: ", np.median(cross_sin), "\nvariance",  np.var(cross_sin),
              "standard deviation:",np.std(cross_sin), "Shape:", cross_sin.shape)
        
        #what is sec?
        if (phasor == 'cos'):
            print('selecting cos as phasor')
            sec = cross_cos
        elif (phasor == 'sin'):
            print('selecting sin as phasor')
            sec = cross_sin
        else:
            print("Error. No correct phasor.")
        
        print("post darkening")
        vec1 = np.reshape(np.power(np.sin(np.multiply(sc.pi/ncfft, fd)), 2), [ncfft, 1])
        vec2 = np.reshape(np.power(np.sin(np.multiply(sc.pi/nrfft, td)), 2), [1, int(nrfft/2)])
        postdark = np.transpose(vec1*vec2)
        postdark[:, int(ncfft/2)] = 1
        postdark[0, :] = 1
        sec = np.divide(sec, postdark)
        print("post darkening done")
            
        #sec = 
        #lamstep sampling lamstep sampling does change the size, but does not change the statistics
        #Windowing
        #Normalization
        #prewhitening
        #Fourier transform
        #Crop post darken
        #log scale
        
        #somewhere along the way, the statistics get screwed up
        
        #print("\nCalculating Secondary Spectrum")
        
        #what needs to get sent out?
        #sec
        #fdop
        #tdel
        #beta
        
        #This saves the attribute
        if input_dyn is None:
            if lamsteps:
                self.lamsspec = sec
            elif trap:
                self.trapsspec = sec
            else:
                self.sspec = sec

            self.fdop = fdop
            self.tdel = tdel
            if lamsteps:
                self.beta = beta
            if plot:
                self.plot_sspec(lamsteps=lamsteps, trap=trap)
        else:#this does not matter as much
            if lamsteps:
                beta = np.divide(td, (nrfft*self.dlam))  # in m^-1
                yaxis = beta
            else:
                yaxis = tdel
            return fdop, yaxis, sec
        
    def calc_norm_cross_corr_coeff(self, cal_rcp_dyn, cal_lcp_dyn,
                   prewhite=False, halve=True,
                   phasor='cos',
                   plot=False, lamsteps=False,
                   input_dyn=None, 
                   input_x=None, 
                   input_y=None, trap=False,
                   window='hamming', window_frac=0.1):
        """
        Calculate secondary spectrum
        """
        #preprocessing:
        
        #my proof, my truth
        #modify this when you are ready to go.
        #It will change memory kernel crash
               
        l_dyn = cp(cal_lcp_dyn.lamdyn)
        r_dyn = cp(cal_rcp_dyn.lamdyn)

        nf = np.shape(cal_rcp_dyn.lamdyn)[0]     #get shapes. Same size
        nt = np.shape(cal_rcp_dyn.lamdyn)[1]

        l_dyn = l_dyn - np.mean(l_dyn)  # subtract mean. Change this to lamdyn
        r_dyn = r_dyn - np.mean(r_dyn) 

        cw = np.hamming(np.floor(window_frac*nt))  #windowing
        sw = np.hamming(np.floor(window_frac*nf))
        chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),np.ones([nt-len(cw)]))
        subint_window = np.insert(sw, int(np.ceil(len(sw)/2)),np.ones([nf-len(sw)]))
        l_dyn = np.multiply(chan_window, l_dyn)
        l_dyn  = np.transpose(np.multiply(subint_window,np.transpose(l_dyn )))

        cw = np.hamming(np.floor(window_frac*nt))  #windowing
        sw = np.hamming(np.floor(window_frac*nf))
        chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),np.ones([nt-len(cw)]))
        subint_window = np.insert(sw, int(np.ceil(len(sw)/2)),np.ones([nf-len(sw)]))
        r_dyn = np.multiply(chan_window, r_dyn)
        r_dyn = np.transpose(np.multiply(subint_window,np.transpose(r_dyn)))

        nrfft = int(2**(np.ceil(np.log2(nf))+1))   #n rows and n columns fft
        ncfft = int(2**(np.ceil(np.log2(nt))+1))

        l_dyn = l_dyn - np.mean(l_dyn)  # subtract mean
        r_dyn = r_dyn - np.mean(r_dyn)  # subtract mean
        
        rcp_ft = np.fft.fft2(r_dyn, s=[nrfft, ncfft])
        lcp_ft = np.fft.fft2(l_dyn, s=[nrfft, ncfft])

        cross_p = np.multiply(lcp_ft,np.conj(rcp_ft))
        RR = np.real(np.multiply(rcp_ft,np.conj(rcp_ft)))
        LL = np.real(np.multiply(lcp_ft,np.conj(lcp_ft)))
        #rcp_ft = np.fft.fft2 (cal_rcp_dyn.lamdyn)
        #lcp_ft = np.fft.fft2 (cal_lcp_dyn.lamdyn)

        #shift and drop delays
        sec_LL = np.fft.fftshift(LL)
        sec_LL = sec_LL[int(nrfft/2):][:]

        sec_RR = np.fft.fftshift(RR)
        sec_RR = sec_RR[int(nrfft/2):][:]

        sec_cross_p = np.fft.fftshift(cross_p)
        sec_cross_p = sec_cross_p[int(nrfft/2):][:]

        #cross_p = np.multiply(lcp_ft,np.conj(rcp_ft))
        #LL = np.real(np.multiply(lcp_ft,np.conj(lcp_ft)))
        #RR = np.real(np.multiply(rcp_ft,np.conj(rcp_ft)))

        norm = np.sqrt(np.multiply(sec_LL,sec_RR))
        cross_cos = np.divide(np.real(sec_cross_p),norm)
        cross_sin = np.divide(np.imag(sec_cross_p),norm)
        
        if halve:
            td = np.array(list(range(0, int(nrfft/2))))
        else:
            td = np.array(list(range(0, int(nrfft))))

        fd = np.array(list(range(int(-ncfft/2), int(ncfft/2))))
        fdop = np.reshape(np.multiply(fd, 1e3/(ncfft*self.dt)),[len(fd)])  # in mHz
        tdel = np.reshape(np.divide(td, (nrfft*self.df)),[len(td)])  # in us

        if lamsteps:
            beta = np.divide(td, (nrfft*self.dlam))  # in m^-1       

        print("These things are necessary")
        print("halve:", halve)
        print("fdop:",fdop)
        print("tdel:",tdel)
        print("beta:",beta)
            
        #norm = np.sqrt (np.multiply(LL,RR))
        #cross_cos = np.divide(np.real(cross_p),norm)
        #cross_sin = np.divide(np.imag(cross_p),norm)
        #print("Lamdyn Expectation verified")

        plt.matshow(cross_cos)
        plt.title("Real Part")
        plt.colorbar()
        plt.savefig('./sanity_check/Real_enhanced.png')
        plt.show()

        plt.matshow(cross_sin)
        plt.title("Imag Part")
        plt.colorbar()
        plt.savefig('./sanity_check/Imag_enhanced.png')
        plt.show()
        
        print("Real part Mean:", np.mean(cross_cos), "median: ", np.median(cross_cos), "\nvariance",  np.var(cross_cos),
              "standard deviation:",np.std(cross_cos), "Shape:", cross_cos.shape)

        print("\nImag part Mean:", np.mean(cross_sin), "median: ", np.median(cross_sin), "\nvariance",  np.var(cross_sin),
              "standard deviation:",np.std(cross_sin), "Shape:", cross_sin.shape)
        
        #what is sec?
        if (phasor == 'cos'):
            print('selecting cos as phasor')
            sec = cross_cos
        elif (phasor == 'sin'):
            print('selecting sin as phasor')
            sec = cross_sin
        else:
            print("Error. No correct phasor.")
        #sec = 
        #lamstep sampling lamstep sampling does change the size, but does not change the statistics
        #Windowing
        #Normalization
        #prewhitening
        #Fourier transform
        #Crop post darken
        #log scale
        
        #somewhere along the way, the statistics get screwed up
        
        #print("\nCalculating Secondary Spectrum")
        
        #what needs to get sent out?
        #sec
        #fdop
        #tdel
        #beta
        
        #This saves the attribute
        if input_dyn is None:
            if lamsteps:
                self.lamsspec = sec
            elif trap:
                self.trapsspec = sec
            else:
                self.sspec = sec

            self.fdop = fdop
            self.tdel = tdel
            if lamsteps:
                self.beta = beta
            if plot:
                self.plot_sspec(lamsteps=lamsteps, trap=trap)
        else:#this does not matter as much
            if lamsteps:
                beta = np.divide(td, (nrfft*self.dlam))  # in m^-1
                yaxis = beta
            else:
                yaxis = tdel
            return fdop, yaxis, sec

    def calc_UL_sspec(self, cal_rcp_dyn, cal_lcp_dyn,
                   prewhite=False, halve=True,
                   phasor='cos',
                   plot=False, lamsteps=False,
                   input_dyn=None, 
                   input_x=None, 
                   input_y=None, trap=False,
                   window='hamming', window_frac=0.1):
        """
        Calculate secondary spectrum
        """
        #preprocessing:
        
        #my proof, my truth
        #modify this when you are ready to go.

        nf = np.shape(cal_rcp_dyn.lamdyn)[0]     #get shapes. Same size
        nt = np.shape(cal_rcp_dyn.lamdyn)[1]

        nrfft = int(2**(np.ceil(np.log2(nf))+1))   #n rows and n columns fft
        ncfft = int(2**(np.ceil(np.log2(nt))+1))

        rcp_ft = np.fft.fft2(cal_rcp_dyn.lamdyn, s=[nrfft, ncfft])
        lcp_ft = np.fft.fft2(cal_lcp_dyn.lamdyn, s=[nrfft, ncfft])

        cross_p = np.multiply(lcp_ft,np.conj(rcp_ft))
        RR = np.real(np.multiply(rcp_ft,np.conj(rcp_ft)))
        LL = np.real(np.multiply(lcp_ft,np.conj(lcp_ft)))
        #rcp_ft = np.fft.fft2 (cal_rcp_dyn.lamdyn)
        #lcp_ft = np.fft.fft2 (cal_lcp_dyn.lamdyn)

        #shift and drop delays
        sec_LL = np.fft.fftshift(LL)
        sec_LL = sec_LL[int(nrfft/2):][:]

        sec_RR = np.fft.fftshift(RR)
        sec_RR = sec_RR[int(nrfft/2):][:]

        sec_cross_p = np.fft.fftshift(cross_p)
        sec_cross_p = sec_cross_p[int(nrfft/2):][:]

        #cross_p = np.multiply(lcp_ft,np.conj(rcp_ft))
        #LL = np.real(np.multiply(lcp_ft,np.conj(lcp_ft)))
        #RR = np.real(np.multiply(rcp_ft,np.conj(rcp_ft)))

        norm = np.sqrt(np.multiply(sec_LL,sec_RR))
        cross_cos = np.divide(np.real(sec_cross_p),norm)
        cross_sin = np.divide(np.imag(sec_cross_p),norm)
        
        if halve:
            td = np.array(list(range(0, int(nrfft/2))))
        else:
            td = np.array(list(range(0, int(nrfft))))

        fd = np.array(list(range(int(-ncfft/2), int(ncfft/2))))
        fdop = np.reshape(np.multiply(fd, 1e3/(ncfft*self.dt)),[len(fd)])  # in mHz
        tdel = np.reshape(np.divide(td, (nrfft*self.df)),[len(td)])  # in us

        if lamsteps:
            beta = np.divide(td, (nrfft*self.dlam))  # in m^-1       

        print("These things are necessary")
        print("halve:", halve)
        print("fdop:",fdop)
        print("tdel:",tdel)
        print("beta:",beta)
            
        #norm = np.sqrt (np.multiply(LL,RR))
        #cross_cos = np.divide(np.real(cross_p),norm)
        #cross_sin = np.divide(np.imag(cross_p),norm)
        #print("Lamdyn Expectation verified")

        plt.matshow(cross_cos)
        plt.title("Real Part")
        plt.colorbar()
        plt.savefig('./sanity_check/Real_enhanced.png')
        plt.show()

        plt.matshow(cross_sin)
        plt.title("Imag Part")
        plt.colorbar()
        plt.savefig('./sanity_check/Imag_enhanced.png')
        plt.show()
        
        print("Real part Mean:", np.mean(cross_cos), "median: ", np.median(cross_cos), "\nvariance",  np.var(cross_cos),
              "standard deviation:",np.std(cross_cos), "Shape:", cross_cos.shape)

        print("\nImag part Mean:", np.mean(cross_sin), "median: ", np.median(cross_sin), "\nvariance",  np.var(cross_sin),
              "standard deviation:",np.std(cross_sin), "Shape:", cross_sin.shape)
        
        #what is sec?
        if (phasor == 'cos'):
            print('selecting cos as phasor')
            sec = cross_cos
        elif (phasor == 'sin'):
            print('selecting sin as phasor')
            sec = cross_sin
        else:
            print("Error. No correct phasor.")
        #sec = 
        #lamstep sampling lamstep sampling does change the size, but does not change the statistics
        #Windowing
        #Normalization
        #prewhitening
        #Fourier transform
        #Crop post darken
        #log scale
        
        #somewhere along the way, the statistics get screwed up
        
        #print("\nCalculating Secondary Spectrum")
        
        #what needs to get sent out?
        #sec
        #fdop
        #tdel
        #beta
        
        #This saves the attribute
        if input_dyn is None:
            if lamsteps:
                self.lamsspec = sec
            elif trap:
                self.trapsspec = sec
            else:
                self.sspec = sec

            self.fdop = fdop
            self.tdel = tdel
            if lamsteps:
                self.beta = beta
            if plot:
                self.plot_sspec(lamsteps=lamsteps, trap=trap)
        else:#this does not matter as much
            if lamsteps:
                beta = np.divide(td, (nrfft*self.dlam))  # in m^-1
                yaxis = beta
            else:
                yaxis = tdel
            return fdop, yaxis, sec
        
    #this is what we want. #I wonder when you wrote this. If only that version of you could meet this version of you.
    def calc_sspec(self, 
                   prewhite=True, halve=True,
                   plot=False, lamsteps=False,
                   input_dyn=None, 
                   input_x=None, 
                   input_y=None, trap=False,
                   window='hamming', window_frac=0.1):
        """
        Calculate secondary spectrum
        """
        #print("\nCalculating Secondary Spectrum")
        if input_dyn is None:  # use self dynamic spectrum
            if lamsteps:
                if not self.lamsteps:
                    self.scale_dyn()
                dyn = cp(self.lamdyn)
            elif trap:
                if not hasattr(self, 'trap'):
                    self.scale_dyn(scale='trapezoid')
                dyn = cp(self.trapdyn)
            else:
                dyn = cp(self.dyn)
        else:
            dyn = input_dyn  # use imput dynamic spectrum

        nf = np.shape(dyn)[0]
        nt = np.shape(dyn)[1]
        
        dyn = dyn - np.mean(dyn)  # subtract mean

        #He does the windowing first
        if window is not None:
            # Window the dynamic spectrum
            if window == 'hanning':
                cw = np.hanning(np.floor(window_frac*nt))
                sw = np.hanning(np.floor(window_frac*nf))
            elif window == 'hamming':
                cw = np.hamming(np.floor(window_frac*nt))
                sw = np.hamming(np.floor(window_frac*nf))
            elif window == 'blackman':
                cw = np.blackman(np.floor(window_frac*nt))
                sw = np.blackman(np.floor(window_frac*nf))
            elif window == 'bartlett':
                cw = np.bartlett(np.floor(window_frac*nt))
                sw = np.bartlett(np.floor(window_frac*nf))
            else:
                print('Window unknown.. Please add it!')
            chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),
                                    np.ones([nt-len(cw)]))
            subint_window = np.insert(sw, int(np.ceil(len(sw)/2)),
                                      np.ones([nf-len(sw)]))
            dyn = np.multiply(chan_window, dyn)
            dyn = np.transpose(np.multiply(subint_window,
                                           np.transpose(dyn)))

        #Some FFT stuff
            
        # find the right fft lengths for rows and columns
        nrfft = int(2**(np.ceil(np.log2(nf))+1))
        ncfft = int(2**(np.ceil(np.log2(nt))+1))
        
        dyn = dyn - np.mean(dyn)  # subtract mean

        if prewhite: #prewhite is always true, prewhite before fft
            simpw = convolve2d([[1, -1], [-1, 1]], dyn, mode='valid')
        else:
            simpw = dyn

        simf = np.fft.fft2(simpw, s=[nrfft, ncfft]) #raw fft...        
                                                    #technically complex valued 
        simf = np.real( np.multiply(simf, np.conj(simf))) #II* LL* RR*
        
        sec = np.fft.fftshift(simf)  #make arcs
        if halve: #which is always halve. Discarding nevative delays
            sec = sec[int(nrfft/2):][:]  # crop #
        
        self.raw_halved_fft = sec
        
        if halve:
            td = np.array(list(range(0, int(nrfft/2))))
        else:
            td = np.array(list(range(0, int(nrfft))))
            
        fd = np.array(list(range(int(-ncfft/2), int(ncfft/2))))
        fdop = np.reshape(np.multiply(fd, 1e3/(ncfft*self.dt)),
                          [len(fd)])  # in mHz
        tdel = np.reshape(np.divide(td, (nrfft*self.df)),
                          [len(td)])  # in us

        if lamsteps:
            beta = np.divide(td, (nrfft*self.dlam))  # in m^-1

        if prewhite:  # Now post-darken. after beta
            if halve:
                vec1 = np.reshape(np.power(np.sin(
                                  np.multiply(sc.pi/ncfft, fd)), 2),
                                  [ncfft, 1])
                vec2 = np.reshape(np.power(np.sin(
                                  np.multiply(sc.pi/nrfft, td)), 2),
                                  [1, int(nrfft/2)])
                postdark = np.transpose(vec1*vec2)
                postdark[:, int(ncfft/2)] = 1
                postdark[0, :] = 1
                sec = np.divide(sec, postdark)
            else:
                raise RuntimeError('Cannot apply prewhite to full frame')
        
        self.lamsspec_prelog_sqld = sec #raw statistics
                
        sec = 10*np.log10(sec) #Difference of Log Powers??? #This is the line that throws the warning

        #load into saver
        if input_dyn is None:
            if lamsteps:
                self.lamsspec = sec
            elif trap:
                self.trapsspec = sec
            else:
                self.sspec = sec

            self.fdop = fdop
            self.tdel = tdel
            if lamsteps:
                self.beta = beta
            if plot:
                self.plot_sspec(lamsteps=lamsteps, trap=trap)
        else:
            if lamsteps:
                beta = np.divide(td, (nrfft*self.dlam))  # in m^-1
                yaxis = beta
            else:
                yaxis = tdel
            return fdop, yaxis, sec

        
    def trans_nan(self, dyn_object ,a_clip= 32, log_multi = 1, 
                                 obs_name=None, lamsteps=False,  
                                 real_arg = False, test_rig=False,
                                 input_dyn=None, trap=False, 
                                 trans_nan_a= None, trans_nan_b=None, trans_nan_c=None):
        
        self.tr_n_a = trans_nan_a
        self.tr_n_b = trans_nan_b
        self.tr_n_c = trans_nan_c
        
        print("Trans Nans!")
        print("Trans nan a:", self.tr_n_a)
        print("Trans nan b:", self.tr_n_b)
        print("Trans nan c:", self.tr_n_c)

        
    def arc_sampling(self, dyn_object ,a_clip= 32, log_multi = 1, 
                                 obs_name=None, lamsteps=False,  
                                 real_arg = False, test_rig=False,
                                 input_dyn=None, trap=False, verbose=True):
        
        #A function that does arc tracing minus the trim of all the analytics
        
        print("For a given dynsepc image. Do arc analytics")
        
        #load the curve

        the_curve = dyn_object.betaeta*np.power(dyn_object.xplot_sspec, 2)
        if (verbose):
            print("Loading fitted arc curve:", the_curve)
            print("xplot_sspec:", dyn_object.xplot_sspec)

            print("Len the curve:", len(the_curve))
            print("Len xplot_sspec:", len(dyn_object.xplot_sspec))

            plt.plot(dyn_object.xplot_sspec, the_curve)
            plt.show()

        arc_del_nu_on_nu = []
        arc_tdel = []
        #print("arc_del_nu_on_nu and arc_tdel:",arc_del_nu_on_nu,arc_tdel)
        #print("Arc_top:", dyn_object.arc_top)
        
        #I constrain by tau?
        for i in range(0, len(the_curve)):
            #for the length of the curve
            if(the_curve[i]<dyn_object.arc_top): #so long as its less than tdel
                #put in arc_del_nu_on_ni the values in xplot_sspec
                arc_del_nu_on_nu.append(dyn_object.xplot_sspec[i])
                arc_tdel.append(the_curve[i])

        #I suspect these lengths will be less than the curves upstaris
        if (verbose):
            print("Len the curve:", len(arc_del_nu_on_nu))
            print("Len xplot_sspec:", len(arc_tdel))

            print("Plotting this. It should be an arc as well.")
            #These are delay doppler units
            plt.plot(arc_del_nu_on_nu, arc_tdel)
            plt.show()

        #print("arc_del_nu_on_nu:", arc_del_nu_on_nu,arc_tdel)
        #print("arc_tdel:", arc_tdel)
        
        nchan_arc_tdel = []
        nchan_arc_del_nu_on_nu = []

        #Therefore it is this that does the sampling?
        #where the magic happens
        for t in range(0, len(arc_del_nu_on_nu)): #iterate through that ard del_nu_on_nu
            #mayhaps you could change the way this is sampled in order to acquire a better sampling resolution
            
            delay_mapable = arc_tdel[t]
            delay_max = dyn_object.beta[:dyn_object.plt_sspec_ind][-1]
            delay_per_chan = delay_max/len(dyn_object.beta[:dyn_object.plt_sspec_ind])
            tdel_nchan = delay_mapable*(1/delay_per_chan)

            nchan_arc_tdel.append(int(tdel_nchan))

            doppler_mapable =  arc_del_nu_on_nu[t]
            tot_dop_shft = dyn_object.xplot_sspec[-1] + abs(dyn_object.xplot_sspec[0])
            dop_shft_per_chan = tot_dop_shft/len(dyn_object.xplot_sspec)

            if(np.sign(doppler_mapable) > 0): #positive
                dop_shft_nchan= (len(dyn_object.xplot_sspec)/2)+doppler_mapable*(1/dop_shft_per_chan)
            elif(np.sign(doppler_mapable) < 0): #negative
                dop_shft_nchan= (len(dyn_object.xplot_sspec)/2)-abs(doppler_mapable*(1/dop_shft_per_chan))
                
            nchan_arc_del_nu_on_nu.append(int(dop_shft_nchan))
            
        if (verbose):
            print("But what are you?") #its an arc
            print("Len of nchan_tdel:", len(nchan_arc_tdel))
            print("Len of tdel:", nchan_arc_tdel)
            print("Len of nchan_del_nu_on_nu:", len(nchan_arc_del_nu_on_nu))
            print("Len of nchan_del_nu_on_nu::", nchan_arc_del_nu_on_nu)
            #They are all arcs that go through various stages of resiziing etc
            #plt.plot(nchan_arc_del_nu_on_nu, nchan_arc_tdel)
            #plt.show()
        
        #ima
        arc_strip = []
        avg_arc_samp = []
        avg_arc_samp_2 = []

        for z in range(0,len(nchan_arc_del_nu_on_nu)):
            try:
                eight_point = [            
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]+1],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]-1],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]+1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]+1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]-1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]-1]
                ]
            except IndexError:
                #print("Triggered!")
                eight_point = [            
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]+1],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]-1],     
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]+1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]+1],   
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]-1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]-1]
                ]

            #lets take adjacent 2
            try:
                eight_point_2 = [            
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]+1],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]-1],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]+1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]+1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]-1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]-1],

                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+2][nchan_arc_del_nu_on_nu[z]],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]+2],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]-2],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+2][nchan_arc_del_nu_on_nu[z]+2],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]+2],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+2][nchan_arc_del_nu_on_nu[z]-2],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]-2],

                    #blindspot
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+2][nchan_arc_del_nu_on_nu[z]+1],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]+1],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]+2],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]-2],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+2][nchan_arc_del_nu_on_nu[z]-1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]-1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]+2],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]-2]

                ]
            except IndexError:
                #print("Triggered!")
                eight_point_2 = [            
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]+1],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]-1],     
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]+1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]+1],   
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]-1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]-1],

                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]+2],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]-2],     
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]+1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]+2],   
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]-1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]-2],

                    #blindspot
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+2][nchan_arc_del_nu_on_nu[z]+1],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]+1],     
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]+2],    
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]-2],     
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+2][nchan_arc_del_nu_on_nu[z]-1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]-1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]+2],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]-2]                
                ]
                
            #I think this is where arc_strip main gets sampled
            cent_point = dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]]

            avg = (cent_point + np.sum(eight_point))/(len(eight_point)+1)
            avg_2 = (cent_point + np.sum(eight_point_2))/(len(eight_point_2)+1) 

            arc_strip.append(cent_point) #append to cent_point
            avg_arc_samp.append(avg)
            avg_arc_samp_2.append(avg_2)

        len_obj = len(dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][0])
        center = len_obj/2

        left_samps = np.linspace(0, center, num=11)[1:-2]
        right_samps = np.linspace(center, len_obj, num=11)[2:-1]

        noise_samples= []
        for i in range(0, len(left_samps)):
            noise_samples.append(int(left_samps[i]))
            noise_samples.append(int(right_samps[i]))

        horizon_point = len(dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:])/2 - len(dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:])/4 - len(dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:])/8
        
        arr_of_nan_perc = []
        for lines in noise_samples:
            vert_slice = dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][:,lines][:] #length of arc is 1271

            col_nan_count = 0
            for bb in range(0, len(vert_slice)):
                    if math.isnan(vert_slice[bb]):
                        col_nan_count+=1

            perc_nan_vert_slice = 100*col_nan_count/len(vert_slice)
            arr_of_nan_perc.append(perc_nan_vert_slice)
        
        if (verbose):
            print("Taking several lines and counting the percentage of nans within them:")
            for gh in range(0, len(noise_samples)):
                print("Vertical Line:",noise_samples[gh], "perc_nan:", arr_of_nan_perc[gh] )
        
        vert_slice_l = dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][:,int(center-(center/2))]
        vert_slice_r = dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][:,int(center+(center/2))]
        horizon_slice = dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][int(horizon_point)]
        
        nan_count_horizon_slice = 0
        for yy in range(0, len(horizon_slice)):
            if math.isnan(horizon_slice[yy]):
                nan_count_horizon_slice+=1

        if (verbose):
            print(*horizon_slice, sep=', ')
            print("nan count left slice:", nan_count_horizon_slice)
            print("Fraction of nan count:", nan_count_horizon_slice, "/",len(horizon_slice))
            print("Percentage of nan:", 100*nan_count_horizon_slice/len(horizon_slice))
        
        nan_count_l = 0
        for xx in range(0, len(vert_slice_l)):
            if math.isnan(vert_slice_l[xx]):
                nan_count_l+=1

        if (verbose):             
            print(*vert_slice_l, sep=', ')
            print("nan count left slice:", nan_count_l)
            print("Fraction of nan count:", nan_count_l, "/",len(vert_slice_l))
            print("Percentage of nan:", 100*nan_count_l/len(vert_slice_l))

        nan_count_r = 0
        for jj in range(0, len(vert_slice_r)):
            if math.isnan(vert_slice_r[jj]):
                nan_count_r+=1
                
        if (verbose):
            print(*vert_slice_r, sep=', ')
            print("nan count right slice:", nan_count_r)
            print("Fraction of nan count:", nan_count_r, "/", len(vert_slice_r))
            print("Percentage of nan:", 100*nan_count_r/len(vert_slice_r))

        #print("\n\n------------------------------------------------------------------------------------------------------------\n\n")

        nan_count = 0
        for xx in range(0, len(arc_strip)):
            if math.isnan(arc_strip[xx]):
                nan_count+=1

        if (verbose):
            print("Arc_strip:", arc_strip)
            print("N pixels in arc:", len(arc_strip))
            print("Number of nans in arc:", nan_count, "/", len(arc_strip))
            print("Fraction of nans in arc:", nan_count, "/", len(arc_strip))
            print("Percentage of nans in the array:", 100*nan_count/len(arc_strip))

        #both of them get output here at the end
        self.exp_arc_delnu_nu = arc_del_nu_on_nu        
        self.arc_strip_main = arc_strip
        
    def arc_sampling(self, dyn_object ,a_clip= 32, log_multi = 1, 
                                 obs_name=None, lamsteps=False,  
                                 real_arg = False, test_rig=False,
                                 input_dyn=None, trap=False):
        
        print("For a given dynsepc image. Do arc analytics")
        
        #load the curve
        the_curve = dyn_object.betaeta*np.power(dyn_object.xplot_sspec, 2)
        
        arc_del_nu_on_nu = []
        arc_tdel = [] 

        for i in range(0, len(the_curve)):
            if(the_curve[i]<dyn_object.arc_top):
                arc_del_nu_on_nu.append(dyn_object.xplot_sspec[i])
                arc_tdel.append(the_curve[i])
        ####
        plt.pcolormesh(dyn_object.xplot_sspec, 
                   dyn_object.beta[:dyn_object.plt_sspec_ind], #y axis...
                   dyn_object.plt_sspec[:dyn_object.plt_sspec_ind, :], #
                   vmin=dyn_object.plt_sspec_vmin,  
                   vmax=dyn_object.plt_sspec_vmax)
        
        plt.ylabel(r'$f_\lambda$ (m$^{-1}$)')
        plt.xlabel(r'$f_t$ (mHz)')
        
        plt.axvline(x=arc_del_nu_on_nu[0], color='red', alpha = 0.4, linestyle='--')
        plt.axvline(arc_del_nu_on_nu[-1], color= 'red', alpha = 0.4, linestyle='--')
        
        bottom, top = plt.ylim()
        
        plt.plot(dyn_object.xplot_sspec, 
                dyn_object.betaeta*np.power(dyn_object.xplot_sspec, 2), #This is literally the arc...
                'r--', alpha=0.7)
        plt.title(obs_name+"arc sampled data")
        plt.ylim(bottom, top)
        plt.show()
        ####
        
        nchan_arc_tdel = []
        nchan_arc_del_nu_on_nu = []

        for t in range(0, len(arc_del_nu_on_nu)):
            delay_mapable = arc_tdel[t]
            delay_max = dyn_object.beta[:dyn_object.plt_sspec_ind][-1]
            delay_per_chan = delay_max/len(dyn_object.beta[:dyn_object.plt_sspec_ind])
            tdel_nchan = delay_mapable*(1/delay_per_chan)

            nchan_arc_tdel.append(int(tdel_nchan))

            doppler_mapable =  arc_del_nu_on_nu[t]
            tot_dop_shft = dyn_object.xplot_sspec[-1] + abs(dyn_object.xplot_sspec[0])
            dop_shft_per_chan = tot_dop_shft/len(dyn_object.xplot_sspec)

            if(np.sign(doppler_mapable) > 0): #positive
                dop_shft_nchan= (len(dyn_object.xplot_sspec)/2)+doppler_mapable*(1/dop_shft_per_chan)
            elif(np.sign(doppler_mapable) < 0): #negative
                dop_shft_nchan= (len(dyn_object.xplot_sspec)/2)-abs(doppler_mapable*(1/dop_shft_per_chan))
            nchan_arc_del_nu_on_nu.append(int(dop_shft_nchan))

    #
        plt.plot(dyn_object.xplot_sspec, 
                 dyn_object.betaeta*np.power(dyn_object.xplot_sspec, 2), #This is literally the arc...
                 'r--', alpha=0.7)
        plt.grid()
        plt.axvline(x=arc_del_nu_on_nu[0], color='red', alpha = 0.4, linestyle='--') 
        plt.text(arc_del_nu_on_nu[0], 0, str(arc_del_nu_on_nu[0]) , color='red', alpha = 0.8, fontsize=16)
        
        plt.axvline(arc_del_nu_on_nu[-1], color= 'red', alpha = 0.4, linestyle='--')
        plt.text(arc_del_nu_on_nu[-1], 0, str(arc_del_nu_on_nu[-1]) , color='red', alpha = 0.8, fontsize=16)
        
        plt.title("Modelled arc on data")
        plt.ylim(bottom, top)
        plt.show()    


        arc_strip = []
        avg_arc_samp = []
        avg_arc_samp_2 = []

        for z in range(0,len(nchan_arc_del_nu_on_nu)):
            try:
                eight_point = [            
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]+1],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]-1],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]+1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]+1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]-1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]-1]
                ]
            except IndexError:
                print("Triggered!")
                eight_point = [            
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]+1],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]-1],     
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]+1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]+1],   
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]-1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]-1]
                ]

            #lets take adjacent 2

            try:
                eight_point_2 = [            
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]+1],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]-1],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]+1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]+1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]-1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]-1],

                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+2][nchan_arc_del_nu_on_nu[z]],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]+2],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]-2],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+2][nchan_arc_del_nu_on_nu[z]+2],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]+2],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+2][nchan_arc_del_nu_on_nu[z]-2],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]-2],

                    #blindspot
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+2][nchan_arc_del_nu_on_nu[z]+1],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]+1],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]+2],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]-2],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]+2][nchan_arc_del_nu_on_nu[z]-1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]-1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]+2],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]-2]

                ]
            except IndexError:
                print("Triggered!")
                eight_point_2 = [            
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]+1],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]-1],     
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]+1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]+1],   
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]-1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]-1],

                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]],     
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]+2],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]-2],     
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]+1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]+2],   
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]-1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]-2],

                    #blindspot
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+2][nchan_arc_del_nu_on_nu[z]+1],    
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]+1],     
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]+2],    
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+1][nchan_arc_del_nu_on_nu[z]-2],     
                    #dyn_object.plt_sspec[:rcp_longtrack_26.plt_sspec_ind,:][nchan_arc_tdel[z]+2][nchan_arc_del_nu_on_nu[z]-1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-2][nchan_arc_del_nu_on_nu[z]-1],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]+2],   
                    dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]-1][nchan_arc_del_nu_on_nu[z]-2]                
                ]

            cent_point = dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][nchan_arc_tdel[z]][nchan_arc_del_nu_on_nu[z]]

            avg = (cent_point + np.sum(eight_point))/(len(eight_point)+1)
            avg_2 = (cent_point + np.sum(eight_point_2))/(len(eight_point_2)+1) 

            arc_strip.append(cent_point)
            avg_arc_samp.append(avg)
            avg_arc_samp_2.append(avg_2)

        len_obj = len(dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][0])
        center = len_obj/2

        left_samps = np.linspace(0, center, num=11)[1:-2]
        right_samps = np.linspace(center, len_obj, num=11)[2:-1]

        noise_samples= []
        for i in range(0, len(left_samps)):
            noise_samples.append(int(left_samps[i]))
            noise_samples.append(int(right_samps[i]))

        horizon_point = len(dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:])/2 - len(dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:])/4 - len(dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:])/8
        
        ########################################################################################################
        
        fig = plt.figure(figsize=(15, 18)) #depth and width or width and depth
        plt.title("Sspec_data alternative plot")
        plt.imshow(  np.flip(dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:], axis= 0)    , 
                   aspect='auto', 
                   cmap='terrain')#, cmap='terrain')#, interpolation='nearest'
        
        for samps in noise_samples:
            plt.axvline(x = samps, alpha=0.5 )
            
        plt.axhline(y = horizon_point, alpha=0.5 )
        plt.colorbar()
        plt.grid(alpha=0.25)
        plt.show()
        plt.close()

        ########################################################################################################

        arr_of_nan_perc = []
        for lines in noise_samples:
            vert_slice = dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][:,lines][:] #length of arc is 1271

            col_nan_count = 0
            for bb in range(0, len(vert_slice)):
                    if math.isnan(vert_slice[bb]):
                        col_nan_count+=1

            perc_nan_vert_slice = 100*col_nan_count/len(vert_slice)
            arr_of_nan_perc.append(perc_nan_vert_slice)

        print("Taking several lines and counting the percentage of nans within them:")
        for gh in range(0, len(noise_samples)):
            print("Vertical Line:",noise_samples[gh], "perc_nan:", arr_of_nan_perc[gh] )

        plt.plot(arr_of_nan_perc)
        plt.title("Number of nans in the vertical stripes")
        plt.axhline(y=np.mean(arr_of_nan_perc))
        plt.grid()
        plt.show()
        
        vert_slice_l = dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][:,int(center-(center/2))]
        vert_slice_r = dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][:,int(center+(center/2))]
        horizon_slice = dyn_object.plt_sspec[:dyn_object.plt_sspec_ind,:][int(horizon_point)]
        
        nan_count_horizon_slice = 0
        for yy in range(0, len(horizon_slice)):
            if math.isnan(horizon_slice[yy]):
                nan_count_horizon_slice+=1

        print(*horizon_slice, sep=', ')
        print("nan count left slice:", nan_count_horizon_slice)
        print("Fraction of nan count:", nan_count_horizon_slice, "/",len(horizon_slice))
        print("Percentage of nan:", 100*nan_count_horizon_slice/len(horizon_slice))

        plt.plot(horizon_slice)
        plt.title("Noise Sample Across Delay val: "+str(int(horizon_point))    )
        plt.grid()
        plt.show()
        

        nan_count_l = 0
        for xx in range(0, len(vert_slice_l)):
            if math.isnan(vert_slice_l[xx]):
                nan_count_l+=1

        print(*vert_slice_l, sep=', ')
        print("nan count left slice:", nan_count_l)
        print("Fraction of nan count:", nan_count_l, "/",len(vert_slice_l))
        print("Percentage of nan:", 100*nan_count_l/len(vert_slice_l))

        plt.plot(vert_slice_l)
        plt.title("Noice Sample Negative Doppler Field: "+str(int(center-(center/2)))    )
        plt.grid()
        plt.show()

        nan_count_r = 0
        for jj in range(0, len(vert_slice_r)):
            if math.isnan(vert_slice_r[jj]):
                nan_count_r+=1

        print(*vert_slice_r, sep=', ')
        print("nan count right slice:", nan_count_r)
        print("Fraction of nan count:", nan_count_r, "/", len(vert_slice_r))
        print("Percentage of nan:", 100*nan_count_r/len(vert_slice_r))

        plt.plot(vert_slice_r)
        plt.title("Noice Sample Positive Doppler Field: "+str(int(center+(center/2))))
        plt.grid()
        plt.show()

        print("\n\n------------------------------------------------------------------------------------------------------------\n\n")

        nan_count = 0
        for xx in range(0, len(arc_strip)):
            if math.isnan(arc_strip[xx]):
                nan_count+=1

        print("Arc_strip:", arc_strip)
        print("N pixels in arc:", len(arc_strip))
        print("Number of nans in arc:", nan_count, "/", len(arc_strip))
        print("Fraction of nans in arc:", nan_count, "/", len(arc_strip))
        print("Percentage of nans in the array:", 100*nan_count/len(arc_strip))

        self.exp_arc_delnu_nu = arc_del_nu_on_nu
        
        plt.plot(arc_del_nu_on_nu, arc_strip)
        plt.title("Arc Sampling")
        plt.grid(alpha = 0.4)
        plt.axvline(x=arc_del_nu_on_nu[int(len(arc_del_nu_on_nu)/2)],color = 'black', alpha=0.9)
        plt.xlabel(r'$f_t$ (mHz)')
        plt.axhline(y=np.mean(arc_strip),color = 'red', alpha=0.2)
        #plt.axvline(x=len(arc_strip)/2,color = 'red', alpha=0.2)
        plt.savefig('./uncalib_save_arc/arc'+obs_name+'.png')
        plt.show()
        
        self.arc_strip_main = arc_strip

        #for zx in range(0, len(arc_strip)):
        #    if math.isnan(arc_strip[zx]):
        #        arc_strip[zx]=0
                
        #print("nans set to 0 arc_strip", len(arc_strip))
        #plt.plot(arc_del_nu_on_nu, arc_strip)
        #plt.title("Arc Sampling nans set to 0")
        #plt.xlabel(r'$f_t$ (mHz)')
        #plt.grid(alpha = 0.4)
        #plt.axvline(x=arc_del_nu_on_nu[int(len(arc_del_nu_on_nu)/2)],color = 'black', alpha=0.9)
        #plt.show()

        plt.plot(arc_del_nu_on_nu, avg_arc_samp)
        plt.title("Arc Sampling with Avg")
        plt.xlabel(r'$f_t$ (mHz)')
        plt.grid(alpha = 0.4)
        plt.axhline(y=np.mean(avg_arc_samp),color = 'red', alpha=0.2)
        plt.axvline(x=arc_del_nu_on_nu[int(len(arc_del_nu_on_nu)/2)],color = 'black', alpha=0.9)
        plt.show()

        plt.plot(arc_del_nu_on_nu, avg_arc_samp_2)
        plt.title("Arc Sampling with Avg_2")
        plt.xlabel(r'$f_t$ (mHz)')
        plt.grid(alpha = 0.4)
        plt.axhline(y=np.mean(avg_arc_samp_2),color = 'red', alpha=0.2)
        plt.axvline(x=arc_del_nu_on_nu[int(len(arc_del_nu_on_nu)/2)],color = 'black', alpha=0.9)
        plt.show()
        

    def cross_prod_mod_b(self, lcp_con, rcp, a_clip= 32, log_multi = 1, 
                                 obs_name=None, lamsteps=False,  
                                 real_arg = False, test_rig=False,
                                 input_dyn=None, trap=False):
        
        print("Mod b protocols engaged!")

        cross_prod = np.multiply(lcp_con.lamsspec_fft_samp_3, rcp.lamsspec_fft_samp_3) #post darken. Took the full pipe..
            
        sec = cross_prod.imag
            
        vec1 = np.reshape(np.power(np.sin(np.multiply(sc.pi/self.ncfft_sv, self.fd_sv)), 2),
                            [self.ncfft_sv, 1])
        vec2 = np.reshape(np.power(np.sin(np.multiply(sc.pi/self.nrfft_sv, self.td_sv)), 2),
                            [1, int(self.nrfft_sv/2)])
        postdark = np.transpose(vec1*vec2)
        postdark[:, int(self.ncfft_sv/2)] = 1
        postdark[0, :] = 1
        sec = np.divide(sec, postdark)

        self.pre_log_cross_sspec_mod_b = sec
        
        print("Applying Transform. Post post darkening")
        for t in range(0, len(sec)):
            for g in range(0, len(sec[0])):
                if(sec[t][g] == 0):
                    sec[t][g] = log_multi*0
                else:
                    sec[t][g] = log_multi*np.log10(abs(sec[t][g]))

        #THE RESULT WILL BE STORED IN lamsspec
        if input_dyn is None:
            if lamsteps:
                self.lamsspec = sec
            elif trap:
                self.trapsspec = sec
        else:
            self.sspec = sec

        #rcp.plot_sspec(lamsteps=True,# filename=('./fourier_images_pngs/single_imager/'+obs_name+'np_log_10_p_postdark_imag_part_LstarR_x_postdark no clip'),
        #                title= "imag("+obs_name+")_preserve_negative")

    def calc_mod_b_UL_log(self, lcp_con, rcp, in_ll_star, in_rr_star, a_clip= 32, log_multi = 1, 
                                 obs_name=None, lamsteps=False, pre_log = False,  
                                 real_arg = False, test_rig=False,
                                 input_dyn=None, trap=False):
        
        print("Mod b protocols engaged!")

        cross_prod = np.multiply(lcp_con.lamsspec_fft_samp_3, rcp.lamsspec_fft_samp_3) #post darken. Took the full pipe..
            
        sec = cross_prod.imag
            
        vec1 = np.reshape(np.power(np.sin(np.multiply(sc.pi/self.ncfft_sv, self.fd_sv)), 2),
                            [self.ncfft_sv, 1])
        vec2 = np.reshape(np.power(np.sin(np.multiply(sc.pi/self.nrfft_sv, self.td_sv)), 2),
                            [1, int(self.nrfft_sv/2)])
        postdark = np.transpose(vec1*vec2)
        postdark[:, int(self.ncfft_sv/2)] = 1
        postdark[0, :] = 1
        sec = np.divide(sec, postdark)

        self.pre_log_cross_sspec_mod_b = sec
        
        print("Applying Transform. Post post darkening")
        for t in range(0, len(sec)):
            for g in range(0, len(sec[0])):
                if(sec[t][g] == 0):
                    sec[t][g] = log_multi*0
                else:
                    sec[t][g] = log_multi*np.log10(abs(sec[t][g]))
                    
        if input_dyn is None:
            if lamsteps:
                self.lamsspec = sec
            elif trap:
                self.trapsspec = sec
        else:
            self.sspec = sec
        #if(count_neg):
        #    if(pre_log):
        #        print("The image is pre log:")
        #    print("Count the number of negatives")
            
        #    c_neg_ll_star = 0
        #    for j in range(0, len(ll_star)):
        #        for k in range(0, len(ll_star[0])):
        #            if(ll_star[j][k] < 0):
        #                c_neg_ll_star += 1
                        
        #print("LL*:", ll_star)
        #print("RR*:", rr_star)
        
        print("CALCULATE UPPER LIMIT!\n")
        
        print("Upper limit = 2log(|im(l*r)|) - log(ll*) - log(rr*)")
              
        UL_log = np.subtract(np.subtract(np.multiply(self.lamsspec, 2), in_rr_star), in_ll_star)
        
        #for v in range(0, len(sec)):
        #    for b in range(0, len(sec[0])):
        #        sec[v][b]= UL_log[v][b]
                
        plt.matshow(UL_log)
        plt.colorbar()
        plt.grid()
        plt.show()

        if input_dyn is None:
            if lamsteps:
                self.lamsspec = UL_log
            elif trap:
                self.trapsspec = UL_log
        else:
            self.sspec = UL_log
            
        #THE RESULT WILL BE STORED IN lamsspec

        #Sending itself to itself. So its sort of like Jesus but without the sacrifice.
        
        #print(rcp.lamsspec[5])
        #plt.matshow(rcp.lamsspec)
        #plt.grid()
        #plt.show()

        #print("\nDo these look the same to you?\n")

        #print(self.lamsspec[5])
        #plt.matshow(self.lamsspec)
        #plt.grid()
        #plt.show()

        #self.plot_sspec(lamsteps=True, title= "imag("+obs_name+")")
        #filename=('./fourier_images_pngs/single_imager/'+obs_name+'np_log_10_p_postdark_imag_part_LstarR_x_postdark no clip'),

    def calc_mod_b_UL_prelog(self, lcp_con, rcp, in_ll_star, in_rr_star, a_clip= 32, log_multi = 1, 
                                 obs_name=None, lamsteps=False, pre_log = False,  
                                 real_arg = False, test_rig=False,
                                 input_dyn=None, trap=False):
        
        print("Mod b protocols engaged!")

        cross_prod = np.multiply(lcp_con.lamsspec_fft_samp_3, rcp.lamsspec_fft_samp_3) #post darken. Took the full pipe..
        
        #ima change this to real and when I do the computation it will be centered at 1.
        print("Taking real part...")
        print("Real:", cross_prod.real)
        
        print("Imag:", cross_prod.imag)
        
        sec = cross_prod.real
        
        prin("Sec")
            
        #vec1 = np.reshape(np.power(np.sin(np.multiply(sc.pi/self.ncfft_sv, self.fd_sv)), 2),
        #                    [self.ncfft_sv, 1])
        #vec2 = np.reshape(np.power(np.sin(np.multiply(sc.pi/self.nrfft_sv, self.td_sv)), 2),
        #                    [1, int(self.nrfft_sv/2)])
        #postdark = np.transpose(vec1*vec2)
        #postdark[:, int(self.ncfft_sv/2)] = 1
        #postdark[0, :] = 1
        #sec = np.divide(sec, postdark)

        self.pre_log_cross_sspec_mod_b = sec
        
        print("Applying Transform. Post post darkening")
        for t in range(0, len(sec)):
            for g in range(0, len(sec[0])):
                if(sec[t][g] == 0):
                    sec[t][g] = log_multi*0
                else:
                    sec[t][g] = log_multi*np.log10(abs(sec[t][g]))
                    
        if input_dyn is None:
            if lamsteps:
                self.lamsspec = sec
            elif trap:
                self.trapsspec = sec
        else:
            self.sspec = sec
        
        print("CALCULATE UPPER LIMIT!\n")
        
        print("Upper limit = im(l*r)/ sqrt(ll* x rr*):")
                
        UL_pre_log = np.true_divide(self.pre_log_cross_sspec_mod_b ,np.sqrt(np.multiply(in_rr_star, in_ll_star))    )
        
        print("UL pre log", UL_pre_log)

        plt.matshow(UL_pre_log)
        plt.colorbar()
        plt.grid()
        plt.show()

        if input_dyn is None:
            if lamsteps:
                self.lamsspec = UL_pre_log
            elif trap:
                self.trapsspec = UL_pre_log
        else:
            self.sspec = UL_pre_log
            
        #THE RESULT WILL BE STORED IN lamsspec

        #Sending itself to itself. So its sort of like Jesus but without the sacrifice.
        
        #print(rcp.lamsspec[5])
        #plt.matshow(rcp.lamsspec)
        #plt.grid()
        #plt.show()

        #print("\nDo these look the same to you?\n")

        #print(self.lamsspec[5])
        #plt.matshow(self.lamsspec)
        #plt.grid()
        #plt.show()

        #self.plot_sspec(lamsteps=True, title= "imag("+obs_name+")")
        #filename=('./fourier_images_pngs/single_imager/'+obs_name+'np_log_10_p_postdark_imag_part_LstarR_x_postdark no clip'),
        
        
    def cross_prod_mod_a(self, lcp_con, rcp, a_clip= 32, log_multi = 1, 
                                 obs_name=None, lamsteps=False,  
                                 real_arg = False, test_rig=False,
                                 input_dyn=None, trap=False):

        print("Mod a protocols engaged!")
        
        cross_prod = np.multiply(lcp_con.lamsspec_fft_samp_3, rcp.lamsspec_fft_samp_3) #post darken. Took the full pipe..
            
        sec = cross_prod.imag #Do we add this
            
        vec1 = np.reshape(np.power(np.sin(np.multiply(sc.pi/self.ncfft_sv, self.fd_sv)), 2),
                            [self.ncfft_sv, 1])
        vec2 = np.reshape(np.power(np.sin(np.multiply(sc.pi/self.nrfft_sv, self.td_sv)), 2),
                            [1, int(self.nrfft_sv/2)])
        postdark = np.transpose(vec1*vec2)
        postdark[:, int(self.ncfft_sv/2)] = 1
        postdark[0, :] = 1
        sec = np.divide(sec, postdark)
        #do we add this
        
        #These logarithms are dangerous. They blow things rediculously out of porportion
        #numbers between -1 and 1 should be blanked.
        
        self.pre_log_cross_sspec_mod_a = sec
        
        #so just get me the statistics first
        size_arr = len(sec)*len(sec[0])
        print("Size of pre log sec:", len(sec), "x", len(sec[0]), "=", len(sec)*len(sec[0]), "Pixels")
        print("Mean of pre log sec:", np.mean(sec))
        print("Standard deviation of pre log sec", np.std(sec))
        
        print("Pre log sec positive/upper bound", np.mean(sec) + np.std(sec))
        print("Pre log sec negative/lower bound", np.mean(sec) - np.std(sec))
        
        n_zero = 0
        n_neg_haz_vals = 0
        n_pos_haz_vals = 0
        norm_vals = 0
        
        vertical_loc = []
        horizontal_loc = []
        
        for t in range(0, len(sec)):
            for g in range(0, len(sec[0])):
                if(sec[t][g] == 0):
                    n_zero+=1
                elif(sec[t][g] > -1 and sec[t][g] < 0): #if negative
                    vertical_loc.append(t)
                    horizontal_loc.append(g)
                    
                    n_neg_haz_vals += 1
                elif(sec[t][g] > 0 and sec[t][g] < 1): #if negative
                    vertical_loc.append(t)
                    horizontal_loc.append(g)
                    
                    n_pos_haz_vals += 1
                else:
                    norm_vals+=1
                    #sec[t][g] = log_multi*np.log10(abs(sec[t][g]))
        print("\n")
        print("Of the array of size", size_arr)
        
        print("There are:", n_zero, "zeroes")
        print("There are:", n_neg_haz_vals, " negative hazard vals in the pre log array. As a percentage:", n_neg_haz_vals*(100/size_arr), "%")
        print("There are:", n_pos_haz_vals, " positive hazard vals in the pre log array. As a percentage:", n_pos_haz_vals*(100/size_arr), "%") 
        print("Thus in total there are:", (n_neg_haz_vals + n_pos_haz_vals), "points that will screw up your logarithm", "As a percentage", (n_neg_haz_vals + n_pos_haz_vals)*(100/size_arr), "% of the array is affected")
              
        print("There are:", norm_vals, " normal non haz values in the pre log array. As a percentage:", norm_vals*(100/size_arr), "%") 
        
        print("Where are they?")
        #plt.matshow(sec)
        #plt.title("Pre log Sec")
        
        #for coors in range(0, len(horizontal_loc)):
        #    plt.axvline(x = horizontal_loc[coors])
            #plt.axvline(x = horizontal_loc[-1])

        #    plt.axhline(y = vertical_loc[coors])
            #plt.axhline(y = vertical_loc[-1])
        
        #    plt.axvline(x = horizontal_loc[i_coor])
        #    plt.axhline(y = vertical_loc[i_coor])
        #plt.show()
    
        #do we add this?
        #61% of your noise should be bounded within 1 standard deviation. So if you suppress them, you will make a really really majority blanked image.
        
        print("Applying Transform. Post post darkening")
        #log is taken care here
        for t in range(0, len(sec)):
            for g in range(0, len(sec[0])):
                if(sec[t][g] == 0):
                    sec[t][g] = log_multi*0
                elif(np.sign(sec[t][g]) < 0): #if negative
                    sec[t][g] = np.sign(sec[t][g])*log_multi*np.log10(abs(sec[t][g]))
                else:
                    sec[t][g] = log_multi*np.log10(abs(sec[t][g]))
        #do we add this?

        if input_dyn is None:
            if lamsteps:
                self.lamsspec = sec
            elif trap:
                self.trapsspec = sec
        else:
            self.sspec = sec

        #rcp.plot_sspec(lamsteps=True,# filename=('./fourier_images_pngs/single_imager/'+obs_name+'np_log_10_p_postdark_imag_part_LstarR_x_postdark no clip'),
        #                title= "imag("+obs_name+")_preserve_negative")
        
    def cross_prod_half_cock(self, lcp_con, rcp, a_clip= 32, log_multi = 1, 
                                 obs_name=None, lamsteps=False,  
                                 real_arg = False, test_rig=False,
                                 input_dyn=None, trap=False):

        print("Half cocked coss prod!")
        
        cross_prod = np.multiply(lcp_con.lamsspec_fft_samp_3, rcp.lamsspec_fft_samp_3)
        #post darken. Took the full pipe..
            
        sec = cross_prod.imag
        #print("Sec")
        #print(sec)
        
        self.raw_imag = sec
        
        if input_dyn is None:
            if lamsteps:
                self.lamsspec = sec
            elif trap:
                self.trapsspec = sec
        else:
            self.sspec = sec
                            
        
    def calc_cross_sspec(self, lcp_con, rcp, a_clip= 32, log_multi = 1, 
                                 obs_name=None, lamsteps=False,  
                                 real_arg = False, test_rig=False, abs_arg= False,
                                 input_dyn=None, trap=False):
                
        cross_prod = np.multiply(lcp_con.lamsspec_fft_samp_3, rcp.lamsspec_fft_samp_3) #post darken. Took the full pipe...
        
        self.raw_cross_p = cross_prod
        
        if(real_arg):
            sec = cross_prod.real
        else:
            sec = cross_prod.imag
                
        #this is post darken
        vec1 = np.reshape(np.power(np.sin(np.multiply(sc.pi/self.ncfft_sv, self.fd_sv)), 2),
                                  [self.ncfft_sv, 1])
        vec2 = np.reshape(np.power(np.sin(np.multiply(sc.pi/self.nrfft_sv, self.td_sv)), 2),
                                  [1, int(self.nrfft_sv/2)])
        postdark = np.transpose(vec1*vec2)
        postdark[:, int(self.ncfft_sv/2)] = 1
        postdark[0, :] = 1
        sec = np.divide(sec, postdark)
        
        #sec=abs(sec)
        if(abs_arg):
            sec = np.abs(sec)
        
        self.pre_log_cross_sspec_the_truth = sec
        
        sec = log_multi*np.log10(sec) #will have nants
        
        self.the_truth_alpha = sec
        
        if input_dyn is None:
            if lamsteps:
                self.lamsspec = sec
            elif trap:
                self.trapsspec = sec
        else:
            self.sspec = sec
        
        if(real_arg):
            print("Plotting Spec. Longtrack")
            #rcp.plot_sspec(lamsteps=True, #filename=('./fourier_images_pngs/single_imager/'+obs_name+'np_log_10_p_postdark_imag_part_LstarR_x_postdark no clip'),
            #                title= "real("+obs_name+")_raw") #this will become speckly image
        else:
            print("Plotting Spec. . Longtrack")
            #rcp.plot_sspec(lamsteps=True, #filename=('./fourier_images_pngs/single_imager/'+obs_name+'np_log_10_p_postdark_imag_part_LstarR_x_postdark no clip'),
            #                title= "imag("+obs_name+")_raw") #this will become speckly image
        
        ###########################################################################################################################
        if(test_rig):       
            #print("Alter the array:")
            #attempt 1
            cross_prod = np.multiply(lcp_con.lamsspec_fft_samp_3, rcp.lamsspec_fft_samp_3) #post darken. Took the full pipe..
            
            sec = cross_prod.imag
            
            #post_darken, log, then plot
            vec1 = np.reshape(np.power(np.sin(np.multiply(sc.pi/self.ncfft_sv, self.fd_sv)), 2),
                                      [self.ncfft_sv, 1])
            vec2 = np.reshape(np.power(np.sin(np.multiply(sc.pi/self.nrfft_sv, self.td_sv)), 2),
                                      [1, int(self.nrfft_sv/2)])
            postdark = np.transpose(vec1*vec2)
            postdark[:, int(self.ncfft_sv/2)] = 1
            postdark[0, :] = 1
            sec = np.divide(sec, postdark)

            print("Applying Transform. Post post darkening")
            for t in range(0, len(sec)):
                for g in range(0, len(sec[0])):
                    if(sec[t][g] == 0):
                        sec[t][g] = log_multi*0 #don't log anything that's 0
                    else:
                        sec[t][g] = log_multi*np.log10(sec[t][g])
                    if(g < 200 or g > (1024-200)): #based on array on array size please. Thank you.
                        sec[t][g] = log_multi*np.log10(abs(sec[t][g])) #get irid of all negative. What are these white points?
                        
            #print("Preplot")
            #fig = plt.figure(figsize=(15, 18)) #depth and width or width and depth
            #plt.title("Sspec_data alternative plot")
            #plt.imshow(sec[:self.plt_sspec_ind,:], 
            #           aspect='auto', 
            #           cmap='terrain')#, cmap='terrain')#, interpolation='nearest'

            #plt.axvline(x = 200, alpha=0.5 )
            #plt.axvline(x = (1024-200), alpha=0.5 )
            #plt.colorbar()
            #plt.grid(alpha=0.25)
            #plt.show()
            #plt.close()
            
            #sec = log_multi*np.log10(sec)

            if input_dyn is None:
                if lamsteps:
                    self.lamsspec = sec
                elif trap:
                    self.trapsspec = sec
            else:
                self.sspec = sec    
            
            rcp.plot_sspec(lamsteps=True,# filename=('./fourier_images_pngs/single_imager/'+obs_name+'np_log_10_p_postdark_imag_part_LstarR_x_postdark no clip'),
                            title="The truth attempt 1") #expectation is a speckly image with a parabola of nans
                   
        ###########################################################################################################################
       
        
    def cross_prod_single_imager(self, lcp_con, rcp, a_clip= 32, log_multi = 1, 
                                 obs_name=None, lamsteps=False,  real_arg = False,
                                 input_dyn=None, trap=False):
        print("Two multiplied post darkened images")
        cross_prod = np.multiply(lcp_con.lamsspec_fft_samp_4, rcp.lamsspec_fft_samp_4)
        
        if(real_arg):
            sec = log_multi*np.log10(cross_prod.real)
        else:
            sec = log_multi*np.log10(cross_prod.imag)
            
        if input_dyn is None:
            if lamsteps:
                self.lamsspec = sec
            elif trap:
                self.trapsspec = sec
        else:
            self.sspec = sec        
        
        rcp.plot_sspec(lamsteps=True,# filename=('./fourier_images_pngs/single_imager/'+obs_name+'_log_10_imag_part_LstarR_postdark'),
                    title="Mutltiplied two post darkened images. Expectation: Empty Case")

        print("Post Darken two multiplied non post darkened images. No clipped")
        #print("First ask yourself. Do I have the relevant parameters?")
                
        cross_prod = np.multiply(lcp_con.lamsspec_fft_samp_3, rcp.lamsspec_fft_samp_3) #post darken. Took the full pipe...
        
        if(real_arg):
            sec = cross_prod.real
        else:
            sec = cross_prod.imag
                
        vec1 = np.reshape(np.power(np.sin(np.multiply(sc.pi/self.ncfft_sv, self.fd_sv)), 2),
                                  [self.ncfft_sv, 1])
        vec2 = np.reshape(np.power(np.sin(np.multiply(sc.pi/self.nrfft_sv, self.td_sv)), 2),
                                  [1, int(self.nrfft_sv/2)])
        postdark = np.transpose(vec1*vec2)
        postdark[:, int(self.ncfft_sv/2)] = 1
        postdark[0, :] = 1
        sec = np.divide(sec, postdark)
        
        sec = log_multi*np.log10(sec)
        
        if input_dyn is None:
            if lamsteps:
                self.lamsspec = sec
            elif trap:
                self.trapsspec = sec
        else:
            self.sspec = sec
                
        rcp.plot_sspec(lamsteps=True,# filename=('./fourier_images_pngs/single_imager/'+obs_name+'np_log_10_p_postdark_imag_part_LstarR_x_postdark no clip'),
                        title="Post darkend mutliplied two non post darkened images. Log. No clipped")
        
                
        if(real_arg):
            rcp.plot_sspec(lamsteps=True, filename=('./fourier_images_pngs/para_f_imag/'+obs_name+'_real_pd_2_x_pd_x_clip'+str(log_multi)+'np_log10'),
                                title=obs_name+'_real')
        else:
            cross_prod = np.multiply(lcp_con.lamsspec_fft_samp_3, rcp.lamsspec_fft_samp_3) #post darken. Took the full pipe..
            
            sec = cross_prod.imag
            
            print("Mean of image:", np.mean(sec))
            
            gaussian_field = []
            
            for b in range(0, int(len(sec)/4)):
                strip = []
                for n in range(0, int(len(sec[0])/4)):
                    strip.append(sec[b][n])
                gaussian_field.append(strip)
                
            print("Mean of gaussian_field:", np.mean(gaussian_field))
            print("Mean of gaussian_field log_10",  np.mean( log_multi*np.log10(gaussian_field)    ))
            
            print("Plot the gaussian field:")
            
            plt.matshow(gaussian_field, cmap='terrain', aspect='auto')
            plt.show()
            plt.close()
            
            plt.matshow(log_multi*np.log10(gaussian_field), cmap='terrain', aspect='auto')
            plt.show()
            plt.close()
                       
            #proceed...
            print("signal neg count:")
            print("Image size:", len(sec), "x", len(sec[0]))
            imag_area=len(sec)*len(sec[0])
            print("Image size:", imag_area)
            
            neg_c = 0
            
            #Do your thing
            for t in range(0, len(sec)):
                for g in range(0, len(sec[0])):
                    if(np.sign(sec[t][g]) < 0):
                        neg_c += 1
                        
            print("neg_count:", neg_c)
            print("Percentage of negatives in the image:", (100*neg_c)/imag_area)
            
            for t in range(0, len(sec)):
                for g in range(0, len(sec[0])): 
                    if(np.sign(sec[t][g]) < 0): #for all negative values...
                          sec[t][g] = abs(sec[t][g]) #half of the image has to be changed...
            
            #post_darken, log, then plot
            vec1 = np.reshape(np.power(np.sin(np.multiply(sc.pi/self.ncfft_sv, self.fd_sv)), 2),
                                      [self.ncfft_sv, 1])
            vec2 = np.reshape(np.power(np.sin(np.multiply(sc.pi/self.nrfft_sv, self.td_sv)), 2),
                                      [1, int(self.nrfft_sv/2)])
            postdark = np.transpose(vec1*vec2)
            postdark[:, int(self.ncfft_sv/2)] = 1
            postdark[0, :] = 1
            sec = np.divide(sec, postdark)

            sec = log_multi*np.log10(sec)

            if input_dyn is None:
                if lamsteps:
                    self.lamsspec = sec
                elif trap:
                    self.trapsspec = sec
            else:
                self.sspec = sec

            rcp.plot_sspec(lamsteps=True,# filename=('./fourier_images_pngs/single_imager/'+obs_name+'np_log_10_p_postdark_imag_part_LstarR_x_postdark no clip'),
                            title="Post darkend mutliplied two non post darkened images. Log. No clipped")
            
            rcp.plot_sspec(lamsteps=True, filename=('./fourier_images_pngs/para_f_imag/'+obs_name+'_imag_pd_2_x_pd_x_clip_corr_'+str(log_multi)+'np_log10'),
                            title=obs_name+'_imag')
        
        
        print("Post Darken two multiplied non post darkened images. Clipped")
        cross_prod_clipped = np.multiply(lcp_con.lamsspec_fft_samp_3, rcp.lamsspec_fft_samp_3)
        
        if(real_arg):
            test = cross_prod.real
        else:
            test = cross_prod.imag
            
        sec = test
        f_max = a_clip
                
        for y in range(0, len(sec)):
            for z in range(0, len(sec[0])):
                if(sec[y][z] > f_max):
                    sec[y][z] = f_max
                elif((sec[y][z] < -1*f_max)):
                    sec[y][z] = -1*f_max
                                                     
        vec1 = np.reshape(np.power(np.sin(np.multiply(sc.pi/self.ncfft_sv, self.fd_sv)), 2),[self.ncfft_sv, 1])
        vec2 = np.reshape(np.power(np.sin(np.multiply(sc.pi/self.nrfft_sv, self.td_sv)), 2),[1, int(self.nrfft_sv/2)])
        postdark = np.transpose(vec1*vec2)
        postdark[:, int(self.ncfft_sv/2)] = 1
        postdark[0, :] = 1
        
        sec = np.divide(sec, postdark)
        
        sec = log_multi*np.log10(sec) #Looks like a blank container with nans

        if input_dyn is None:
            if lamsteps:
                self.lamsspec = sec
            elif trap:
                self.trapsspec = sec
        else:
            self.sspec = sec

        rcp.plot_sspec(lamsteps=True, #filename=('./fourier_images_pngs/single_imager/'+obs_name+'np_log_10_p_postdark_imag_part_LstarR_x_postdark_clipped'),
                        title="Post darkend mutliplied two non post darkened images. Log. Clipped")

    
    def cross_prod_no_bs(self, lcp_con, rcp, cross_prod_samp=4,  a_clip = 1000, obs_name=None,
                         post_postdark =False, lamsteps=False, input_dyn=None, trap=False):
        
        print("Calculating Cross Product!!!")

        lcp_con.plot_dyn(title="LCP")
        rcp.plot_dyn(title="RCP")

        #print("Imags of LCP and RCP, I expect a complex array of numbers")
        #print("LCP_con", lcp_con.lamsspec_fft_samp_4)
        #print("\n")
        #print("RCP", rcp.lamsspec_fft_samp_4)
        #print("\n")
        
        if(cross_prod_samp==4):
            print("Multyiplying two post darkened images")
            cross_prod = np.multiply(lcp_con.lamsspec_fft_samp_4, rcp.lamsspec_fft_samp_4) #post darken. Took the full pipe...
            
            sec = cross_prod.imag
            
            if input_dyn is None:
                if lamsteps:
                    self.lamsspec = sec
                elif trap:
                    self.trapsspec = sec
            else:
                self.sspec = sec
            
            #rcp.plot_sspec(lamsteps=True, filename=('./fourier_images_pngs/image_dump/'+obs_name+'_imag_part_LstarR_postdark'),
            #               title="Multiply two post darkened images")
            
            sec = np.log10(cross_prod.imag)
            
            
            if input_dyn is None:
                if lamsteps:
                    self.lamsspec = sec
                elif trap:
                    self.trapsspec = sec
            else:
                self.sspec = sec
            
            #rcp.plot_sspec(lamsteps=True, filename=('./fourier_images_pngs/image_dump/'+obs_name+'_log_10_imag_part_LstarR_postdark'),
            #               title="np.log10 of Multiply two post darkened images")
                
        elif(cross_prod_samp==3):
            print("Multyiplying two non post darkened images. Naked Image.")
            cross_prod = np.multiply(lcp_con.lamsspec_fft_samp_3, rcp.lamsspec_fft_samp_3) #post darken. Took the full pipe...

            sec = cross_prod.imag #I have a feeling the the plot error will be incompatible blah. Just a feeling
            #can we do the double plot?
            if input_dyn is None:
                if lamsteps:
                    self.lamsspec = sec
                elif trap:
                    self.trapsspec = sec
            else:
                self.sspec = sec
                
            rcp.plot_sspec(lamsteps=True, filename=('./fourier_images_pngs/image_dump/'+obs_name+'_imag_part_LstarR_x_postdark'),
                           title="Multiplied two Non Post Darkened Images. Not clipped")
            
            sec = np.log10(cross_prod.imag) #Looks like a blank container with nans

            if input_dyn is None:
                if lamsteps:
                    self.lamsspec = sec
                elif trap:
                    self.trapsspec = sec
            else:
                self.sspec = sec
                
            rcp.plot_sspec(lamsteps=True, filename=('./fourier_images_pngs/image_dump/'+obs_name+'_log10_imag_part_LstarR_x_postdark'),
                           title="Multiplied two Non Post Darkened Images. Not clipped")
            
            if(post_postdark):
                print("Post Darkening the multiplied two non post darkened images")
                #print("First ask yourself. Do I have the relevant parameters?")
                
                cross_prod = np.multiply(lcp_con.lamsspec_fft_samp_3, rcp.lamsspec_fft_samp_3) #post darken. Took the full pipe...
                sec = cross_prod.imag
                
                #print(self.td_sv, self.fd_sv, self.nrfft_sv, self.ncfft_sv)
                
                #print("Performing post darkening")
                
                                
                vec1 = np.reshape(np.power(np.sin(
                                  np.multiply(sc.pi/self.ncfft_sv, self.fd_sv)), 2),
                                  [self.ncfft_sv, 1])
                vec2 = np.reshape(np.power(np.sin(
                                  np.multiply(sc.pi/self.nrfft_sv, self.td_sv)), 2),
                                  [1, int(self.nrfft_sv/2)])
                postdark = np.transpose(vec1*vec2)
                postdark[:, int(self.ncfft_sv/2)] = 1
                postdark[0, :] = 1
                sec = np.divide(sec, postdark)
                
                if input_dyn is None:
                    if lamsteps:
                        self.lamsspec = sec
                    elif trap:
                        self.trapsspec = sec
                else:
                    self.sspec = sec
                
                #rcp.plot_sspec(lamsteps=True, #filename=('./fourier_images_pngs/image_dump/'+obs_name+'p_postdark_imag_part_LstarR_x_postdark'),
                           #title="post darkend a Multiplied two Non Post Darkened Images. Not clipped")

                sec = 10*np.log10(sec) #Looks like a blank container with nans

                if input_dyn is None:
                    if lamsteps:
                        self.lamsspec = sec
                    elif trap:
                        self.trapsspec = sec
                else:
                    self.sspec = sec
                
                rcp.plot_sspec(lamsteps=True, #filename=('./fourier_images_pngs/image_dump/'+obs_name+'np_log_10_p_postdark_imag_part_LstarR_x_postdark'),
                               title="10*log 10 post darkend a Multiplied two Non Post Darkened Images. Not clipped")
            
            #Last Resort... Clipping...
            
            #Now we clip that image to see if there is any difference...
            cross_prod_clipped = np.multiply(lcp_con.lamsspec_fft_samp_3, rcp.lamsspec_fft_samp_3)
            
            test = cross_prod_clipped.imag
            f_max = a_clip
            
            print("Max test:", np.max(test))
            print("test_size:", len(test), "x", len(test[0]))
            
            for y in range(0, len(test)):
                for z in range(0, len(test[0])):
                    if(test[y][z] > f_max):
                        test[y][z] = f_max
                    elif((test[y][z] < -1*f_max)):
                        test[y][z] = -1*f_max
                    #if(z > 512+100 and z < 512-100): #between two limits in the array...
                        #test[y][z] = f_max
                    #if(test[y][z] < 0): #enhancing negative points...
                    #    test[y][z] = f_max
                    
            #numpy dimensions ar not the same as fdop and tau dimensions. Its time you grew up and realized this...
            print("Max test post clip:", np.max(test)) #SO if the max doesnt change then something is wrong with                    
               
            if input_dyn is None:
                if lamsteps:
                    self.lamsspec = test
                elif trap:
                    self.trapsspec = test
            else:
                self.sspec = test
            
            #rcp.plot_sspec(lamsteps=True, plotlims = False,  plotarc = False, 
                           #filename=('./fourier_images_pngs/image_dump/'+obs_name+'_imag_part_LstarR_x_postdark_clipped_'+ str(f_max)),
                           #title="Multiplied two Non Post Darkened Images. Clipped at "+ str(f_max))
            
            test = np.log10(test)
            
            if input_dyn is None:
                if lamsteps:
                    self.lamsspec = test
                elif trap:
                    self.trapsspec = test
            else:
                self.sspec = test            
            
            #rcp.plot_sspec(lamsteps=True, plotlims = False,  plotarc = False, 
                           #filename=('./fourier_images_pngs/image_dump/'+obs_name+'_log10_imag_part_LstarR_Non_postdark_clipped_'+ str(f_max)),
                           #title="Multiplied two Non Post Darkened Images. Clipped at "+ str(f_max))
            
            if(post_postdark):
                cross_prod_clipped = np.multiply(lcp_con.lamsspec_fft_samp_3, rcp.lamsspec_fft_samp_3)
            
                test = cross_prod_clipped.imag
                sec = test
                f_max = a_clip
                
                for y in range(0, len(sec)):
                    for z in range(0, len(sec[0])):
                        if(sec[y][z] > f_max):
                            sec[y][z] = f_max
                        elif((sec[y][z] < -1*f_max)):
                            sec[y][z] = -1*f_max
                                                     
                vec1 = np.reshape(np.power(np.sin(
                                  np.multiply(sc.pi/self.ncfft_sv, self.fd_sv)), 2),
                                  [self.ncfft_sv, 1])
                vec2 = np.reshape(np.power(np.sin(
                                  np.multiply(sc.pi/self.nrfft_sv, self.td_sv)), 2),
                                  [1, int(self.nrfft_sv/2)])
                postdark = np.transpose(vec1*vec2)
                postdark[:, int(self.ncfft_sv/2)] = 1
                postdark[0, :] = 1
                sec = np.divide(sec, postdark)
                
                if input_dyn is None:
                    if lamsteps:
                        self.lamsspec = sec
                    elif trap:
                        self.trapsspec = sec
                else:
                    self.sspec = sec
                
                #rcp.plot_sspec(lamsteps=True, #filename=('./fourier_images_pngs/image_dump/'+obs_name+'p_postdark_imag_part_LstarR_x_postdark'),
                           #title="post darkend a Multiplied two Non Post Darkened Images. clipped")

                sec = 10*np.log10(sec) #Looks like a blank container with nans

                if input_dyn is None:
                    if lamsteps:
                        self.lamsspec = sec
                    elif trap:
                        self.trapsspec = sec
                else:
                    self.sspec = sec
                
                rcp.plot_sspec(lamsteps=True, #filename=('./fourier_images_pngs/image_dump/'+obs_name+'np_log_10_p_postdark_imag_part_LstarR_x_postdark'),
                               title="10*log_10_post darkend a Multiplied two Non Post Darkened Images. clipped")
            
        
    #this is what we want.
    def calc_sspec_conjugate(self, prewhite=True, halve=True, plot=False, lamsteps=False,
                   input_dyn=None, input_x=None, input_y=None, trap=False,
                   window='hamming', window_frac=0.1, neg_filt_std = 1,
                   conj_arg=False, real_arg = True, neg_filt=False, filenom = ''):
        
        """
        Calculate secondary spectrum
        """
        
        print("Calc sspec birefringence.")
        #print("\nCalculating Secondary Spectrum")
        if input_dyn is None:  # use self dynamic spectrum
            if lamsteps:
                if not self.lamsteps:
                    self.scale_dyn()
                dyn = cp(self.lamdyn)
            elif trap:
                if not hasattr(self, 'trap'):
                    self.scale_dyn(scale='trapezoid')
                dyn = cp(self.trapdyn)
            else:
                dyn = cp(self.dyn)
        else:
            dyn = input_dyn  # use imput dynamic spectrum

        nf = np.shape(dyn)[0]
        nt = np.shape(dyn)[1]
        dyn = dyn - np.mean(dyn)  # subtract mean

        #He does the windowing first
        if window is not None:
            # Window the dynamic spectrum
            if window == 'hanning':
                cw = np.hanning(np.floor(window_frac*nt))
                sw = np.hanning(np.floor(window_frac*nf))
            elif window == 'hamming':
                cw = np.hamming(np.floor(window_frac*nt))
                sw = np.hamming(np.floor(window_frac*nf))
            elif window == 'blackman':
                cw = np.blackman(np.floor(window_frac*nt))
                sw = np.blackman(np.floor(window_frac*nf))
            elif window == 'bartlett':
                cw = np.bartlett(np.floor(window_frac*nt))
                sw = np.bartlett(np.floor(window_frac*nf))
            else:
                print('Window unknown.. Please add it!')
            chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),
                                    np.ones([nt-len(cw)]))
            subint_window = np.insert(sw, int(np.ceil(len(sw)/2)),
                                      np.ones([nf-len(sw)]))
            dyn = np.multiply(chan_window, dyn)
            dyn = np.transpose(np.multiply(subint_window,
                                           np.transpose(dyn)))

        #Some FFT stuff
            
        # find the right fft lengths for rows and columns
        nrfft = int(2**(np.ceil(np.log2(nf))+1))
        ncfft = int(2**(np.ceil(np.log2(nt))+1))
        
        self.nrfft_sv = nrfft
        self.ncfft_sv = ncfft
        
        dyn = dyn - np.mean(dyn)  # subtract the mean flux from tehe dyn
        
        #
        if prewhite: #prewhite is always true
            print("Prewhitening")
            simpw = convolve2d([[1, -1], [-1, 1]], dyn, mode='valid')
        else:
            print("Turning off Prewhitening")
            simpw = dyn
        
        #Behold, the all powerful FFT!
        simf = np.fft.fft2(simpw, s=[nrfft, ncfft]) #raw fft...
        print("Pre conjugation:", simf[0],  simf[1],  simf[2] )
        
        if(conj_arg):
            print("Computing Conjugate of", filenom) #lcp
            simf = np.conjugate(simf)
            print("Post conjugation:", simf[0],  simf[1],  simf[2] )
            #print(simf)
        else:
            print("No Changes:", simf[0],  simf[1],  simf[2] )
            print("Taking", filenom, "as is")
            #print(simf)
            
        self.lamsspec_fft_samp_1 = simf
        
        sec = np.fft.fftshift(simf)  

        self.lamsspec_fft_samp_2 = sec
        
        if halve: #which is always halve. Discarding nevative delays
            sec = sec[int(nrfft/2):][:]  # crop #

        if halve:
            td = np.array(list(range(0, int(nrfft/2))))
        else:
            td = np.array(list(range(0, int(nrfft))))
        fd = np.array(list(range(int(-ncfft/2), int(ncfft/2))))
        
        self.td_sv=td
        self.fd_sv=fd
        
        fdop = np.reshape(np.multiply(fd, 1e3/(ncfft*self.dt)),
                          [len(fd)])  # in mHz
        tdel = np.reshape(np.divide(td, (nrfft*self.df)),
                          [len(td)])  # in us

        if lamsteps:
            beta = np.divide(td, (nrfft*self.dlam))  # in m^-1
        
        self.lamsspec_fft_samp_3 = sec #These things need fdop and tdop...
                                       #we multiply this...
                                       #Then take imag on the other side
                
        #in order to postdark you must have...
        #ncfft, and fd
        #nrfft, and td
        #they compute vec1 and vec2
        
        '''
        if prewhite:#Now post-darken
            if halve:
                vec1 = np.reshape(np.power(np.sin(
                                  np.multiply(sc.pi/ncfft, fd)), 2),
                                  [ncfft, 1])
                vec2 = np.reshape(np.power(np.sin(
                                  np.multiply(sc.pi/nrfft, td)), 2),
                                  [1, int(nrfft/2)])
                postdark = np.transpose(vec1*vec2)
                postdark[:, int(ncfft/2)] = 1
                postdark[0, :] = 1
                sec = np.divide(sec, postdark)
            else:
                raise RuntimeError('Cannot apply prewhite to full frame')
        else:
            print("Not Post darkening")
        
        self.lamsspec_fft_samp_4 = sec #This is complex bro
        
        if(real_arg):
            print("Taking REAL part of Sec Spec")
            print("Mean of real part of the secondary spectrum pre log:", np.mean(sec.real))
            print("Median of real part of the secondary spectrum pre log:", np.median(sec.real))
            
            if(neg_filt):
                sec = sec.real
                print("Rationalizing Negative Elements. Our Expectation is that the white specs will diappear.")
                
                print("Just get rid of speckle please")
                img_std = np.std(sec)
                
                print("1 Standard deviation of sec:", img_std)
                n = neg_filt_std
                
                for t in range(0, len(sec)):
                    for u in range(0, len(sec[0])):
                        if(np.sign(sec[t][u]) < 0): #negative number
                            sec[t][u] = np.sign(sec[t][u])*np.log10(abs(sec[t][u]))   #not normalized over 1 standard deviation
                        elif(np.sign(sec[t][u]) > 0):
                            sec[t][u] = np.log10(sec[t][u])
                            
                #This code block sets the attribute
                if input_dyn is None:
                    if lamsteps:
                        self.lamsspec = sec

                    elif trap:
                        self.trapsspec = sec
                    else:
                        self.sspec = sec

                    self.fdop = fdop
                    self.tdel = tdel
                    if lamsteps:
                        self.beta = beta
                    if plot:
                        self.plot_sspec(lamsteps=lamsteps, trap=trap)
                else:
                    if lamsteps:
                        beta = np.divide(td, (nrfft*self.dlam))  # in m^-1
                        yaxis = beta
                    else:
                        yaxis = tdel
                    return fdop, yaxis, sec
                
                self.plot_sspec(lamsteps=True, title= "Approach 1 mod 1")
                            
            else:
                self.fft_samp_5 = sec.real #why this sample will never work because
                
                #imag(l*R) != imag(l*) x imag(R)
                #a x b where a and b are complex 
                
                sec = np.log10(sec.real) #The diffrences would be very small. Collapese into either real of false
            
            print("Mean of real part of the secondary spectrum post log:", np.mean(sec))
            print("Median of real part of the secondary spectrum post log:", np.median(sec))
           
        else:
            print("Taking IMAG part of Sec Spec")                  
            print("Mean of imag part of the secondary spectrum pre log:", np.mean(sec.imag))
            print("Median of imag part of the secondary spectrum pre log:", np.median(sec.imag))   
            
            if(neg_filt):
                sec = sec.real
                print("Rationalizing Negative Elements. Our Expectation is that the white specs will diappear.")
                
                print("Just get rid of speckle please")
                img_std = np.std(sec)
                
                print("1 Standard deviation of sec:", img_std)
                n = neg_filt_std
                
                for t in range(0, len(sec)):
                    for u in range(0, len(sec[0])):
                        if(np.sign(sec[t][u]) < 0): #negative number
                            sec[t][u] = np.sign(sec[t][u])*np.log10(abs(sec[t][u]))   #not normalized over 1 standard deviation
                        elif(np.sign(sec[t][u]) > 0):
                            sec[t][u] = np.log10(sec[t][u]) 
                            
                #This code block sets the attribute
                if input_dyn is None:
                    if lamsteps:
                        self.lamsspec = sec

                    elif trap:
                        self.trapsspec = sec
                    else:
                        self.sspec = sec

                    self.fdop = fdop
                    self.tdel = tdel
                    if lamsteps:
                        self.beta = beta
                    if plot:
                        self.plot_sspec(lamsteps=lamsteps, trap=trap)
                else:
                    if lamsteps:
                        beta = np.divide(td, (nrfft*self.dlam))  # in m^-1
                        yaxis = beta
                    else:
                        yaxis = tdel
                    return fdop, yaxis, sec
                
                self.plot_sspec(lamsteps=True, title= "Approach 1 mod 1")
                
            else:
                self.fft_samp_5 = sec.imag
                sec = np.log10(sec.imag) #The diffrences would be very small. Collapese into either real of false
            print("Mean of imag part of the secondary spectrum post log:", np.mean(sec)) #This should no longer be nan
            print("Median of imag part of the secondary spectrum post log:", np.median(sec))
                
        '''                      
        #This code block sets the attribute
        #I could say that if you sample sspec from birefring method it will be complex valued.
        if input_dyn is None:
            if lamsteps:
                self.lamsspec = sec
                
            elif trap:
                self.trapsspec = sec
            else:
                self.sspec = sec

            self.fdop = fdop
            self.tdel = tdel
            if lamsteps:
                self.beta = beta
            if plot:
                self.plot_sspec(lamsteps=lamsteps, trap=trap)
        else:
            if lamsteps:
                beta = np.divide(td, (nrfft*self.dlam))  # in m^-1
                yaxis = beta
            else:
                yaxis = tdel
            return fdop, yaxis, sec
        
        
    def calc_acf(self, scale=False, input_dyn=None, plot=True):
        """
        Calculate autocovariance function. All that shit is there. You just have to go get it!
        """

        if input_dyn is None:
            # mean subtracted dynspec
            arr = cp(self.dyn) - np.mean(self.dyn[is_valid(self.dyn)])
            nf = self.nchan
            nt = self.nsub
        else:
            arr = input_dyn
            nf = np.shape(input_dyn)[0]
            nt = np.shape(input_dyn)[1]
            
        arr = np.fft.fft2(arr, s=[2*nf, 2*nt])  # zero-padded. Here is your 2d FFT again
        arr = np.abs(arr)  # absolute value
        arr **= 2  # Squared manitude
        arr = np.fft.ifft2(arr)
        arr = np.fft.fftshift(arr)
        arr = np.real(arr)  # real component, just in case
        
        if input_dyn is None:
            self.acf = arr
        else:
            return arr

    def crop_dyn(self, fmin=0, fmax=np.inf, tmin=0, tmax=np.inf):
        """
        Crops dynamic spectrum in frequency to be between fmin and fmax (MHz)
            and in time between tmin and tmax (mins)
        """

        # Crop frequencies
        crop_array = np.array((self.freqs > fmin)*(self.freqs < fmax))
        self.dyn = self.dyn[crop_array, :]
        self.freqs = self.freqs[crop_array]
        self.nchan = len(self.freqs)
        self.bw = round(max(self.freqs) - min(self.freqs) + self.df, 2)
        self.freq = round(np.mean(self.freqs), 2)

        # Crop times
        tmin = tmin*60  # to seconds
        tmax = tmax*60  # to seconds
        if tmax < self.tobs:
            self.tobs = tmax - tmin
        else:
            self.tobs = self.tobs - tmin
        crop_array = np.array((self.times > tmin)*(self.times < tmax))
        self.dyn = self.dyn[:, crop_array]
        self.nsub = len(self.dyn[0, :])
        self.times = np.linspace(self.dt/2, self.tobs - self.dt/2, self.nsub)
        self.mjd = self.mjd + tmin/86400

    def zap(self, method='median', manual = False, 
            plot_verb=False , man_blank_arr=[],
            nchan_arr_purge=[], duckie_loc = [] ,sigma=7, m=3):
        """
        Basic zapping of dynamic spectrum
        """

        if method == 'median':
            print("Applying median!")
            d = np.abs(self.dyn - np.median(self.dyn[~np.isnan(self.dyn)]))
            mdev = np.median(d[~np.isnan(d)])
            s = d/mdev
            count = 0
            #i want to know how many points is it actually purging?
            #for i in range(0, len(self.dyn)):
            #    for j in range(0, len(self.dyn[0])):
            #        if(self.dyn[i][j] )
            self.dyn[s > sigma] = np.nan #for points of which values are greater than sigma s > sigma set it to nan...
        elif method == 'medfilt':
            print("Applying median filter!")
            self.dyn = medfilt(self.dyn, kernel_size=m)
        if manual:
            print("Initiating Manual Blanking Procedure")
            print("Blanking these channels:", man_blank_arr)
            self.plot_dyn(man_blank_arr=man_blank_arr)
            print("Match these:", self.freqs)
            print("To these:",man_blank_arr)
            
            found_l3 = [x for x in man_blank_arr if x in self.freqs]
            n_found = [x for x in man_blank_arr if x not in self.freqs]
            
            #print("Matched!", found_l3)
            print(len(found_l3), "Matched")
            
            if(len(found_l3) != len(man_blank_arr)):
                print(len(man_blank_arr) - len(found_l3), "chans not found")
                print(n_found, " not found. Find the closes match please.")
            
            print("Size of Dyn")
            print(len(self.dyn), "x", len(self.dyn[0]))
            
            print("Blanking")
            for chans in nchan_arr_purge:
                if plot_verb:
                    plt.plot(self.dyn[chans])
                    plt.axvline(x=1801)
                    plt.title("Pre Blank")
                    plt.show()
                
                for ts in range(0, len(self.dyn[chans])):
                    self.dyn[chans][ts] = np.NaN
                    
                if plot_verb:
                    plt.plot(self.dyn[chans])
                    plt.axvline(x=1801)
                    plt.title("Post Blank. The whole channel should be 0.")
                    plt.show()
                    
            #print("x axis:", self.times/60)
            #print("y axis:", self.freqs)
            
            center_point_x = []
            center_point_y = []
            
            for locs in duckie_loc:
                #print("These are your requested coordinates:", locs)
                print("For each coordinate, match to the closest x,y in times and freq")
                
                print("Time coordinate:",locs[0]) #Time
                print("Frequency coordinate:",locs[1]) #Frequency
                
                time_bin = []
                freq_bin = []
                
                for pos_times in self.times/60:
                    time_bin.append([ abs(locs[0]-pos_times), pos_times])
                    
                for pos_freqs in self.freqs:
                    freq_bin.append([ abs(locs[1]-pos_freqs), pos_freqs])

                t_bin = np.array(time_bin)
                freq_bin = np.array(freq_bin)
                
                distances_t_bin = t_bin[:, 0]
                distances_f_bin = freq_bin[:, 0]
                
                #print("Array of time distancs", distances_t_bin)
                #print("Minimum distance", np.min(distances_t_bin))
                #print("Location:", np.where(distances_t_bin == np.min(distances_t_bin))[0][0]         )               
                #print("The x coordinate:", t_bin[np.where(distances_t_bin == np.min(distances_t_bin))[0][0]][1])
                
                center_point_x.append(np.where(distances_t_bin == np.min(distances_t_bin))[0][0])
                c_p_x = np.where(distances_t_bin == np.min(distances_t_bin))[0][0]
                #print("\n\n")
                
                #print("Array of freq distancs", distances_f_bin)
                #print("Minimum distance", np.min(distances_f_bin))
                #print("Location:", np.where(distances_f_bin == np.min(distances_f_bin))[0][0])               
                #print("The x coordinate:", freq_bin[np.where(distances_f_bin == np.min(distances_f_bin))[0][0]][1])
                
                center_point_y.append(np.where(distances_f_bin == np.min(distances_f_bin))[0][0])
                c_p_y = np.where(distances_f_bin == np.min(distances_f_bin))[0][0]
                
                print(c_p_x, c_p_y)
                
                for h in range(c_p_x-42, c_p_x+42):
                    for j in range(c_p_y-90, c_p_y+90):
                        self.dyn[j][h]  = np.NaN
                
                print("\n")
                #distances = time_bin[0:len(time_bin), 0]
                #print(distances[0:4])
                #print(time_bin[0:4])
                #It is essentially an image
                #What is x location?
                #for each
                #what is y location?
                #blank out a region of 15 MHz wide, 8 minutes long
            
            print(center_point_x, center_point_y)
            
            plt.matshow(self.dyn)
            
            for xpoint in center_point_x:
                plt.axvline(x=xpoint)
                
            for ypoint in center_point_y:
                plt.axhline(y=ypoint)
                
            #plt.axvline(x=center_point_x[0])
            #plt.axvline(x=center_point_x[1])
            #plt.axvline(x=center_point_x[2])
            #plt.axhline(y=center_point_y[0])
            #plt.axhline(y=center_point_y[1])
            #plt.axhline(y=center_point_y[2])
            plt.show()           


    def scale_dyn(self, scale='lambda', factor=1, window_frac=0.1,
                  window='hanning'):
        """
        Scales the dynamic spectrum along the frequency axis,
            with an alpha relationship
        """

        if scale == 'factor':
            # scale by some factor
            print("This doesn't do anything yet")
        elif scale == 'lambda':
            # function to convert dyn(feq,t) to dyn(lameq,t)
            # fbw = fractional BW = BW / center frequency
            arin = cp(self.dyn)  # input array
            nf, nt = np.shape(arin)
            freqs = cp(self.freqs)
            lams = np.divide(sc.c, freqs*10**6)
            dlam = np.max(np.abs(np.diff(lams)))
            lam_eq = np.arange(np.min(lams), np.max(lams), dlam)
            self.dlam = dlam
            feq = np.divide(sc.c, lam_eq)/10**6
            arout = np.zeros([len(lam_eq), int(nt)])
            for it in range(0, nt):
                f = interp1d(freqs, arin[:, it], kind='cubic')
                arout[:, it] = f(feq)
            self.lamdyn = np.flipud(arout)
            self.lam = np.flipud(lam_eq)
        elif scale == 'trapezoid':
            dyn = cp(self.dyn)
            dyn -= np.mean(dyn)
            nf = np.shape(dyn)[0]
            nt = np.shape(dyn)[1]
            if window is not None:
                # Window the dynamic spectrum
                if window == 'hanning':
                    cw = np.hanning(np.floor(window_frac*nt))
                    sw = np.hanning(np.floor(window_frac*nf))
                elif window == 'hamming':
                    cw = np.hamming(np.floor(window_frac*nt))
                    sw = np.hamming(np.floor(window_frac*nf))
                elif window == 'blackman':
                    cw = np.blackman(np.floor(window_frac*nt))
                    sw = np.blackman(np.floor(window_frac*nf))
                elif window == 'bartlett':
                    cw = np.bartlett(np.floor(window_frac*nt))
                    sw = np.bartlett(np.floor(window_frac*nf))
                else:
                    print('Window unknown.. Please add it!')
                chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),
                                        np.ones([nt-len(cw)]))
                subint_window = np.insert(sw, int(np.ceil(len(sw)/2)),
                                          np.ones([nf-len(sw)]))
                dyn = np.multiply(chan_window, dyn)
                dyn = np.transpose(np.multiply(subint_window,
                                               np.transpose(dyn)))
            arin = dyn  # input array
            nf, nt = np.shape(arin)
            scalefrac = 1/(max(self.freqs)/min(self.freqs))
            timestep = max(self.times)*(1 - scalefrac)/(nf + 1)  # time step
            trapdyn = np.empty(shape=np.shape(arin))
            for ii in range(0, nf):
                idyn = arin[ii, :]
                maxtime = max(self.times)-(nf-(ii+1))*timestep
                # How many times to resample to, for a given frequency
                inddata = np.argwhere(self.times <= maxtime)
                # How many trailing zeros to add
                indzeros = np.argwhere(self.times > maxtime)
                # Interpolate line
                newline = np.interp(
                          np.linspace(min(self.times), max(self.times),
                                      len(inddata)), self.times, idyn)

                newline = list(newline) + list(np.zeros(np.shape(indzeros)))
                trapdyn[ii, :] = newline
            self.trapdyn = trapdyn

    def info(self):
        """
        print properties of object
        """

        print("\tOBSERVATION PROPERTIES\n")
        print("Lets make some Fourier Images!")
        print("Filename:\t\t\t{0}".format(self.name))
        print("MJD:\t\t\t\t{0}".format(self.mjd))
        print("Centre frequency (MHz):\t\t{0}".format(self.freq))
        print("Bandwidth (MHz):\t\t{0}".format(self.bw))
        print("Channel bandwidth (MHz):\t{0}".format(self.df))
        print("Integration time (s):\t\t{0}".format(self.tobs))
        print("Integration time (hrs):\t\t{0}".format(self.tobs/3600))
        #print("Integration time (hrs):\t\t{0}".format(self.tobs)/3600)
        print("Subintegration time (s):\t{0}".format(self.dt))
        return


class BasicDyn():
    """
    Define a basic dynamic spectrum object from an array of fluxes
        and other variables, which can then be passed to the dynspec
        class to access its functions with:
    BasicDyn_Object = BasicDyn(dyn)
    Dynspec_Object = Dynspec(BasicDyn_Object)
    """

    def __init__(self, dyn, name="BasicDyn", header=["BasicDyn"], times=[],
                 freqs=[], nchan=None, nsub=None, bw=None, df=None,
                 freq=None, tobs=None, dt=None, mjd=None):

        # Set parameters from input
        if times.size == 0 or freqs.size == 0:
            raise ValueError('must input array of times and frequencies')
        self.name = name
        self.header = header
        self.times = times
        self.freqs = freqs
        self.nchan = nchan if nchan is not None else len(freqs)
        self.nsub = nsub if nsub is not None else len(times)
        self.bw = bw if bw is not None else abs(max(freqs)) - abs(min(freqs))
        self.df = df if df is not None else freqs[1] - freqs[2]
        self.freq = freq if freq is not None else np.mean(np.unique(freqs))
        self.tobs = tobs
        self.dt = dt
        self.mjd = mjd
        self.dyn = dyn
        return


class MatlabDyn():
    """
    Imports simulated dynamic spectra from Matlab code by Coles et al. (2010)
    """

    def __init__(self, matfilename):

        self.matfile = loadmat(matfilename)  # reads matfile to a dictionary
        try:
            self.dyn = self.matfile['spi']
        except NameError:
            raise NameError('No variable named "spi" found in mat file')

        try:
            dlam = float(self.matfile['dlam'])
        except NameError:
            raise NameError('No variable named "dlam" found in mat file')
        # Set parameters from input
        self.name = matfilename.split()[0]
        self.header = [self.matfile['__header__'], ["Dynspec loaded \
                       from Matfile {}".format(matfilename)]]
        self.dt = 2.7*60
        self.freq = 1400
        self.nsub = int(np.shape(self.dyn)[0])
        self.nchan = int(np.shape(self.dyn)[1])
        lams = np.linspace(1, 1+dlam, self.nchan)
        freqs = np.divide(1, lams)
        self.freqs = self.freq*np.linspace(np.min(freqs), np.max(freqs),
                                           self.nchan)
        self.bw = max(self.freqs) - min(self.freqs)
        self.times = self.dt*np.arange(0, self.nsub)
        self.df = self.bw/self.nchan
        self.tobs = float(self.times[-1] - self.times[0]) #
        self.mjd = 50000.0  # dummy.. Not needed
        self.dyn = np.transpose(self.dyn)

        return


class SimDyn():
    """
    Imports Simulation() object from scint_sim to Dynspec class
    """

    def __init__(self, sim, freq=1400, dt=0.5, mjd=50000):

        self.name =\
            'sim:mb2={0},ar={1},psi={2},dlam={3}'.format(sim.mb2, sim.ar,
                                                         sim.psi, sim.dlam)
        if sim.lamsteps:
            self.name += ',lamsteps'

        self.header = self.name
        self.dyn = sim.spi
        dlam = sim.dlam

        self.dt = dt
        self.freq = freq
        self.nsub = int(np.shape(self.dyn)[0])
        self.nchan = int(np.shape(self.dyn)[1])
        lams = np.linspace(1, 1+dlam, self.nchan)
        freqs = np.divide(1, lams)
        self.freqs = self.freq*np.linspace(np.min(freqs), np.max(freqs),
                                           self.nchan)
        self.bw = max(self.freqs) - min(self.freqs)
        self.times = self.dt*np.arange(0, self.nsub)
        self.df = self.bw/self.nchan
        self.tobs = float(self.times[-1] - self.times[0])
        self.mjd = mjd
        self.dyn = np.transpose(self.dyn)
        return


def sort_dyn(dynfiles, outdir=None, min_nsub=10, min_nchan=50, min_tsub=10,
             min_freq=0, max_freq=5000, remove_nan_sspec=False, verbose=True,
             max_frac_bw=2):
    """
    Sorts dynamic spectra into good and bad files based on some conditions
    """

    if verbose:
        print("Sorting dynspec files in {0}".format(split(dynfiles[0])[0]))
        n_files = len(dynfiles)
        file_count = 0
    if outdir is None:
        outdir, dummy = split(dynfiles[0])  # path of first dynspec
    bad_files = open(outdir+'/bad_files.txt', 'w')
    good_files = open(outdir+'/good_files.txt', 'w')
    bad_files.write("FILENAME\t REASON\n")
    for dynfile in dynfiles:
        if verbose:
            file_count += 1
            print("{0}/{1}\t{2}".format(file_count, n_files,
                  split(dynfile)[1]))
        # Read in dynamic spectrum
        dyn = Dynspec(filename=dynfile, verbose=False, process=False)
        if dyn.freq > max_freq or dyn.freq < min_freq:
            # outside of frequency range
            if dyn.freq < min_freq:
                message = 'freq<{0} '.format(min_freq)
            elif dyn.freq > max_freq:
                message = 'freq>{0}'.format(max_freq)
            bad_files.write("{0}\t{1}\n".format(dynfile, message))
            continue
        if dyn.bw/dyn.freq > max_frac_bw:
            # bandwidth too large
            bad_files.write("{0}\t frac_bw>{1}\n".format(dynfile, max_frac_bw))
            continue
        # Start processing
        dyn.trim_edges()  # remove band edges
        if dyn.nchan < min_nchan or dyn.nsub < min_nsub:
            # skip if not enough channels/subints
            message = ''
            if dyn.nchan < min_nchan:
                message += 'nchan<{0} '.format(min_nchan)
            if dyn.nsub < min_nsub:
                message += 'nsub<{0}'.format(min_nsub)
            bad_files.write("{0}\t {1}\n".format(dynfile, message))
            continue
        elif dyn.tobs < 60*min_tsub:
            # skip if observation too short
            bad_files.write("{0}\t tobs<{1}\n".format(dynfile, min_tsub))
            continue
        dyn.refill()  # linearly interpolate zero values
        dyn.correct_dyn()  # correct for bandpass and gain variation
        dyn.calc_sspec()  # calculate secondary spectrum
        # report error and proceed
        if np.isnan(dyn.sspec).all():  # skip if secondary spectrum is all nan
            bad_files.write("{0}\t sspec_isnan\n".format(dynfile))
            continue
        # Passed all tests so far - write to good_files.txt!
        good_files.write("{0}\n".format(dynfile))
    bad_files.close()
    good_files.close()
    return
