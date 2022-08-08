#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Morlet-Wave damping identification method
==================================================
@author: Ivan Tomac

"""
import numpy as np
import mwdi as mw
import matplotlib.pyplot as plt

# pylint: disable=no-name-in-module
from scipy.special import erf
from warnings import warn

class ExtendedMW(object):
    """
    Extended Morlet-Wave damping identification method
    ==================================================
    Literature:
        [1]: Slavič, J., Boltežar, M., Damping identification with the Morlet-wave,
             Mechanical Systems and Signal Processing, 2011, 5
             doi: 10.1016/j.ymssp.2011.01.008
        [2]: Tomac, I., Lozina, Ž., Sedlar, D., Extended Morlet-Wave damping identification
             method, International journal of mechanical sciences, 2017, 127
             doi: 10.1016/j.ijmecsci.2017.01.013
        [3]: Tomac, I., Slavič, J., Damping identification based on a high-speed camera,
             Mechanical Systems and Signal Processing, 2022, 166
             doi: 10.1016/j.ymssp.2021.108485
    """

    def __init__(self, fs=None, free_response=None, nat_freqs=(None, None), time_spread=(7, 14),
                 k_range=(10, 400)):
        """
        Constructor of eMWDI object which sets initial parameters for the method.

            Self:
        n1              - n1 time spread parameter
        n2              - vector of n2 parameters
        k               - vector of all k parameters
        omega_estimated - estimated natural frequency (omega estimated)
        time            - time vector
        fs              - sampling frequency
        irf             - impulse response function
        zeta_detected   - 2d array of identified damping ratios
        omega_detected  - 2d array of identified natural frequencies
        zeta            - estimated damping ratio
        omega           - estimated natural frequency
        k_est           - k parameter for the estimated damping ratio

        :param fs:               - sampling frequency of the signal
        :param free_response:    - impulse response function of the mechanical system (SDOF)
        :param nat_freqs:        - natural frequencies (main and the closest one) in radians per second
        :param time_spread:      - tuple containing n1 and n2 time spread parameters
        :param k_range:          - tuple setting range of k parameter that define number of wave
                                  function cycles
        :param verb:             - enable/disable messages
        """
        # Initialization
        dif = np.diff(time_spread)
        if dif == 5:
            spread_jump = 1 + 1
        elif 5 < dif <= 7:
            spread_jump = 1 + 2
        elif 7 < dif < 10:
            spread_jump = 1 + 3
        elif dif >= 10:
            spread_jump = 1 #+ 4
        else:
            print("Greska u parametrima n_1 i n_2!")
            return None

        self.identifier_morlet_wave = mw.MorletWave(free_response, fs)

        self.n1 = time_spread[0]
        self.n2 = np.arange(time_spread[0]+spread_jump, time_spread[1]+1)
        self.k = np.arange(k_range[0], k_range[1]+1)

        self.omega_estimated = nat_freqs[0]
        self.omega_neighbor = nat_freqs[1]

        self.irf = free_response
        self.fs = fs

        self.zeta_detected = np.zeros((self.n2.size, self.k.size))
        self.zeta_detected2 = np.zeros((self.k.size))
        self.omega_detected = np.zeros(self.k.size)

        self.zeta = 0
        self.omega = 0
        self.k_est = 0
        self.X = 0
        self.phi = 0

        return None

    def plot(self, plt1=True, plt2=True, plt3=False):
        """
        Visualization of damping identification. Three types of plots are available that are
        selected using the method's argument.

        :param plt1: - plots mean values along n2 axis in function of k with standard deviation from
                       the mean value
        :param plt2: - plots just the standard deviation
        :param plt3: - plots the 3D map of the all damping ratios identified in the given range of
                       parameter n2 and k
        """
        zt_me = np.mean(self.zeta_detected, 0)
        zt_st = np.std(self.zeta_detected, 0)
        if plt1:
            plt.figure(figsize=(8, 6), dpi=80)
            plt.plot(self.k, zt_me)
            plt.plot(self.k, zt_me + zt_st, '--', linewidth=1)
            plt.plot(self.k, zt_me - zt_st, 'C1--', linewidth=1)
            plt.plot([self.k_est, self.k_est], [np.nanmin(zt_me-zt_st), np.nanmax(zt_me+zt_st)], \
                        'k-', linewidth=1)
            plt.xlabel(r'$k$')
            plt.ylabel(r'$\bar{\zeta}_{n_2}$')
            plt.title('Mean values of damping along n2 with std')

        if plt2:
            plt.figure(figsize=(8, 6), dpi=80)
            plt.plot(self.k, zt_st)
            plt.plot([self.k_est, self.k_est], [np.nanmin(zt_st), np.nanmax(zt_st)], \
                        'k-', linewidth=1)
            plt.xlabel(r'$k$')
            plt.ylabel(r'$\sigma\left(\bar{\zeta}\right)$')
            plt.title('Std dev from mean damping value')

        if plt3:
            # filters NaN values for all n_2 values.
            flt = np.sum(np.isnan(self.zeta_detected), axis=0) != self.zeta_detected.shape[0]
            X, Y = np.meshgrid(self.k[flt], self.n2)
            Z = self.zeta_detected[:, flt]*100
            fig = plt.figure(figsize=(8, 6), dpi=80)
            ax3 = fig.gca(projection='3d')#plt.axes(projection='3d')
            ax3.plot_surface(X, Y, Z, cmap='viridis', vmin=np.nanmin(Z), vmax=np.nanmax(Z))
            ax3.set(xlabel=r'$k$', ylabel=r'$n_2$', zlabel=r'$\zeta\ (\%)$', title='Damping 3D map')
            ax3.xaxis.set_ticks(self.k[flt])
            ax3.yaxis.set_ticks(self.n2)
        return None

    def estimate(self, verb=True):
        """
        Estimates damping from identified damping values using the detect_damping() method.

        :param verb: - enable/disable messages
        """
        try:
            i = np.nanargmin(np.std(self.zeta_detected, 0))
        except ValueError:
            self.zeta = np.nan
            self.omega = np.nan
            self.k_est = np.nan
            if verb:
                print("Damping not identified!")
            return None

        self.zeta = np.mean(self.zeta_detected, 0)[i]
        self.omega = self.omega_detected[i]
        self.k_est = self.k[i]

        if verb:
            print("k: %d\tzeta: %.4f%%\tomega = %.2f Hz (%.3f s^-1)"
                  % (self.k_est, self.zeta*100, self.omega/(2*np.pi), self.omega))
        return None

    def detect_amplitude(self, verb=True):
        """
        Identify amplitude anf phase for the given natural frequency, damping ratio and k

        :param verb: - enable/disable messages
        """
        if self.zeta is np.nan or self.omega is np.nan:
            self.X = np.nan
            self.phi = np.nan
            if verb:
                print("Input values are NaN.")
            return None

        k = self.k_est
        n = self.n1#2[-1]

        I_tilde = self.identifier_morlet_wave.morlet_integrate(self.omega, n, k) 

        self.X = get_amplitude(k, n, self.zeta, self.omega, np.abs(I_tilde))
        self.phi = -np.angle((-1)**(k) * I_tilde)

        if verb:
            print(f'X: {self.X:.4f}\tphi: {self.phi:.3f} ({np.rad2deg(self.phi):.1f} deg)')
        return None

    def detect_frequency(self, use_estimated=False, verb=False):
        """
        Identify natural frequency by searching the maximal absolute value of the wavelet
        coefficient.

        :param use_estimated:    - do not search for natural frequencies, use estimated instead
        :param verb:             - enable/disable messages
        """
        if use_estimated:
            self.omega_detected = np.full_like(self.k, self.omega_estimated, dtype=float)
            return None

        for i, k_ in enumerate(self.k):
            delta = self.omega_estimated * self.n1 / (4 * k_ * np.pi) # reduction for optimizer
            lim = int(2*np.pi*k_/(self.omega_estimated - delta)*self.fs) + 1   # [2] Eq.(12)
            if lim > self.irf.size:
                # print(lim, self.irf.size, self.omega_estimated, self.fs)
                warn(f'Maximum iterations reached for: k = {k_}, k_hi {self.k[-1]} reduced to {k_-1}.', Warning)
                self.zeta_detected = self.zeta_detected[:, :i]
                self.zeta_detected2 = self.zeta_detected2[:i]
                self.omega_detected = self.omega_detected[:i]
                self.k = self.k[:i]
                return None

            self.omega_detected[i] = self.identifier_morlet_wave.find_natural_frequency(self.omega_estimated, self.n1, k_)

            if self.omega_neighbor is not None:
                test = np.array([self.omega_detected[i], self.omega_neighbor])
                if self.n2[-1]*np.max(test)/(4*np.pi*k_) >= np.abs(np.diff(test)):    # [2] Eq.(24)
                    if verb:
                        print('Frequency resolution is insufficient!')
                        print(np.abs(np.diff(test)))
                        print('k = ', k_)
                        print('n2 = ', self.n2[-1])
                    return None
            if verb:
                print(f'{k_}\t{self.omega_estimated:.2f}\t{self.omega_detected[i]:.2f}')
        return None

    def detect_damp(self, verb=False, init_damping_ratio='auto', method='auto'):
        """
        Method detects damping for given ranges of k and n2 parameters and checks if detected
        damping is feasible. If not then it is set as NaN.

        :param verb: - enable/disable messages
        :param damping_ratio_init: initial search value, if 'auto', the closed_form solution is used
        :param root_finding: root finding algorithm to use: 'auto', 'closed-form' ,'Newton', 'Ridder'
        """
        if self.n1 < 10 and method=='auto':
            method='Newton' # Set Exact method
        else:
            method='closed-form' # Set Closed-form method

        # kitr = 0
        for i, k_ in enumerate(self.k):
            lim = int(2*np.pi*k_/(self.omega_estimated)*self.fs) + 1     # [2] Eq.(12)
            if lim > self.irf.size:
                warn(f'Maximum iterations reached for: k = {k_}, k_hi {self.k[-1]} reduced to {k_-1}.', Warning)
                self.zeta_detected = self.zeta_detected[:, :i]
                self.omega_detected = self.omega_detected[:i]
                self.k = self.k[:i]
                break

            for j, n2_ in enumerate(self.n2):

                if np.isnan(self.omega_detected[i]):
                    self.zeta_detected[j, i] = np.NaN
                    if verb:
                        print("Damping not detected because frequency is not detected.")
                    break

                try:
                    dmp = self.identifier_morlet_wave.identify_damping(w=self.omega_detected[i], \
                            n_1=self.n1, n_2=n2_, k=k_, find_exact_freq=False, root_finding=method, \
                                                damping_ratio_init=init_damping_ratio)
                except Exception:
                    warn(f'Damping not identified for: k:{k_}, n2:{n2_}.', Warning)
                    dmp = np.nan

                if isinstance(dmp, float) and dmp > 0 and dmp != np.inf:
                    self.zeta_detected[j, i] = dmp
                    #if self.n1**2/(8*np.pi*i) < dmp or n2**2/(8*np.pi*i) < dmp:
                    if k_ > self.n1**2/(8*np.pi*dmp):    # [2] Eq.(21)
                        self.zeta_detected[j, i] = np.NaN
                        if verb:
                            print('zeta = ', dmp)
                            print('Basic condition is not met: k <= n^2/(8*pi*zeta)')
                            print('k = ', k_, '\tIteration: ', i)
                else:
                    self.zeta_detected[j, i] = np.NaN

                if verb:
                    print(f'{k_}\t{n2_}\t{self.omega_detected[i]:.2f}\t{self.zeta_detected[j, i]:.6f}')

    def detect_damp2(self, verb=False, init_damping_ratio='auto', method='auto'):
        """
        Method detects damping for given range of k parameter and checks if detected
        damping is feasible. If not then it is set as NaN.

        :param verb: - enable/disable messages
        :param damping_ratio_init: initial search value, if 'auto', the closed_form solution is used
        :param root_finding: root finding algorithm to use: 'auto', 'closed-form' ,'Newton', 'Ridder'
        """
        if self.n1 < 10 and method=='auto':
            method='Newton' # Set Exact method
        else:
            method='closed-form' # Set Closed-form method

        # kitr = 0
        for i, k_ in enumerate(self.k):
            lim = int(2*np.pi*k_/(self.omega_estimated)*self.fs) + 1     # [2] Eq.(12)
            if lim > self.irf.size:
                warn(f'Maximum iterations reached for: k = {k_}, k_hi {self.k[-1]} reduced to {k_-1}.', Warning)
                self.zeta_detected2 = self.zeta_detected2[:i]
                self.omega_detected = self.omega_detected[:i]
                self.k = self.k[:i]
                return None
            # damp.k = k_

            if np.isnan(self.omega_detected[i]):
                self.zeta_detected2[i] = np.NaN
                if verb:
                    print("Damping not detected because frequency is not detected.")
                return None

            try:
                dmp = self.identifier_morlet_wave.identify_damping(w=self.omega_detected[i], \
                                        n_1=self.n1, n_2=self.n2[-1], k=k_, \
                                            find_exact_freq=False, root_finding=method, \
                                                damping_ratio_init=init_damping_ratio)
            except Exception:
                warn(f'Damping not identified: k:{k_}.', Warning)
                dmp = np.nan

            if isinstance(dmp, float) and dmp > 0 and dmp != np.inf:
                self.zeta_detected2[i] = dmp
                #if self.n1**2/(8*np.pi*i) < dmp or n2**2/(8*np.pi*i) < dmp:
                if k_ > self.n1**2/(8*np.pi*dmp):    # [2] Eq.(21)
                    self.zeta_detected2[i] = np.NaN
                    if verb:
                        print('zeta = ', dmp)
                        print('Basic condition is not met: k <= n^2/(8*pi*zeta)')
                        print('k = ', k_, '\tIteration: ', i)
            else:
                self.zeta_detected2[i] = np.NaN

            if verb:
                print(f'{k_}\t{self.omega_detected[i]:.2f}\t{self.zeta_detected[i]:.6f}')
        return None

def get_amplitude(k, n, damping_ratio, omega, I_tilde):
    """
    Determines amplitude from the Morlet-integral.

    :param k: number of oscillations
    :param n: time spread parameter
    :param damping_ratio: damping ratio
    :param omega: circular natural frequency
    :param I_tilde: Morlet integral
    :return: amplitude constant
    """
    const = (np.pi/2)**0.75 * np.sqrt(k / (n * omega))
    e_1 = 2*k*np.pi*damping_ratio / (n*np.sqrt(1 - damping_ratio**2))
    e_2 = 0.25 * n
    err = erf(e_1 + e_2) - erf(e_1 - e_2)
    I_abs = np.exp(e_1**2 - 0.5*n*e_1)
    return np.abs(I_tilde) / (const * I_abs * err)

if __name__ == "__main__":
    fs = 64
    n = 48 * fs
    T = n / fs
    t = np.arange(n) / fs

    w = 2*np.pi
    zeta = 0.01
    sig = np.cos(w * np.sqrt(1 - zeta**2) * t) * np.exp(-zeta * w * t)
    sig += np.random.default_rng().normal(0, 0.01**0.5, n)

    identifier = ExtendedMW(fs, sig, (w+0.01, None))

    identifier.detect_frequency()
    identifier.detect_damp()
    identifier.estimate()
    # identifier.plot()
