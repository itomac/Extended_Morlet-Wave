#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Morlet-Wave damping identification method
==================================================
@author: Ivan Tomac
"""
import numpy as np
# from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
# pylint: disable=no-name-in-module
from scipy.special import erf
from MorletDamping.morletdamping import MorletDamping

class ExtendedMW(object):
    """
    Extended Morlet-Wave damping identification method
    ==================================================
    Literature:
        [1]: Tomac, I., Lozina, Ž., Sedlar, D., Extended Morlet-Wave damping identification method
             International journal of mechanical sciences, 117 (2017), 31-40
             doi:10.1016/j.ijmecsci.2017.01.013
        [2]: Slavič, J., Boltežar, M., Damping identification with the Morlet-wave, Mechanical
             Systems and Signal Processing, 2011, Volume 25, Issue 5, July 2011, Pages 1632-1645,
             doi: 10.1016/j.ymssp.2011.01.008
    """

    def __init__(self, fs=None, irf=None, nat_freqs=(None, None), time_spread=(7, 14),
                 num_cycls_range=None):
        """
        Constructor of eMWDI object which sets initial parameters for the method.

        Args:
            fs               - sampling frequency of the signal
            irf              - impulse response function of the mechanical system (SDOF)
            nat_freqs        - natural frequencies (main and the closest one) in radians per second
            time_spread      - tuple containing n1 and n2 time spread parametes
            num_cycls_range  - tuple setting range of k parameter that define number of wave
                               function cycles
            verb             - enable/disable messages
        Self:
            n1              - n1 time spread parameter
            n2              - vector of n2 parameters
            k               - vector of all k parameters
            omega_estimated - estimated natural frequency (omega estimated)
            time            - time vector
            fs              - sampling frequency
            irf             - impulse response function
            zeta_detected   - 2d array of identified dampig ratios
            omega_detected  - 2d array of identified natural frequencies
            zeta            - estimated damping ratio
            omega           - estimated natural frequency
            k_est           - k parameter for the estimated damping ratio

        """
        # Initialisation
        if time_spread[0] == 5 and time_spread[1] == 10:
            spread_jump = 1 + 1
        elif time_spread[0] == 7 and time_spread[1] == 14:
            spread_jump = 1 + 2
        elif time_spread[0] == 10 and time_spread[1] == 20:
            spread_jump = 1 + 4 # it is noticed that for lower damping 4 is slightly better
        else:
            spread_jump = 1

        self.n1 = time_spread[0]
        self.n2 = np.arange(time_spread[0]+spread_jump, time_spread[1]+1)
        self.k = np.arange(num_cycls_range[0], num_cycls_range[1]+1)

        self.omega_estimated = nat_freqs[0]
        self.omega_next = nat_freqs[1]

        self.irf = irf
        self.fs = fs

        self.zeta_detected = np.zeros((self.n2.size, self.k.size))
        self.omega_detected = np.zeros((self.n2.size, self.k.size))

        self.zeta = 0
        self.omega = 0
        self.k_est = 0
        self.X = 0
        self.phi = 0

    def plot(self, plt1=True, plt2=True, plt3=True):
        """
        Visualisation of damping identification. Three types of plots are available that are
        selected using the method's argument.
        Args:
            plt1 - plots mean values along n2 axis in function of k with standard deviation from
                the mean value
            plt2 - plots just the standard deviation
            plt3 - plots the 3D map of the all damping ratios identified in the given range of
                   parametes n2 and k
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
            X, Y = np.meshgrid(self.k, self.n2)
            Z = self.zeta_detected*100
            fig = plt.figure(figsize=(8, 6), dpi=80)
            ax3 = fig.gca(projection='3d')#plt.axes(projection='3d')
            ax3.plot_surface(X, Y, Z, cmap='viridis', vmin=np.nanmin(Z), vmax=np.nanmax(Z))
            ax3.set(xlabel=r'$k$', ylabel=r'$n_2$', zlabel=r'$\zeta\ (\%)$', title='Damping 3D map')
            ax3.xaxis.set_ticks(self.k)
            ax3.yaxis.set_ticks(self.n2)

    def estimate(self, verb=True):
        """
        Estimates damping from identified damping values usign the detect() method.

        Args:
            verb - enable/disable meaasges
        """
        try:
            i = np.nanargmin(np.std(self.zeta_detected, 0))
        except ValueError:
            self.zeta = np.nan
            self.omega = np.nan
            self.k_est = np.nan
            if verb:
                print("Damping not identified!")
            return

        self.zeta = np.mean(self.zeta_detected, 0)[i]
        self.omega = np.mean(self.omega_detected, 0)[i]
        self.k_est = self.k[i]

        if verb:
            print("k: %d\tzeta: %.4f%%\tomega = %.2f Hz (%.3f s^-1)"
                  % (self.k_est, self.zeta*100, self.omega/(2*np.pi), self.omega))

    def detect_amplitude(self, verb=True):
        """
        Identify amplitude anf phase for the given natural frequency, damping ratio and k
        """
        if self.zeta is np.nan or self.omega is np.nan:
            self.X = np.nan
            self.phi = np.nan
            if verb:
                print("Input values are NaN.")
            return

        k = self.k_est
        n = self.n2[-1]

        damp = MorletDamping(self.irf, self.fs, k, self.n1, n)
        damp.set_int_method(np.trapz)

        I = damp.morlet_integrate(n, self.omega)

        div = (2 * np.pi**3)**0.25 * np.sqrt(k / (n * self.omega)) *\
                np.exp(np.pi * k * self.zeta * (4*np.pi * k * self.zeta - n**2) / n**2) *\
                (erf(2 * np.pi * k * self.zeta / n + n / 4) -\
                 erf(2 * np.pi * k * self.zeta / n - n / 4))

        self.X = np.abs(I) / div
        if np.mod(k, 2) == 0:
            self.phi = -np.angle(I)
        else:
            self.phi = np.pi - np.angle(I)
        if verb:
            print("X: %.4e\tphi = %.4f (%.2f deg)" % (self.X, self.phi, self.phi*180/np.pi))

    def detect_frequency(self, use_estimated=False, verb=False):
        """
        Identify natural frequency by searching the maximal absolute value of the wavelet
        coefficient.

        Args:
            use_estimated    - do not search for natural frequencies, use estimated insted
            verb             - enable/disable messages
        """
        if use_estimated:
            self.omega_detected = self.omega_estimated * np.ones((self.n2.size, self.k.size))
            return
        # This part of code defines search region for the methods that requier region
        # instead of starting point.
        ratio = 2*np.log2(61/60) # omega_upper / omega_center (arbitrary - 1Hz on 60Hz)

        if self.omega_next is not None:
            gold_ratio = 2 / (1 + np.sqrt(5))
            omega_test = self.omega_next - (self.omega_next - self.omega_estimated) * gold_ratio
            if self.omega_estimated < self.omega_next:
                ratio_test = 2 * np.log2(omega_test / self.omega_estimated)
                # print("A")
            else:
                ratio_test = -2 * np.log2(omega_test / self.omega_estimated)
                # print("B")
            if ratio_test < ratio:
                ratio = ratio_test
                # print(ratio)

        upr = self.omega_estimated * 2**(0.5 * ratio)
        lwr = 2 * self.omega_estimated - upr
        print(np.array([lwr, self.omega_estimated, upr])/(2*np.pi))

        damp = MorletDamping(self.irf, self.fs, self.k[0], self.n1, self.n2[0])
        damp.set_int_method(np.trapz)

        kitr = 0
        for i in self.k:
            lim = int(2*np.pi*i/(self.omega_estimated)*self.fs + 1)
            if lim > self.irf.size:
                # print(lim, self.omega_estimated, self.fs)
                print('Maximum iterations reached for: k = ', i)
                self.zeta_detected = self.zeta_detected[:, :kitr]
                self.omega_detected = self.omega_detected[:, :kitr]
                self.k = self.k[:kitr]
                break

            damp.k = i
            nitr = 0
            for n2 in self.n2:
                damp.n2 = n2

                omega_test = self.omega_estimated

                # Adjustment of search region in case of boundary cases when high k numbers for
                # some natural frequencies can generate mother wavelet function larger then signal.
                # -1 is added below to be on the safe side, but with short signals it may cause
                # problems.
                lwr_test = 2 * np.pi * i * self.fs / (self.irf.size - 1)
                if lwr < lwr_test or i > int((self.irf.size - 1) * lwr / (2*np.pi*self.fs)):
                    lwr = lwr_test
                    omega_test = 0.5 * (upr + lwr)
                    print(lwr, omega_test, upr)

                # fun_M = lambda x: -np.abs(damp.morlet_integrate(damp.n1, x)) /\
                #                    np.abs(damp.morlet_integrate(damp.n2, x))
                fun_M = lambda x: -np.abs(damp.morlet_integrate(damp.n2, x))

                try:
                    mnm = minimize_scalar(fun_M, bounds=(lwr, upr), method='bounded', \
                        options={'maxiter': 20, 'disp': 0})
                    # mnm = minimize(fun_M, x0=omega_test, method='Powell')
                except RuntimeWarning:
                    print("Minimize raised RuntimeWarning.")
                    # if verb:
                    #     print("Minimize raised RuntimeWarning.")

                try:
                    self.omega_detected[nitr, kitr] = mnm.x
                except UnboundLocalError:
                    self.omega_detected[nitr, kitr] = np.nan
                    print("Raised UnboundLocalError.")

                if self.omega_next is not None:
                    test = np.array([self.omega_detected[nitr, kitr], self.omega_next])
                    if n2*np.max(test)/(4*np.pi*i) >= np.abs(np.diff(test)):
                        if verb:
                            print('Frequency resolution is insufficient!')
                            print(np.abs(np.diff(test)))
                            print('k = ', i)
                            print('n2 = ', n2)
                        break
                if verb:
                    print("%d\t%d\t%.6f\t%.6f"
                          % (i, n2, omega_test, self.omega_detected[nitr, kitr]))

                nitr += 1
            kitr += 1

    def detect_damp(self, verb=False):
        """
        Method detects damping for given ranges of k and n2 parameters and checks if detected
        damping is feasible. If not then it is set as NaN.

        Args:
            fsearch - disable/enable searching of natural frequency
            verb - enable/disable messages
        """
        damp = MorletDamping(self.irf, self.fs, self.k[0], self.n1, self.n2[0])
        damp.set_int_method(np.trapz)

        kitr = 0
        for i in self.k:
            lim = int(2*np.pi*i/(self.omega_estimated)*self.fs + 1)
            if lim > self.irf.size:
                # print(lim, self.omega_estimated, self.fs)
                print('Maximum iterations reached for: k = ', i)
                self.zeta_detected = self.zeta_detected[:, :kitr]
                self.omega_detected = self.omega_detected[:, :kitr]
                self.k = self.k[:kitr]
                break
            damp.k = i
            nitr = 0
            for n2 in self.n2:
                damp.n2 = n2

                if np.isnan(self.omega_detected[nitr, kitr]):
                    self.zeta_detected[nitr, kitr] = np.NaN
                    if verb:
                        print("Damping not detected because frequency is not detected.")
                    break

                if self.n1 < 10:
                    # Exact method
                    # damp.set_root_finding(method="exact", x0=0.001)
                    damp.set_root_finding(method="exact")
                    dmp = damp.identify_damping(self.omega_detected[nitr, kitr], verb)
                else:
                    # Closed-form method
                    damp.set_root_finding(method="close")
                    dmp = damp.identify_damping(self.omega_detected[nitr, kitr], verb)

                if isinstance(dmp, float) and dmp > 0 and dmp != np.inf:
                    self.zeta_detected[nitr, kitr] = dmp
                    if self.n1**2/(8*np.pi*i) < dmp or n2**2/(8*np.pi*i) < dmp:
                        if verb:
                            print('zeta = ', dmp)
                            print('Basic condition is not met: zeta <= n^2/(8*pi*k)')
                            print('k = ', i, '\tIteration: ', kitr)
                        self.zeta_detected[nitr, kitr] = np.NaN
                else:
                    self.zeta_detected[nitr, kitr] = np.NaN

                if verb:
                    print("%d\t%d\t%.6f\t%.6f"
                          % (i, n2, self.omega_detected[nitr, kitr], \
                              self.zeta_detected[nitr, kitr]))
                nitr += 1
            kitr += 1

if __name__ == "__main__":
    fs1 = 64
    N1 = 16 * fs1
    T1 = N1 / fs1
    t1 = np.linspace(0, T1 - 1/fs1, N1)

    w1 = 2*np.pi
    zeta1 = 0.01
    sig1 = np.cos(w1 * np.sqrt(1 - zeta1**2) * t1) * np.exp(-zeta1 * w1 * t1)

    noise_std1 = np.std(sig1) * 10**(-.05 * 15) # add noise SnR = 15
    sig1 += np.random.normal(0, noise_std1, sig1.shape)

#    Exact
    identifier = ExtendedMW(fs1, sig1, (w1+0.3,), (7, 14), (8, 17))

#    Close form
    # identifier = ExtendedMW(fs1, sig1, w1+0.3, (10, 20), (8, 17))

    identifier.detect_frequency(False, True)
    identifier.detect_damp(True)
    identifier.estimate()
    identifier.plot()
