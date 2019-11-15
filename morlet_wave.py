#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Morlet-Wave damping identification method
==================================================
"""
import numpy as np
#from scipy.optimize import brentq
from scipy.optimize import newton
from scipy.optimize import minimize
from scipy import special
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d

class EMWdiEMA():
    """
    EMA using Extended Morlet-Wave damping identification method
    ============================================================
    Literature:
        [1]: Tomac, I., Lozina, Ž., Sedlar, D., Extended Morlet-Wave damping identification method
             International journal of mechanical sciences, 117 (2017), 31-40
             doi:10.1016/j.ijmecsci.2017.01.013
        [2]: Slavič, J., Boltežar, M., Damping identification with the Morlet-wave, Mechanical
             Systems and Signal Processing, 2011, Volume 25, Issue 5, July 2011, Pages 1632-1645,
             doi: 10.1016/j.ymssp.2011.01.008
    """

    def __init__(self, time=None, irf=None, omega_estimated=None, time_spread=(7, 14), num_cycls_range=None, verb=False):
        """
        Constructor of eMWDI object which sets initial parameters for the method.

        Args:
            time             - time vector of the signal
            irf              - impulse response function of the mechanical system (SDOF)
            nat_freqs        - natural frequencies in radians per second
            time_spread      - tuple condaining n1 and n2 time sperad parametes
            num_cycls_range  - tuple seting range of k parametr that define number of wave function
                               cycles
            verb             - enabel/disable meaasges
        Self:
            base_time_spread      - n1 time spread parameter
            ntime_spread_divisos       - vector of n2 parameters
            num_cycls       - vector of all k parameters
            omest   - estimated natural frequency (omega estimated)
            time       - time vector
            fs      - sampling frequency
            irf     - impulse response function
            zeta    - 2d array of identified dampig ratios
            omega   - 2d array of identified natural frequencies
            zetae   - estimated damping ratio
            omegae  - estimated natural frequency
            kest    - k parameter for the estimated damping ratio

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

        self.base_time_spread = time_spread[0]
        self.time_spread_divisos = np.arange(time_spread[0]+spread_jump, time_spread[1]+1)
        self.num_cycls = np.arange(num_cycls_range[0], num_cycls_range[1]+1)
        self.omega_estimated = omega_estimated

        self.time = time
        self.irf = irf

        self.zeta_detected = np.zeros((self.time_spread_divisos.size, self.num_cycls.size))
        self.omega_detected = np.zeros((self.time_spread_divisos.size, self.num_cycls.size))

        self.zeta = 0
        self.omega = 0
        self.k = 0

    def plot(self, plt1=True, plt2=True, plt3=True):
        """
        Visualisation of damping identification. Three types of plots are available that are
        selected using the method's argumens.
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
            plt.plot(self.num_cycls, zt_me)
            plt.plot(self.num_cycls, zt_me + zt_st, '--', linewidth=1)
            plt.plot(self.num_cycls, zt_me - zt_st, 'C1--', linewidth=1)
            plt.plot([self.k, self.k], [np.min(zt_me), np.max(zt_me)], 'k-', linewidth=1)
            plt.xlabel(r'$k$')
            plt.ylabel(r'$\bar{\zeta}_{n_2}$')
            plt.title('Mean values of damping alogn n2 with std')

        if plt2:
            plt.figure(figsize=(8, 6), dpi=80)
            plt.plot(self.num_cycls, zt_st)
            plt.plot([self.k, self.k], [np.min(zt_st), np.max(zt_st)], 'k-', linewidth=1)
            plt.xlabel(r'$k$')
            plt.ylabel(r'$\sigma\left(\bar{\zeta}\right)$')
            plt.title('Std dev from mean damping value')

        if plt3:
            X, Y = np.meshgrid(self.num_cycls, self.time_spread_divisos)
            plt.figure(figsize=(8, 6), dpi=80)
            ax3 = plt.axes(projection='3d')
            ax3.plot_surface(X, Y, self.zeta_detected*100, cmap='viridis')
            ax3.set(xlabel=r'$k$', ylabel=r'$n_2$', zlabel=r'$\zeta\ (\%)$', title='Damping 3D map')
            ax3.xaxis.set_ticks(self.num_cycls)
            ax3.yaxis.set_ticks(self.time_spread_divisos)

    def estimate(self, verb=True):
        """
        Estimates damping from identified damping values usign the detect() method.

        Args:
            verb - enable/disable meaasges
        """
        self.zeta = np.nanmin(np.mean(self.zeta_detected, 0))
        i = np.nanargmin(np.mean(self.zeta_detected, 0))
        self.omega = np.mean(self.omega_detected, 0)[i]
        self.k = self.num_cycls[i]

        if verb:
            print("k: %d\tzeta: %.4f %%\tomega = %.2f Hz (%.3f s^-1)"
                  % (self.k, self.zeta*100, self.omega/(2*np.pi), self.omega))

    def detect(self, fsearch=True, verb=False):
        """
        Metohd detects damping for given ranges of k and n2 parameters and checks if detected
        damping is feasible. If not then it is set as NaN.

        Args:
            fsearch - disable/enable searching of natural frequency
            verb - enable/disable meaasges
        """
        samp_freq = (self.time[1] - self.time[0])**-1
        kitr = 0
        for i in self.num_cycls:
            lim = int(2*np.pi*i/(self.omega_estimated)*samp_freq + 1) # update for MDOF!
            if lim > self.time.size:
                print('Maximum iterations reached for: k = ', i)
                self.zeta_detected = self.zeta_detected[:, :kitr]
                self.omega_detected = self.omega[:, :kitr]
                self.num_cycls = self.num_cycls[:kitr]
                break

            nitr = 0
            for n2 in self.time_spread_divisos:
                if fsearch:
                    upr = self.omega_estimated + 1
                    lwr = self.omega_estimated - 1
                    # self.omega_detected[nitr, kitr], M = bisek(calc_ratio, lwr, upr, self.time, self.irf,
                    #                                   (self.base_time_spread, n2), i)
                    mnm = minimize(calc_ratio, x0=self.omega_estimated, args=(self.time, self.irf, \
                                    (self.base_time_spread, n2), i), bounds=[(lwr, upr)], \
                                    options={'gtol': 1e-2, 'disp': False})
                    self.omega_detected[nitr, kitr] = mnm.x[0]
                    M = -mnm.fun
                else:
                    M = -calc_ratio(self.omega_estimated, self.time, self.irf, \
                                    (self.base_time_spread, n2), i)

                if self.base_time_spread < 10:
                    # Exact method
                    # dmp = exact((self.base_time_spread, n2), i, M, verb)
                    dmp, r = newton(exact_fun, .001, args=((self.base_time_spread, n2), i, M), \
                                    maxiter=20, full_output=True, disp=False)
                    if not r.converged:
                        dmp = np.NaN
                        if verb:
                            print('Newton-Ralphson: maximum iterations limit reached!')
                else:
                    # Closed-form method
                    dmp = closed_form((self.base_time_spread, n2), i, M)

                if isinstance(dmp, float) and dmp > 0 and dmp != np.inf:
                    self.zeta_detected[nitr, kitr] = dmp
                    if self.base_time_spread**2/(8*np.pi*i) < dmp or n2**2/(8*np.pi*i) < dmp:
                        if verb:
                            print('zeta = ', dmp)
                            print('Basic condition is not met: zeta <= n^2/(8*pi*k)')
                            print('k = ', i, '\tIteration: ', kitr)
                        self.zeta_detected[nitr, kitr] = np.NaN
                else:
                    self.zeta_detected[nitr, kitr] = np.NaN

                if verb:
                    print("%d\t%d\t%.6f\t%.6f\t%.6f\t%.6f"
                          % (i, n2, self.omega_estimated, self.omega_detected[nitr, kitr], M, \
                              self.zeta_detected[nitr, kitr]))
                nitr += 1
            kitr += 1

def calc_ratio(omega, time, irf, tspread, k):
    """
    Function calculates ratio of the absolte values from the two morlet-wave coefficients
    calculated with different time spread parameters.

    Args:
        omega       - base angular frequency of the MW function
        time        - time np.array
        irf         - np.array which contains one IRF
        tspread     - tuple which contains time spread parametes n1 and n2
        k           - number of cycles of the Morlet Wave function

    Returns:
        M - calculated ratio
    """
    fs = (time[1] - time[0])**-1
    n = np.asarray(tspread)
    lim = int(2*np.pi*k / omega * fs + 1)

    # psi = ((2*pi)^(3/4)*sqrt(k/(n*w)))^-1*exp(-n^2*(k*pi-t*w).^2/(16*k^2*pi^2)+1i.*(k*pi-t*w))
    A = 0.25197943553838073034791409490358 # (2*pi)^-(3/4) - calculated due to the acceleration
    B = time*omega - k*np.pi
    psi = A * np.sqrt(n[np.newaxis].T * omega / k) \
        * np.exp(-(n[np.newaxis].T / (4*k*np.pi))**2 * B**2 - B*1j)
    I = np.abs(np.trapz(irf[0:lim] * psi[:, 0:lim], time[0:lim]))
    return -I[0] / I[1]

# def exact(n, k, M, verb=False):
#     """
#     Function estimates the damping ratio using the Morlet Wave Exact method.

#     Args:
#         n - array that contains n1 and n2 MW function time spread parameters
#         k - MV function number of cycles
#         M - ratio of the MW coefficinet calculated form the signal

#     Returns:
#         dmp - estimated damping ratio
#     """

#     dmp = .001
#     fun = exact_fun(dmp, n, k, M)
#     count = 0
#     while np.abs(fun) > 1e-6:
#         if count > 20:
#             dmp = np.NaN
#             if verb:
#                 print('Newton-Ralphson: maximum iterations limit reached!')
#         else:
#             grad = (exact_fun(dmp+1e-5, n, k, M) - fun)*1e5
#             dmp -= fun/grad
#             fun = exact_fun(dmp, n, k, M)
#             count += 1

#     # This method proven to be slightly slower then newton-ralphson
#     #d = brentq(exact_fun, 0, .1, (n, k, M), xtol=1e-5, maxiter=20, full_output=False)

#     return dmp

def exact_fun(d, n, k, M): # arg = (n, k, M)
    """
    Function calculates difference between ratio of the morlet wave coefficients calculated
    using the exact analytical expression and ones calculated form the signal, both for the
    given parameters set. Function is used by the optimisator for the estimation of the damping
    ratio by finding the minimal difference.

    Args:
        d - estimated damping coefficient
        n - array that contains n1 and n2 MW function time spread parameters
        k - MV function number of cycles
        M - ratio of the MW coefficinet calculated form the signal

    Returns:
        difference between analytical and caluclated for the signal
    """
    return np.exp((2*np.pi*k*d/(n[0]*n[1]))**2 * (n[1]**2-n[0]**2)) * np.sqrt(n[1]/n[0]) \
        * (special.erf(2*np.pi*k*d/n[0]+n[0]*.25) - special.erf(2*np.pi*k*d/n[0]-n[0]*.25)) \
        / (special.erf(2*np.pi*k*d/n[1]+n[1]*.25) - special.erf(2*np.pi*k*d/n[1]-n[1]*.25)) - M

def closed_form(n, k, M):
    """
    Function estimates the damping ratio using the Morlet Wave Closed form method.

    Args:
        n - array that contains n1 and n2 MW function time spread parameters
        k - MV function number of cycles
        M - ratio of the MW coefficinet calculated form the signal

    Returns:
        dmp - estimated damping ratio
    """

    return .5*n[0]*n[1] / (np.pi*k * np.sqrt(n[1]**2 - n[0]**2)) \
        * np.sqrt(np.log(M * np.sqrt(n[1]/n[0])))

# def bisek(fun, bot, upr, *arg):
#     """
#     Function searches for maximum value of 1D function within given range using Bisection method

#     Args:
#         fun  - function (x, *args) -> maximum is sought againts x
#         low  - bottom boundary
#         high - upper boundary
#         *arg - extra arguments that are passed to a fun()

#     Returns:
#         extrem, func(extrem)
#     """
#     a = np.zeros(3)
#     b = np.zeros(3)
#     eps = .01

#     a[0] = .5*(bot + upr)
#     a[1] = .5*(bot + a[0])
#     a[2] = .5*(upr + a[0])

#     args = list(arg)
#     args.insert(0, 0)
#     for i in range(0, 3):
#         args[0] = a[i]
#         b[i] = fun(*args)

#     while np.abs(bot - upr) > eps:
#         if b[0] > b[1] and b[2] > b[0]:
#             bot = a[0]
#         elif b[0] < b[1] and b[2] < b[0]:
#             upr = a[0]
#         else:
#             bot = a[1]
#             upr = a[2]

#         a[0] = .5*(bot + upr)
#         a[1] = .5*(bot + a[0])
#         a[2] = .5*(upr + a[0])

#         for i in range(0, 3):
#             args[0] = a[i]
#             b[i] = fun(*args)

#     args[0] = .5*(bot + upr)
#     return args[0], fun(*args)
