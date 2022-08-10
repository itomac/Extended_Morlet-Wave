.. role:: raw-html(raw)
    :format: html

.. role:: py(code)
   :language: python

Extended Morlet-Wave identification 
------------------------------------------
This is the Python implementation of the extended Morlet-Wave damping identification method, see [1]_ and [2]_ for details.

This package is based on the `mwdi`_ code developed by TOMAC Ivan and SLAVIČ Janko, see [3]_ for details. It was upgraded within the 
MSCA IF project `NOSTRADAMUS`_ during development of [2]_.

Basic usage
-----------
User is required to supply sampling frequncy in S/s and impulse response functions as a numpy array of shape :py:`(number_of_samples, measure_points)`

* For SDOF systems, define estimated natural frequency as :raw-html:`<br />` :py:`omega = (100, None)` [rad/s]

* For MDOF systems, the first frequency in tuple is estimated one and the second one it the closest frequency to the estimated: :raw-html:`<br />` :py:`omega = (100, 150)` [rad/s]

Additionally to make instance of ExtendedMW class default values of tupples: :py:`time_spread` and :py:`num_cycls_range` can be changed. Tuple :py:`time_spread` contains :py:`n1` and :py:`n2` time sperad parametes, tuple :py:`num_cycls_range` sets the range of :py:`k` parameter.

.. code-block:: python

   identifier = ExtendedMW(fs=None,
                           irf=None,
                           nat_freqs=(None, None),
                           time_spread=(7, 14),
                           k_range = (10, 400))

Detect natural frequencies: :raw-html:`<br />`
:py:`identifier.detect_frequency()`

Detect damping: :raw-html:`<br />`
:py:`identifier.detect_damp()`

Identify damping ratio and natural frequerncy::raw-html:`<br />`
:py:`identifier.estimate()`

Optionaly identification can be ploted using the following metod::raw-html:`<br />`
:py:`identifier.plot()`


Simple example
---------------

A simple example how to identify damping using MWDI method:

.. code-block:: python

   import morlet_wave as emw
   import numpy as np

   # set time domain
   fs = 50 # sampling frequency [Hz]
   N = int(50*fs) # number of data points of time signal
   time = np.arange(N) / fs # time vector

   # generate a free response of a SDOF damped mechanical system
   w_n = 2*np.pi * 1 # undamped natural frequency
   d = 0.01 # damping ratio
   x = 1 # amplitude
   phi = 0.3 # phase
   response = x * np.exp(-d * w_n * time) * np.cos(w_n * np.sqrt(1 - d**2) * time - phi)

   # set MWDI object identifier
   identifier = emw.ExtendedMW(fs=fs, \
                               free_response=response, \
                               nat_freqs=(w_n, None))

   # identify natural frequency and damping ratio:
   identifier.detect_frequency()
   identifier.detect_damp()
   identifier.estimate()

   # plot optimization results
   identifier.plot()

References
----------
.. [1] I\. Tomac, Ž. Lozina, D. Sedlar, Extended Morlet-Wave damping identification method, International Journal of Mechanical Sciences, 2017, doi: `10.1016/j.ijmecsci.2017.01.013`_.
.. [2] I\. Tomac, J. Slavič, Damping identification based on a high-speed camera. Mechanical Systems and Signal Processing, 166 (2022) 108485–108497, doi: `10.1016/j.ymssp.2021.108485`_.
.. [3] J\. Slavič, M. Boltežar, Damping identification with the Morlet-wave, Mechanical Systems and Signal Processing, 25 (2011) 1632–1645, doi: `10.1016/j.ymssp.2011.01.008`_.

.. image:: https://zenodo.org/badge/220045505.svg
   :target: https://zenodo.org/badge/latestdoi/220045505

.. _NOSTRADAMUS: http://ladisk.si/?what=incfl&flnm=nostradamus.php
.. _mwdi: https://github.com/ladisk/mwdi
.. _10.1016/j.ymssp.2011.01.008: https://doi.org/10.1016/j.ymssp.2011.01.008
.. _10.1016/j.ijmecsci.2017.01.013: https://doi.org/10.1016/j.ijmecsci.2017.01.013
.. _10.1016/j.ymssp.2021.108485: https://doi.org/10.1016/j.ymssp.2021.108485
