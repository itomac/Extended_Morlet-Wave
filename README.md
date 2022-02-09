# Extended Morlet-Wave identification

Morlet-wave method for identification of damping is published in 1. and extension of the method for identification of damping is published in 2 and 3.

## Basic usage
User is required to supply sampling frequncy in S/s and impulse response functions as a numpy array of shape `(number_of_samples, measure_points)`

* For SDOF systems, define estimated natural frequency as:\
`omega = (100, None)` [rad/s]
* For MDOF systems, the first frequency in tuple is estimated one and the second one it the closest frequency to the estimated:\
`omega = (100, 150)` [rad/s]


Additionally to make instance of ExtendedMW class default values of tupples: `time_spread` and `num_cycls_range` can be changed. Tuple `time_spread` contains `n1` and `n2` time sperad parametes, tuple `num_cycls_range` sets the range of `k` parameter.


```python
sys = ExtendedMW(fs=None,
                 irf=None,
                 nat_freqs=(None, None),
                 time_spread=(7, 14),
                 num_cycls_range = (30, 300))
```

Detect natural frequencies:\
note: first argument of the method enables/disables natural frequency identification


```python
sys.detect_frequency(False, True)
```

Detect damping with verbose:


```python
sys.detect_damp(True)
```

If everything went well estimate damping ratio and natural frequerncy:


```python
sys.estimate()
```

Detect amplitude and phase vith verbose:


```python
sys.detect_amplitude(True)
```

Optionaly identification can be ploted using the following metod:


```python
sys.plot()
```

**Run simple example from jupyter notebook!**

## References
1. Slavič, J., Boltežar, M., Damping identification with the Morlet-Wave, Mechanical Systems and Signal Processing, 2011, DOI: [10.1016/j.ymssp.2011.01.008](https://doi.org/10.1016/j.ymssp.2011.01.008)
2. Tomac, I., Lozina, Ž., Sedlar, D., Extended Morlet-Wave damping identification method, International Journal of Mechanical Sciences, 2017, DOI: [10.1016/j.ijmecsci.2017.01.013](https://doi.org/10.1016/j.ijmecsci.2017.01.013)
3. Tomac, I., Slavič, J., Damping identification based on a high-speed camera, Mechanical Systems and Signal Processing, 2022, DOI: [10.1016/j.ymssp.2021.108485](https://doi.org/10.1016/j.ymssp.2021.108485)
