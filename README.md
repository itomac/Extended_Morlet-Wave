# Extended Morlet-Wave identification

Extended Morlet-Wave identification method

<!-- #region -->
## Basic usage
User is required to supply sampling frequncy in S/s and impulse response functions as a numpy array of shape `(number_of_samples, measure_points)`

* For SDOF systems, define estimated natural frequency as:\
`omega = (100, )` [rad/s]
* For MDOF systems, the first frequency in tuple is estimated one and the second one it the closest frequency to the estimated:\
`omega = (100, 150)` [rad/s]


Additionally to make instance of ExtendedMW class default values of tupples: `time_spread` and `num_cycls_range` can be changed. Tuple `time_spread` contains `n1` and `n2` time sperad parametes, tuple `num_cycls_range` sets the range of `k` parameter.
<!-- #endregion -->

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
