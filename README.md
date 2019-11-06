---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# EMWdi

Extended Morlet-Wave damping identification method


## Basic usage
Curent version only works for the SODF system. It can be applied to MDOF too, but checking if the parameters are ok for the mode separation is not implemented jet. 

To make instance of EMWDI class, time and impulse response fucntion arrays must me supplied, estimated natural angular frequency, tuple condaining n1 and n2 time sperad parametes, tuple seting range of k parametr:

```python
sys = EMWdiEMA(
    time,
    irf,
    omega,
    tsprd=(7, 14)
    ncycl
    )
```

Detect damping with verbose

```python
sys.detect(True, True)
```

If everything went well estimate damping ratio and natural frequerncy:

```python
sys.estimate()
```

Optionaly identification can be ploted using the following metod:

```python
sys.plot()
```
