# UliAcceleration
Fast accelerated math routines for Python

## How to install

To install `UliAcceleration`, simply run
```
sudo pip3 install UliAcceleration
```

In order to install the bleeding edge version:
```
sudo pip3 install git+https://github.com/ulikoehler/UliAcceleration.git
```

## About UliAcceleration

My main library [UliEngineering](https://github.com/ulikoehler/UliEngineering) is focused on ease of use and flexibility, but sometimes this yields a huge performance tradeoff.

*UliAcceleration* aims to provide less flexible but faster routines especially for the following areas:

* Signal processing
* Electronics
* Simulation
* Data science

Algorithms will only be published here if they offer significant performance gains compared.

I use [Numba](https://numba.pydata.org) for accelerating routines ; this provides the added advantage of platform-specific JIT optimization. However, it also means the algorithms are not optimized for load time and might take several hundred milliseconds (or more) to be compiled on first use, or on use with another data type.
