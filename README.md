# lensQUEST (public beta v0.1)
An implementation quadratic estimator for lensing extraction of full-sky CMB data in python, following Okamoto & Hu's paper [*CMB Lensing Reconstruction on the Full Sky*](https://arxiv.org/abs/astro-ph/0301031).

## Installation

### Prerequisites
The module requires an installation of [HEALPix C++](http://healpix.sourceforge.net/). Furthermore, the python modules `numpy`, `cython` and `healpy` should be installed.

### Compilation
The module can be easily installed with pip by executing
```
pip install . [--user]
```
in your local `lensquest` directory.

## Quick Start

![lensQUEST usage](https://github.com/doicbek/lensquest/blob/master/lensquest/lensquest.png)


- `maps`: CMB maps (T or list of T,Q,U) or corresponding harmonic coefficients (T or list of T,E,B) in healpy format
- `wcl`: Power spectra used in the weights of the quadratic estimator (array of TT or TT,EE,BB,TE power spectra)
- `dcl`: Power spectra used in the Wiener-filter of the input fields (array of TT or TT,EE,BB,TE power spectra)


```python
import lensquest

questobject=lensquest.quest(maps, wcl, dcl, lmin=2, lmax=None, lminCMB=2, lmaxCMB=None)

questobject.grad(XY)
# returns a_lm^Phi XY, where XY=TT,TE,EE,TB,EB or BB

lensquest.quest_norm(wcl, dcl, lmin=2, lmax=None, lminCMB=2, lmaxCMB=None, bias=False)
# returns dictionary of A_L (and N_L if bias=True) of TT or TT,TE,EE,TB,EB
```

## Contact
Dominic Beck: dbeck [at] apc.in2p3.fr

## License [![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
- [valandil/wignerSymbols](https://github.com/valandil/wignerSymbols)
- [Libsharp/libsharp](https://github.com/Libsharp/libsharp)
