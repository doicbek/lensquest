# Install

pip install . (--user)

# Usage

```python
import lensquest
```


```python
questobject=lensquest.quest(maps, wcl, dcl, lmin=2, lmax=None, lminCMB=2, lmaxCMB=None)

questobject.grad(XY)
# returns a_lm^Phi XY

questobject.mask(XY)
# returns a_lm^M XY

questobject.noise(XY)
# returns a_lm^N XY

lensquest.lensing(maps,phi,cl,lminCMB=2,lmaxCMB=None)
# lenses T, QU or TQU in maps (list or nparray)
```


```python
lensquest.quest_norm(wcl, dcl, lmin=2, lmax=None, lminCMB=2, lmaxCMB=None, curl=False, rdcl=None, bias=False)
# returns dict of A_L (and N_L if bias=True) of TT or TT,TE,EE,TB,EB

lensquest.quest_norm_bh(spec, wcl, dcl, lmin=2, lmax=None, lminCMB=2, lmaxCMB=None)
# returns dict of A_L^XY for spec="TT","TE","EE","TB" or "EB"
```
