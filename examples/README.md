# Quickstarts

Install the package with the viz extra, then generate one case and reconstruct:

~~~bash
bash examples/synthetic_quickstart.sh /tmp/usct-example
~~~

The deterministic circular phantom uses c0=1500 m/s and straight-ray ToF. Read
/tmp/usct-example/runs/cgls/result.h5 or preview.png. No RNG is used.

For an existing pressure case and an installed protocol-v2 WUST runtime:

~~~bash
WUST_BACKEND=cpu WUST_MATLAB_EXECUTABLE=/path/to/matlab \
  bash examples/fwi_quickstart.sh /path/to/case.h5 \
  configs/algorithms/fwi_wust.yaml /path/to/WaveformInversionUST runs/fwi
~~~

The supplied FWI config must match the physical grid, bounds and PML.
[Usage](../docs/usage.md) explains data import and result inspection.
