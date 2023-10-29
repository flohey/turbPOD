# turbPOD
Python code for performing the snapshot Proper Orthogonal Decomposition (POD) fluid flow data. The script computes the spatial modes and the temporal coefficients of the flow data and stores them on the disc.

## Examples
- Two-dimensinal Rayleigh-Bénard convection at ${\rm Ra}= 10^6$ and ${\rm Pr}= 10$ in a rectangular box with aspect ratio $\Gamma = L_X/H = 4$. The boundary conditions were free-slip and constant temperature.

## Manual
For a tutorial, see the prepared IPython Notebook `tutorial.ipynb` under /examples/two_dimensional_rbc/. Moreover, a post-processing script for the computed POD in `read_pod.ipynb` (same directory). For this, the POD has to be computed via the `run_config.py` script in a previous step. 
    

## About POD 
The POD is a common Reduced Order Modeling technique in, e.g., the field of Fluid Dynamics. It decomposes the fluid flow into a set of orthogonal spatially dependent modes and their time-dependent coefficients. By restricting oneself to a subset of these modes and coefficients, the amount of input data can drastically be reduced. In this way, information on the initial data is lost. In the POD case, this information loss is optimal in a mean square sense. Note that the POD is also known as Principle Component Analysis (PCA).

Further information on the POD can be found here:
- Brunton, S. L., & Kutz, J. N. (2019). Reduced Order Models (ROMs). In Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control (pp. 375–402). chapter, Cambridge: Cambridge University Press.
  https://www.doi.org/10.1017/9781108380690.012
- Weiss, J. (2019). A Tutorial on the Proper Orthogonal Decomposition. Technische Universität Berlin. https://doi.org/10.14279/DEPOSITONCE-8512 


## Requirements
Stable for
- `python` >=  3.6.0
- `matplotlib` >= 3.6.2
- `torch`  >= 1.10.0
- `numpy`  >= 1.20.1
- `h5py`   >= 2.10.0 
- `pyyaml` >= 6.0
