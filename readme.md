# Spatiotemporal Maps (STM) Computation and Reconstruction Software v1.0

## Overview

This MATLAB software provides examples of the reconstruction framework proposed in [1]. It performs reconstruction of 2D multichannel retrospectively undersampled dynamic MRI k-space data by computing and using spatiotemporal maps.

## Notes

The reconstruction in the provided examples uses a simple Tikhonov-regularized model-based approach using STMs. However, once STMs are computed, more advanced reconstruction approaches or regularizers could be used. 

## Contents

### Main Scripts

- **`STM_computation.m`** - Function for STM computation
- **`example_STM_recon_2D_multichannel.m`** - Example using 2D multichannel retrospectively undersampled data. STMs are computed and used for reconstruction.

## References

**[1]** R. A. Lobos, X. Wang, R. T. L. Fung, Y. He, D. Frey, D. Gupta, Z. Liu, J. A. Fessler, D. C. Noll.  
"Spatiotemporal Maps for Dynamic MRI Reconstruction," 2025, arXiv:2507.14429.  
https://arxiv.org/abs/2507.14429