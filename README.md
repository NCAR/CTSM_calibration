# CTSM_calibration

**CTSM_calibration** is an open-source Python package designed to calibrate and regionalize hydrologic parameters for the Community Terrestrial Systems Model (CTSM) using AI-based emulators. This package enables both single-site and large-sample optimization of CTSM's hydrological performance across diverse catchments and supports efficient model calibration at scale, tested over the CAMELS (Catchment Attributes for Large-Sample Studies) dataset over U.S.  

Developed as part of the study _"On using AI-based large-sample emulators for land/hydrology model calibration and regionalization"_ (Tang et al., 2025), `CTSM_calibration` provides tools for flexible, extensible, and scalable parameter estimation in complex land models.

## Features

- **Emulator-based calibration**: Use Gaussian Process Regression (GPR) and Random Forest (RF) emulators with adaptive surrogate optimization.
- **Large-sample emulator (LSE)**: Jointly optimize CTSM parameters across 627+ CONUS basins using geo-attributes for model generalization.
- **Single-site emulator (SSE)**: Basin-by-basin multi-objective optimization via MO-ASMO (Gong et al., 2016) methodology.
- **Dynamically dimensioned search (DDS)**: A caliration algorithm developed by Tolson and Shoemaker (2007). Relevant workflow and codes are adapted from Tang et al (2023).
- **Regionalization-ready**: Predict parameters in ungauged basins using attribute-based transfer from trained emulators when LSE is adopted.
- **Tested at scale**: Supports tens of thousands of CTSM runs; compatible with NCAR HPC and other parallelized environments.
- **Evaluation tools**: Integrated performance metrics (KGE', MAE, MMAE), cross-validation routines, and visualization support.

## Acknowledgements  
This study is supported by research grants to NCAR from the United States Army Corps of Engineers Climate Preparedness and Resilience Program (the ‘Robust Hydrology’ projects), and by the NASA Subseasonal-to-Seasonal Hydrometeorological Prediction Program (Award #80NSSC23K0502). We acknowledge high‐performance computing support provided by NCAR’s Computational and Information Systems Laboratory, sponsored by the National Science Foundation.  

## Citation  
Tang, G., Wood, A., & Swenson, S. (2025). On using AI-based large-sample emulators for land/hydrology model calibration and regionalization. Water Resources Research

## Relevant reference  
Tang, G., Clark, M. P., Knoben, W. J., Liu, H., Gharari, S., Arnal, L., ... & Papalexiou, S. M. (2023). The impact of meteorological forcing uncertainty on hydrological modeling: A global analysis of cryosphere basins. Water Resources Research, 59(6), e2022WR033767.  
Gong, W., Duan, Q., Li, J., Wang, C., Di, Z., Ye, A., ... & Dai, Y. (2016). Multiobjective adaptive surrogate modeling‐based optimization for parameter estimation of large, complex geophysical models. Water Resources Research, 52(3), 1984-2008.  
Tolson, B. A., & Shoemaker, C. A. (2007). Dynamically dimensioned search algorithm for computationally efficient watershed model calibration. Water Resources Research, 43(1).  
