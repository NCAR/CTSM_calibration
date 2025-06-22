# CTSM_calibration
**CTSM_calibration** is an open-source Python package designed to calibrate and regionalize model parameters for the Community Terrestrial Systems Model (CTSM) using an AI-based model response-surface emulator. This package enables both single-site and large-sample optimization of CTSM's performance, and introduces a new method for the latter called 'large sample emulation (LSE)'.  The emulator is trained through sequentially optimization, yielding an efficient tool for searching the model parameter space cheaply to identify performative parameter sets.  The emulator and optimization process can also be used to regionalize the model implementation to unseen locations, including national to global simulation domains.  

The current implementation and work to date with the LSE focus on _hydrological_ performance, using a large sample of diverse watersheds, yet the methodology and supporting software are general to efficient complex model calibration for a wide range of desired outcomes (such as for land-atmosphere fluxes of heat and moisture, or other model prognostic and diagnostic variables (e.g, LAI, GPP). Even more generally, **CTSM_calibration** provides tools for flexible, extensible, and scalable parameter estimation in complex land models. It has since been implemented for other land/hydrology models such as SUMMA (Farahani et al., 2025) and HBV (Tang et al, 2025).

The application for which this package was developed is detailed in _"On using AI-based large-sample emulators for land/hydrology model calibration and regionalization"_ (Tang et al., 2025). The initial implemenation was designed to run on NCAR Casper High Performance Computing (HPC) resources.

## Features
- **Emulator-based calibration**: Currently use Gaussian Process Regression (GPR) and Random Forest (RF) emulators with adaptive surrogate optimization via a genetic algorithm, but these choices are extensible to other emulator forms and optimization methods.
- **Large-sample emulator (LSE)**: Jointly optimize CTSM parameters across 627+ CONUS basins using static geo-attributes for model generalization.
- **Single-site emulator (SSE)**: Basin-by-basin multi-objective optimization via MO-ASMO (Gong et al., 2016) methodology.
- **Dynamically dimensioned search (DDS)**: An alternative (benchmarking) calibration algorithm developed by Tolson and Shoemaker (2007). Relevant workflow and codes are adapted from Tang et al (2023), leveraging prior model implementations developed at NCAR. 
- **Regionalization-ready**: Predict parameters in ungauged basins using attribute-based transfer from trained emulators when LSE is adopted.
- **Tested at scale**: Supports tens of thousands of CTSM runs; compatible with NCAR HPC and other parallelized environments.
- **Evaluation tools**: Integrated performance metrics (e.g., KGE', MAE, MMAE), cross-validation routines, and visualization support.

## Acknowledgements  
This work has been supported by research grants to NCAR from the United States Army Corps of Engineers (the ‘Robust Hydrology’ projects), and from the NOAA Climate Observations and Modeling program. These sponsored projects (led by PI A. Wood) seek to advance scientific and technical capabilities for large-domain land/hydrology model calibration so as to create new modeling resources for national-scale water security analyses and risk projection.  We acknowledge high‐performance computing support provided by NCAR’s Computational and Information Systems Laboratory, sponsored by the National Science Foundation. 

## Contacts
- ** Guoqiang Tang, Wuhan University, guoqiang.tang@whu.edu.cn 
- ** Andy Wood, NSF NCAR, andywood@ucar.edu

## Major Citation  
Tang, G., Wood, A., & Swenson, S. (2025). On using AI-based large-sample emulators for land/hydrology model calibration and regionalization. Water Resources Research (accepted)

## Related references 
- **Tang, G., Clark, M. P., Knoben, W. J., Liu, H., Gharari, S., Arnal, L., ... & Papalexiou, S. M. (2023). The impact of meteorological forcing uncertainty on hydrological modeling: A global analysis of cryosphere basins. Water Resources Research, 59(6), e2022WR033767.  
- **Gong, W., Duan, Q., Li, J., Wang, C., Di, Z., Ye, A., ... & Dai, Y. (2016). Multiobjective adaptive surrogate modeling‐based optimization for parameter estimation of large, complex geophysical models. Water Resources Research, 52(3), 1984-2008.  
- **Tolson, B. A., & Shoemaker, C. A. (2007). Dynamically dimensioned search algorithm for computationally efficient watershed model calibration. Water Resources Research, 43(1).  
- **Tang, G, AW Wood, Y. Song and C. Shen, 2025. Global calibration and locally-informed regionalization of hydrological model parameters using AI-based large-sample emulators.  Water Resources Research (in review).
- **Farahani, MA, G Tang, N Mizukami, and AW Wood, 2025. Calibrating large-domain land/hydrology process models in the age of AI: the SUMMA CAMELS experiments. HESS (in review).  https://egusphere.copernicus.org/preprints/2025/egusphere-2025-38/ 
- **Newman, AJ, MP Clark, K Sampson, AW Wood, LE Hay, A Bock, R Viger, D Blodgett, L Brekke, JR Arnold, T Hopson, and Q Duan, 2015, Development of a large-sample watershed-scale hydrometeorological data set for the contiguous USA: data set characteristics and assessment of regional variability in hydrologic model performance, Hydrol. Earth Syst. Sci., 19, 209-223, www.hydrol-earth-syst-sci.net/19/209/2015/doi:10.5194/hess-19-209-2015
- **Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The CAMELS data set: catchment attributes and meteorology for large-sample studies, Hydrol. Earth Syst. Sci., 21, 5293–5313, doi:10.5194/hess-21-5293-2017, 2017.
