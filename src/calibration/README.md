# CTSM calibration
The calibration settings are in the .toml configuration file. Currently, it only accepts mesh domain file. In the future, domain settings can be added.  

# Workflow
1. Prepare configuration files. This step depends on users and datasets. CAMELS is used as an example here. Files are automatically generated. Configurations are critical.
2. Generate case, including create new case, case set up, case setting change, case build, etc.
3. Generate Ostrich settings for calibration. The script can deal with three types of parameter files: parameter nc file, surfdata nc file, and lnd_in namelist. The script can deal with two parameter perturbation methods, including multiplicative and additive.  
4. Create subset forcing. Basin scale simulation using continental/global forcing can waster a lot of time.  
5. Create spin up status. Create restart files for the target period. 

# How to use the package
1. Define parameters
2. Prepare configuration files
3. Run main.py

# Issues
Each basin will have a subset of forcing files, resulting in a large number of files. If this is not recommended, forcing file merging can be added.  
Some inelegant functions are not added to the workflow. They are named as outworkflow_XXX.py, which can be run after the workflow is complete. In the future, they can be wrapped up if there are too many outworkflow functions.   