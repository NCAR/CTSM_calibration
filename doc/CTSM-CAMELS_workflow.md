# How to build the CTSM-CAMELS workflow from scratch

1. Prepare basin files  
This includes, (1) generating an improved CAMELS shapefile dataset, and (2) generating the mesh and surface data files needed by CTSM  
It's better to convert the mesh back to shapefile to inspect whether the mesh file is reasonable（a script in the ./tools folder)   

2. Select a suitable CTSM version  
Hillslope hydrology version from Sean  

3. Prepare CAMELS data  
3.1 Create basin information csv file  
3.2 Disaggregate basin mesh and surface data files since they are all contained in one file  
3.3 Create configration files (Ostrich, MO-ASMO) for each basin  

4. Generate ERA5-Land + EM-Earth forcing  
This is gridded forcing for CONUS. The codes are in "CTSM_dataprepare" repo. CAMELS will extract gridded forcing from this dataset  

5. Run the workflow  
5.1 Create the case "Build,MOASMO,SubForc,NameList,SpinUp" # run complete MO-ASMO calibration  
This step will also archive the spin up outputs using default parameters. NameList should be run before SpinUp because it changes model settings  
I plan to output more restart files for each year, but unless the last restart file, all other files are removed by the model  
I submit basin-0 of MO-ASMO. After CTSM is compiled after 10-20 mintues, submit basin-1 to X which will clone basin-0.  

Numerical problems:  
(1)
`export MPI_DSM_DISTRIBUTE=0`
necessary for the regular queue to run CTSM model in parallel. The share queue already has this setting.  
unfortunately, this only works for node=1. for multiple nodes, CTSM create_case or clone_case do not run at all  
and even for one node, when I try to run CTSM at the same time, many cores jut failed and only some of tasks are running  

(2) Forcing subsetting:  
Cheyenne memory may not be enough to run 36 jobs on 36-CPU node. require the large memory node (i.e., mem=109GB)  
For CTSM, the lat/lon dims have to be >1. Otherwise, the job would just fail ... This is a problem for datm mapalgo=bilinear  

(3) Forcing NaN values:  
Sometimes some grids and some time steps have NaN values from ERA5-Land or EM-Earth. Need to run a post-processing function to fill those NaN values. ./src/tools/forcing_modification/findfill_nan_inforcing.py which runs in parallel so is fast.  
Thus, when creating cases, first generating subset forcing, then doing filling, and finally doing spin up  

(4) Other unknown reasons  
Solution: perturb forcing a little bit for a specific period


5.2 Generate MOASMO settings for sensitivity analysis  
To select parameters, I further perform a sensitivity analysis based on PPE results.  
This uses MO-ASMO functions and files (e.g, subset forcing and spin up files from the previous step)  


5.2 Create the case "Build,MOASMO,NameList" for Ostrich.  
Post-processing is needed to use the forcing and restart files from MO-ASMO  

Archive problem:  
For Ostrich, the archive folders are Run-1, Run-2, Run-3, ... OstModel0.txt also have runs starting from 1. For example, for Run-51, the OstModel0.txt has the last line corresponding to 51. But the true outputs are in Run-50 folders. Need to be check. This means the first model run (i.e., default parameters are not saved)  
For PreserveBestModel folder which saves the best results, the OstModel0.txt and *.nc files are matched, which means the last line of OstModel0.txt shows the best model results, and the *.nc files are just in that folder.  











