# Generate surface dataset
The surface dataset netcdf file needs to match the domain/mesh file. It can be generated using the tools provided by CTSM.  
Note: Below steps are based on experience at Feb 27, 2023  

## Load these modules
module load ncl

## Step-1: Prepare SCRIP netcdf file
A SCRIP file is needed to define the target region. Users should prepare this file by themselves. The mapping tool now requires SCRIP files as inputs, while CTSM5.2 would change to ESMF mesh files.  
For test purpose, the below codes create a test SCRIP file:  

```
# Define test folders and define CTSM paths
path_ctsm="/glade/u/home/$USER/CTSM_repos/CTSM"
path_test="/glade/scratch/$USER/test_create_surfdata"
mkdir -p $path_test
cd $path_test

# Crete an example SCRIP file. ncl module is needed
script=${path_ctsm}/tools/site_and_regional/mknoocnmap.pl
$script -centerpoint 40,110 -name test -nx 2 -ny 3
```

## Step-2: Create mapping files  
Many source datasets are needed to create the surface dataset. These datasets have different resolutions. Mapping files will be generated to link source datasets and the target domain.  

```
# Note: mapping requires very large memory and should be submitted as jobs
gridfile="${path_test}/SCRIPgrid_test_nomask_c230227.nc" # from step-1
script=${path_ctsm}/tools/mkmapdata/mkmapdata.sh
usr_gname="testgrid"
$script -f ${gridfile} -r $usr_gname -t regional
```

## Step-3: Create surface data files  
Based on files from Step-2, surface dataset files can be generated. 

To produce a regional surface dataset, the mksrf_gridtype namelist option has to be changed to "regional". There are two ways:  
(1) Directly edit mksurfdata.pl. Change mksrf_gridtype to "regional" in mksurfdata.pl.  
(2) Run ./mksurfdata.pl with the "-debug" option so it produces a namelist file. Then, change mksrf_gridtype to "regional" in the namelist file. Finally, use that namelist file to run mksurfdata_map with
something like ./mksurfdata_map < namelist_file.nl.  

After mksrf_gridtype has been set up, run the following codes on compute nodes, which use (1) to create a mksurfdata_regional.pl.     

```
script="${path_ctsm}/tools/mksurfdata_map/mksurfdata_regional.pl"
usr_gname="testgrid" # same with that in Step-2
usr_gdate="230227"
usr_mapdir=$path_test
year="2000"
scenario="hist"
${script} -r usrspec -usr_gname ${usr_gname} -usr_gdate ${usr_gdate} -usr_mapdir ${usr_mapdir} -y ${year} -ssp_rcp ${scenario}
```

# Reference
- Personal communication with Sean and Erik  
- CTSM documents