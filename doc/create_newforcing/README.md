# Create a new forcing dataset for CTSM
For regional/global studies, other forcing datasets could be needed.  

## Example of existing forcing datasets
NLDAS
```
ll /glade/p/cesmdata/cseg/inputdata/atm/datm7/atm_forcing.datm7.NLDAS2.0.125d.v1/

# outputs are:
ctsmforc.NLDAS2.0.125d.v1.ESMFmesh_120620.nc
ctsmforc.NLDAS2.cdf5.0.125d.v1.ESMFmesh_120620_c210330.nc
Precip
README_190425
Solar
TPQWL
```
where Precip, Solar, and TPQWL are folders containing forcing files. ESMFmesh files define grid locations and shapes.

## Step-1: Create forcing gridded files  
Example: 
```
ncdump -h /glade/p/cesmdata/cseg/inputdata/atm/datm7/atm_forcing.datm7.NLDAS2.0.125d.v1/TPQWL/ctsmforc.NLDAS2.0.125d.v1.TPQWL.2000-01.nc

# outputs are:
netcdf ctsmforc.NLDAS2.0.125d.v1.TPQWL.2000-01 {
dimensions:
        scalar = 1 ;
        lon = 464 ;
        lat = 224 ;
        time = 744 ;
variables:
        float time(time) ;
                time:long_name = "observation time" ;
                time:units = "days since 2000-01-01 00:00:00" ;
                time:calendar = "noleap" ;
        float LONGXY(lat, lon) ;
                LONGXY:long_name = "longitude" ;
                LONGXY:units = "degrees_east" ;
                LONGXY:mode = "time-invariant" ;
        float LATIXY(lat, lon) ;
                LATIXY:long_name = "latitude" ;
                LATIXY:units = "degrees_north" ;
                LATIXY:mode = "time-invariant" ;
        float EDGEE(scalar) ;
                EDGEE:long_name = "eastern edge in atmospheric data" ;
                EDGEE:units = "degrees_east" ;
                EDGEE:mode = "time-invariant" ;
        float EDGEW(scalar) ;
                EDGEW:long_name = "western edge in atmospheric data" ;
                EDGEW:units = "degrees_east" ;
                EDGEW:mode = "time-invariant" ;
        float EDGES(scalar) ;
                EDGES:long_name = "southern edge in atmospheric data" ;
                EDGES:units = "degrees_north" ;
                EDGES:mode = "time-invariant" ;
        float EDGEN(scalar) ;
                EDGEN:long_name = "northern edge in atmospheric data" ;
                EDGEN:units = "degrees_north" ;
                EDGEN:mode = "time-invariant" ;
        float PSRF(time, lat, lon) ;
                PSRF:long_name = "surface pressure at the lowest atm level" ;
                PSRF:units = "Pa" ;
                PSRF:mode = "time-dependent" ;
                PSRF:_FillValue = 1.e+36f ;
                PSRF:missing_value = 1.e+36f ;
        float TBOT(time, lat, lon) ;
                TBOT:long_name = "temperature at the lowest atm level" ;
                TBOT:units = "K" ;
                TBOT:mode = "time-dependent" ;
                TBOT:_FillValue = 1.e+36f ;
                TBOT:missing_value = 1.e+36f ;
        float WIND(time, lat, lon) ;
                WIND:long_name = "wind at the lowest atm level" ;
                WIND:units = "m/s" ;
                WIND:mode = "time-dependent" ;
                WIND:_FillValue = 1.e+36f ;
                WIND:missing_value = 1.e+36f ;
        float QBOT(time, lat, lon) ;
                QBOT:long_name = "specific humidity at the lowest atm level" ;
                QBOT:units = "kg/kg" ;
                QBOT:mode = "time-dependent" ;
                QBOT:_FillValue = 1.e+36f ;
                QBOT:missing_value = 1.e+36f ;
        float FLDS(time, lat, lon) ;
                FLDS:long_name = "incident longwave radiation" ;
                FLDS:units = "W/m**2" ;
                FLDS:mode = "time-dependent" ;
                FLDS:_FillValue = 1.e+36f ;
                FLDS:missing_value = 1.e+36f ;

// global attributes:
                :case_title = "NLDAS 1-Hourly Atmospheric Forcing: Temperature, Pressure, Winds, Humidity, and Longwave" ;
}
```

## Step-2: Create mesh files
Reference: https://gist.github.com/uturuncoglu/9c638f003e0bf9dc089c8298d5e24c0a by Ufuk Turunçoğlu  

```
# use gen_scrip.ncl to generate SCIRP format from gridded forcing inputs
module load ncl  
ncl gen_scrip.ncl  

# use ESMF_Scrip2Unstruct function to generate ESMF mesh from SCRIP file
qsub create_mesh.job
```

