One mesh file contains many basins in one file. To calibrate the model, I disaggregate this file into one for one basin.

Here I use xarray to do subsetting. Before save to netcdf, use netCDF4 package to check the version of the netcdfile

