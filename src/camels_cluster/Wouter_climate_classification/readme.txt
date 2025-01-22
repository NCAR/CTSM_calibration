Title: 		A Quantitative Hydrological Climate Classification Evaluated with Independent Streamflow Data
Authors: 	Knoben, W.J.M., Woods, R.A., Freer, J.E.
DOI:		10.1029/2018WR022913
Contact: 	w.j.m.knoben@bristol.ac.uk


Data statement
==============

This study uses climate data (CRU TS v3.23) and streamflow data (GRDC "Pristine basins"). 

CRU TS climate datasets are freely available from https://crudata.uea.ac.uk/cru/data/hrg/. This study uses version 3.23 (Harris et al., 2014), downloaded on 02-07-2016. Indices derived from this data set are provided as part of this data package. 

GRDC streamflow data are available on request from http://www.bafg.de/GRDC/. This study uses a sub set known as “Climate Sensitive Stations Dataset (Pristine River Basins)” (The Global Runoff Data Centre, 2017) downloaded on 16-05-2017.


License
=======

Included in this data package are climate indices derived from CRU TS v3.23 data, which is made available here under the Open Database License (ODbL).

Data source: https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_3.23/
License: https://opendatacommons.org/licenses/odbl/1.0/


Data set contents
=================

In addition to this readme file, this data set contains the following items (explanation of files given below):

- ClimateClassification_mainMap_geoReferenced.tif
- ClimateClassification_mainMap_legendPlot_threeIndexMaps.png
- loadAndPlotClimateClassificationData.m
- HydrologicClimateClassification.nc


System/Software requirements
============================

Matlab is required to open the .m file. This file is not required to open or use the other files, but is provided for the convenience of Matlab users.


File descriptions
=================

IMAGE
-----
The file 'ClimateClassification_mainMap_legendPlot_threeIndexMaps.png' is a direct copy of figure 2 in the paper.

IMAGE
-----
The file 'ClimateClassification_mainMap_geoReferenced.tif' provides the main classification map in GeoTIFF format, referenced on a latitude/longitude grid.

M-SCRIPT
--------
Matlab script that extracts data from the netcdf file and provides some example plots. 

DATA
----
The file 'HydrologicClimateClassification.nc' contains indices derived from CRU TS v3.23 climate data. File contents are split between gridded (0.5x0.5 degree lat/lon resolution) and array (67214 values - 1 per land cell) formats.

NetCDF contents:

Variable name 			- description
-----------------------------------------------

Gridded values: index values given for land cells only, ocean cells as NaN

grid_aridity_Im 		- gridded values for the aridity index Im 
grid_seasonalityOfAridity_Imr	- gridded values for the aridity seasonality index Imr
grid_annualSnowFraction_fs	- gridded values for the precipitation as snow index fs
grid_latitude			- gridded latitude values (all cells, no distinction land/ocean)
grid_longitude			- gridded longitude values (all cells, no distinction land/ocean)


Array values: index values given for land cells only, use array_latitude and array_longitude for location

array_aridity_Im		- values for the aridity index Im
array_seasonalityOfAridity_Imr	- values for the aridity seasonality index Imr
array_annualSnowFraction_fs	- values for the precipitation as snow index fs
array_latitude			- latitude of all land cells
array_longitude			- longitude of all land cells
array_rgbColour			- index values transformed into RGB colour scheme


References
==========

Harris, I., Jones, P. D., Osborn, T. J., & Lister, D. H. (2014). Updated high-resolution grids of monthly climatic observations - the CRU TS3.10 Dataset. International Journal of Climatology, 34(3), 623–642. https://doi.org/10.1002/joc.3711

The Global Runoff Data Centre. (2017). GRDC pristine catchments data set, 1984-2014. 56068 Koblenz, Germany