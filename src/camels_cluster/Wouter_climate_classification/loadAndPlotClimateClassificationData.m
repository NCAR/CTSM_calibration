% Climate classification data loading and plotting.
%
% This script loads data from the file 'HydrologicClimateClassification.nc'
% (NetCDF format) and provides a few example plots. Climate index values
% are calculated using the CRU TS v3.23 data set (Harris et al, 2014). The 
% values provided in this file cover the years 1984-2014. 
%
% File contents
% 1. Data import from netcdf file
% 2. Example plots
%
% References
% Harris, I., Jones, P. D., Osborn, T. J., & Lister, D. H. (2014). Updated 
%   high-resolution grids of monthly climatic observations - the CRU TS3.10
%   Dataset. International Journal of Climatology, 34(3), 623–642. 
%   https://doi.org/10.1002/joc.3711
%
% 
%
% Date:     23/06/2018
% Author:   W. Knoben
% Contact:  w.j.m.knoben@bristol.ac.uk

%% 1. Data import from netcdf file
% The netcdf file contains the values of the three climate indices (Im, 
% Imr and fps) in gridded (360x720) and array (67214x1) format. Values are
% only given for land cells, ocean cells have NaN values.

% Filename
filename = 'HydrologicClimateClassification.nc';

% Variables names
varNames = {...
            'grid_aridity_Im';...
            'grid_seasonalityOfAridity_Imr';...
            'grid_annualSnowFraction_fs';...
            'grid_latitude';...
            'grid_longitude';...
            'array_aridity_Im';...
            'array_seasonalityOfAridity_Imr';...
            'array_annualSnowFraction_fs';...
            'array_latitude';...
            'array_longitude';...
            'array_rgbColour'};

% 1a. Display contents of the NetCDF file.
ncdisp(filename);

% 1b. Extract data into a structure.
for ii = 1:size(varNames,1) 
    ClimateClassification.(varNames{ii}) = ncread(filename,varNames{ii});
end

%% 2. Example plots
% The data is provided in both gridded and array formats. This section
% gives a few example plots for both types of data.

% 2a. Individual maps of climate indices
    % aridity Im
    figure('color','w')
    mesh(ClimateClassification.grid_aridity_Im,...
         'FaceColor','interp')
    title('Aridity index I_m')
    caxis([-1,1])
    colorbar
    view(0,90)

    % aridity seasonality Imr
    figure('color','w')
    mesh(ClimateClassification.grid_seasonalityOfAridity_Imr,...
         'FaceColor','interp')
    title('Aridity seasonality index I_{m,r}')
    caxis([0,2])
    colorbar
    view(0,90)

    % fraction precipitation as snow fs
    figure('color','w')
    mesh(ClimateClassification.grid_annualSnowFraction_fs,...
         'FaceColor','interp')
    title('Fraction precipitation as snow index f_s')
    caxis([0,1])
    colorbar
    view(0,90)

% 2b. Combined climate map
figure('color','w')
scatter(ClimateClassification.array_longitude,...
        ClimateClassification.array_latitude,...
        5,...
        ClimateClassification.array_rgbColour,...
        'filled')
title('Climatic gradients from climate indices')     

% 2c. RGB colour legend (try rotating this plot, it's quite interesting)
figure('color','w')
scatter3(ClimateClassification.array_aridity_Im,...
         ClimateClassification.array_seasonalityOfAridity_Imr,...
         ClimateClassification.array_annualSnowFraction_fs,...
         5,...
         ClimateClassification.array_rgbColour,...
         'filled')
title('Legend: climate indices')
xlabel('Aridity index I_m')
ylabel('Aridity seasonality index I_{m,r}')
zlabel('Fraction precipitation as snow index f_s')

% 2d. GeoTIff plot of the main map. Note that climate indices are
% transformed to create the colour scheme as follows:
%
% R = 1 - (aridity_Im + 1)./2       [-1,1] to [1,0] [most arid, least arid]
% G = (aridity_seasonality_Imr)./2  [ 0,2] to [0,1] [least seasonal, most seaonal]
% B = precipitation_as_snow_fs      [ 0,1] to [0,1] [least snow, most snow]
geofile = 'ClimateClassification_mainMap_geoReferenced.tif';

figure('color','w')
geoshow(geofile);
title('Climatic gradients from climate indices') 
xlim([-180,180]);
ylim([-90,90]);











