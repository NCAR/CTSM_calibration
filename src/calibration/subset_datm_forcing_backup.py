# Create a regional subset for the study area can accelerate modeling
import shutil
import xarray as xr
import numpy as np
import xmltodict
import os, time, sys, pathlib

def subset_forcing(infile, lat_range, lon_range):
    # gridded forcing
    ds_data = xr.open_dataset(infile)
    lonmin = np.min(ds_data.LONGXY.values, axis=0)
    lonmax = np.max(ds_data.LONGXY.values, axis=0)
    latmin = np.min(ds_data.LATIXY.values, axis=1)
    latmax = np.max(ds_data.LATIXY.values, axis=1)
    indexlat = np.where((latmin >= lat_range[0]) & (latmax <= lat_range[1]))[0]
    indexlon = np.where((lonmin >= lon_range[0]) & (lonmax <= lon_range[1]))[0]
    if len(indexlon) == 0:
        ind1 = np.where((lonmin >= lon_range[0]))[0][0]
        indexlon = [ind1-1, ind1]
    if len(indexlat) == 0:
        ind1 = np.where((latmin >= lat_range[0]))[0][0]
        indexlat = [ind1-1, ind1]
    ds_data = ds_data.load()
    ds_data_out = ds_data.sel(lat=indexlat, lon=indexlon)
    return ds_data_out

def subset_meshfile(meshfile, example_subsetfile):
    # mesh file, which is subset based on an example_subsetfile
    ds_mesh = xr.open_dataset(meshfile)
    mesh_centerCoords = ds_mesh.centerCoords.values
    ds_subset = xr.open_dataset(example_subsetfile)
    LONGXY = ds_subset['LONGXY'].values
    LATIXY = ds_subset['LATIXY'].values
    index_elementCount = []
    for i in range(LATIXY.shape[0]):
        for j in range(LATIXY.shape[1]):
            latij = LATIXY[i, j]
            lonij = LONGXY[i, j]
            indexij = np.where((mesh_centerCoords[:, 0] == lonij) & (mesh_centerCoords[:, 1] == latij))[0][0]
            index_elementCount.append(indexij)
    ds_mesh = ds_mesh.isel(elementCount=index_elementCount)
    return ds_mesh

def subset_allfiles(datm_xml_dict, keyword_data, outpathSubset):
    # e.g., datm_xml_dict['file']['stream_info'][0]['datafiles']['file']
    inout_maplist = []
    for kyd in keyword_data:
        flag = False
        for dicti in datm_xml_dict['file']['stream_info']:
            if kyd in dicti['@name']:
                flag = True
                infilelist = dicti['datafiles']['file']
                meshfile = dicti['meshfile']
                if isinstance(infilelist, str):
                    infilelist = [infilelist]
                break
        if flag == False:
            sys.exit(f'Error!!! Cannot find keyword {kyd} in datm_xml_dict!')
        # subset forcing files
        outfilelist = []
        for i in range(len(infilelist)):
            t1 = time.time()
            infilei = infilelist[i]
            filep = pathlib.Path(infilei)
            filename = filep.name
            foldername = filep.parent.name
            if i == 0:
                outpath = f'{outpathSubset}/{foldername}'
                os.makedirs(outpath, exist_ok=True)
            outfilei = f'{outpath}/subset_{filename}'
            outfilelist.append(outfilei)
            if os.path.isfile(outfilei):
                # print('Outfile exists:', outfilei)
                pass
            else:
                # print('Generating outfile:', outfilei)
                ds_data_out = subset_forcing(infilei, lat_range, lon_range)
                ds_data_out.to_netcdf(outfilei, format=ncformat)
            t2 = time.time()
            # print('Time cost (sec):', t2-t1)
        # subset mesh files
        outfilemesh = f'{outpath}/subset_{pathlib.Path(meshfile).name}'
        if not os.path.isfile(outfilemesh):
            ds_mesh = subset_meshfile(meshfile, outfilelist[0])
            ds_mesh.to_netcdf(outfilemesh, format=ncformat)
        # write input file list
        outfile_infilelist = f'{outpath}/input_output_filelist.txt'
        with open(outfile_infilelist, 'w') as f:
            for i in range(len(infilelist)):
                _ = f.write(f'datafile: {infilelist[i]} -> {outfilelist[i]}\n')
            f.write(f'meshfile: {meshfile} -> {outfilemesh}\n')
        inout_maplist.append(outfile_infilelist)
    return inout_maplist


inpathCTSMcase = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib/CAMELS_0'
infileMESH = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_basin_divide/ESMFmesh_ctsm_HCDN_nhru_final_671_v0_8e-3_basin0.nc'

outpathSubset = inpathCTSMcase + '_SubsetForcing'
os.makedirs(outpathSubset, exist_ok=True)

ncformat = 'NETCDF3_CLASSIC'

########################################################################################################################

# back up files (generated when creating new case). only back up for the first-time run
datm_streams_xml = f'{inpathCTSMcase}/Buildconf/datmconf/datm.streams.xml'
initial_datm_streams_xml = datm_streams_xml + '-initial'
if not os.path.isfile(initial_datm_streams_xml):
    _ = shutil.copy(datm_streams_xml, initial_datm_streams_xml + '-initial')

user_nl_datm_streams = f'{inpathCTSMcase}/user_nl_datm_streams'
initial_user_nl_datm_streams = user_nl_datm_streams + '-initial'
if not os.path.isfile(initial_user_nl_datm_streams):
    _ = shutil.copy(user_nl_datm_streams, initial_user_nl_datm_streams)

########################################################################################################################
# basin boundary
ds_basin = xr.load_dataset(infileMESH)
nodeCoords = ds_basin['nodeCoords'].values
lat_range = [np.min(nodeCoords[:, 1]), np.max(nodeCoords[:, 1])]
lon_range = [np.min(nodeCoords[:, 0]), np.max(nodeCoords[:, 0])]

########################################################################################################################
# create subset forcing / mesh files
with open(initial_datm_streams_xml) as f:
    datm_xml_dict = xmltodict.parse(f.read())

# index_data = [0, 1, 2] # e.g., 0: 'CLMNLDAS2.Solar'; 1: 'CLMNLDAS2.Precip', 2: 'CLMNLDAS2.TPQW', etc
# keyword_data = ['CLMNLDAS2.Solar', 'CLMNLDAS2.Precip', 'CLMNLDAS2.TPQW', 'topo.observed']
keyword_data = ['CLMNLDAS2.Solar', 'CLMNLDAS2.Precip', 'CLMNLDAS2.TPQW']
inout_maplist = subset_allfiles(datm_xml_dict, keyword_data, outpathSubset)

########################################################################################################################

# update user_nl_datm_streams
with open(initial_user_nl_datm_streams, 'r') as f:
    contents = f.readlines()

contents.append('\n')
for i in range(len(keyword_data)):
    # data files and mesh file
    datafilesi = ''
    with open(inout_maplist[i], 'r') as f:
        for li in f:
            li = li.strip().split(':')
            li_tag = li[0]
            if li_tag == 'datafile':
                datafilesi = datafilesi + li[1].strip().split('->')[1].strip() + ','
            elif li_tag == 'meshfile':
                meshfile = li[1].strip().split('->')[1].strip()
    datafilesi = datafilesi[:-1]
    # add contents
    contents.append(f'{keyword_data[i]}:meshfile={meshfile}\n')
    if not keyword_data[i] == 'topo.observed':
        contents.append(f'{keyword_data[i]}:mapalgo=nn\n')
    contents.append(f'{keyword_data[i]}:datafiles={datafilesi}\n')

# Add Sean's topo data
contents.append('topo.observed:meshfile=/glade/work/swensosc/topo_data/ESMFmesh_ctsm_elev_Conus_0.125d_210810.cdf5.nc\n')
contents.append('topo.observed:datafiles=/glade/work/swensosc/topo_data/ctsm_elev_Conus_0.125d.cdf5.nc\n')

# write to file
with open(user_nl_datm_streams, 'w') as f:
    for l in contents:
        _ = f.write(l)