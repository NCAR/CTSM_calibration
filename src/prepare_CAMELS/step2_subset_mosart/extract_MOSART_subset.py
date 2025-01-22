# interpolate gridded MOSART data to mesh grids of basins
# adapted from CTSM regional data subset codes

import os, sys
from datetime import datetime
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def subset_mesh(mesh_infile, mesh_outfile, lat1, lat2, lon1, lon2):
    """
    Subsets the mesh file based on latitude and longitude bounds.
    """
    print("Subsetting mesh file for the specified region.")

    today = datetime.today()
    today_string = today.strftime("%y%m%d")

    print("mesh_in  : ", mesh_infile)
    print("mesh_out : ", mesh_outfile)

    node_coords, subset_element, subset_node, conn_dict = subset_mesh_at_reg(mesh_infile, lat1, lat2, lon1, lon2)

    f_in = xr.open_dataset(mesh_infile)
    write_mesh(f_in, node_coords, subset_element, subset_node, conn_dict, mesh_outfile)

def subset_mesh_at_reg(mesh_in, lat1, lat2, lon1, lon2):
    """
    Subsets the mesh based on latitude and longitude bounds.
    """
    f_in = xr.open_dataset(mesh_in)
    elem_count = len(f_in["elementCount"])
    elem_conn = f_in["elementConn"]
    num_elem_conn = f_in["numElementConn"]
    node_count = len(f_in["nodeCount"])
    node_coords = f_in["nodeCoords"]

    subset_element = []
    subset_node = []
    conn_dict = {}
    cnt = 1

    for n in range(elem_count):
        endx = elem_conn[n, : num_elem_conn[n].values].values - 1  # convert to zero-based index
        nlon = node_coords[endx.astype(int), 0].values
        nlat = node_coords[endx.astype(int), 1].values

        if np.all((nlon >= lon1) & (nlon <= lon2) & (nlat >= lat1) & (nlat <= lat2)):
            subset_element.append(n)

    for n in range(node_count):
        nlon = node_coords[n, 0].values
        nlat = node_coords[n, 1].values

        if (nlon >= lon1) and (nlon <= lon2) and (nlat >= lat1) and (nlat <= lat2):
            subset_node.append(n)
            conn_dict[n + 1] = cnt
            cnt += 1
        else:
            conn_dict[n + 1] = -9999

    return node_coords, subset_element, subset_node, conn_dict

def write_mesh(f_in, node_coords, subset_element, subset_node, conn_dict, mesh_out):
    """
    Writes out the subsetted mesh file.
    """
    corner_pairs = f_in.variables["nodeCoords"][subset_node]
    variables = f_in.variables
    global_attributes = f_in.attrs

    max_node_dim = len(f_in["maxNodePElement"])

    elem_count = len(subset_element)
    elem_conn_out = np.empty(shape=[elem_count, max_node_dim])
    elem_conn_index = f_in.variables["elementConn"][subset_element]

    for n in range(elem_count):
        for m in range(max_node_dim):
            ndx = int(elem_conn_index[n, m])
            elem_conn_out[n, m] = conn_dict[ndx]

    num_elem_conn_out = f_in.variables["numElementConn"][subset_element]
    center_coords_out = f_in.variables["centerCoords"][subset_element]

    f_out = xr.Dataset()

    f_out["nodeCoords"] = xr.DataArray(corner_pairs, dims=("nodeCount", "coordDim"), attrs={"units": "degrees"})
    f_out["elementConn"] = xr.DataArray(elem_conn_out, dims=("elementCount", "maxNodePElement"),
                                         attrs={"long_name": "Node indices that define the element connectivity"})
    f_out["numElementConn"] = xr.DataArray(num_elem_conn_out, dims=("elementCount"),
                                            attrs={"long_name": "Number of nodes per element"})
    f_out["centerCoords"] = xr.DataArray(center_coords_out, dims=("elementCount", "coordDim"), attrs={"units": "degrees"})

    # Write out the dataset
    for attr in global_attributes:
        f_out.attrs[attr] = global_attributes[attr]

    f_out.attrs.update({
        "title": "ESMF unstructured grid file for a region",
        "created_by": "subset_data",
        "date_created": "{}".format(datetime.now()),
    })

    f_out.to_netcdf(mesh_out)
    print("Successfully created file (mesh_out):", mesh_out)


####### get boundary
# get target boundary
basin = int(sys.argv[1])
print('processing basin', basin)

outpath = '/glade/work/guoqiang/CTSM_CAMELS/data_mosart'
os.makedirs(outpath, exist_ok=True)

file_mesh = f'/glade/work/guoqiang/CTSM_CAMELS/data_mesh_surf/HillslopeHydrology/disaggregation/corrected_HCDN_nhru_final_671_buff_fix_holes.CAMELSandTDX_areabias_fix.simp0.001.level1_polygons_neighbor_group_esmf_mesh_{basin}.nc'

ds_basin = xr.load_dataset(file_mesh)
nodeCoords = ds_basin.nodeCoords.values
elementConn = ds_basin.elementConn.values
numElementConn = ds_basin.numElementConn.values
nodeCoords_valid = np.zeros([0, 2])

for i in range(ds_basin.elementCount.size):
	if ds_basin.elementMask.values[i] == 1:
		index_elementCount = [i]
		index_connectionCount = np.arange(np.sum(numElementConn[:i]), np.sum(numElementConn[:i+1])).astype(int)
		index_nodeCount = elementConn[index_connectionCount].astype(int) - 1 # index starts from 0
		nodeCoords_valid = np.vstack((nodeCoords_valid, nodeCoords[index_nodeCount, :]))

buffer = 0.2
lat_range = [np.min(nodeCoords_valid[:, 1])-buffer, np.max(nodeCoords_valid[:, 1])+buffer]
lon_range = [np.min(nodeCoords_valid[:, 0])-buffer, np.max(nodeCoords_valid[:, 0])+buffer]

lat1, lat2 = lat_range[0], lat_range[1]
lon1, lon2 = lon_range[0], lon_range[1]
print(lat1, lat2, lon1, lon2)


######### subset mesh
mesh_infile = '/glade/campaign/cesm/cesmdata/inputdata/share/meshes/0.125nldas2_ESMFmesh_cd5_241220.nc'
mesh_outfile = f'{outpath}/CAMELS_level1_{basin}_0.125nldas2_ESMFmesh.nc'
if os.path.isfile(mesh_outfile):
    print('file exists', mesh_outfile)
else:
    subset_mesh(mesh_infile, mesh_outfile, lat1, lat2, lon1, lon2)

######### subset mosart
infile_mosart = '/glade/campaign/cesm/cesmdata/inputdata/rof/mosart/MOSART_routing_0.125nldas2_cdf5_c200727.nc'
outfile_mosart = f'{outpath}/CAMELS_level1_{basin}_MOSART_routing_0.125nldas2.nc'
res = 0.125

if os.path.isfile(outfile_mosart):
    print('file exists', outfile_mosart)
else:
# if True:
    print('mosart in', infile_mosart)
    print('mosart out', outfile_mosart)
    
    ds_in = xr.load_dataset(infile_mosart)
    lon = ds_in.longxy.values[0,:]
    lat = ds_in.latixy.values[:,0]


    dsmesh = xr.open_dataset(mesh_outfile)
    lonmesh, latmesh = np.unique(dsmesh.nodeCoords.values[:,0]-360), np.unique(dsmesh.nodeCoords.values[:,1])
    lat1m, lat2m, lon1m, lon2m = np.min(latmesh), np.max(latmesh), np.min(lonmesh), np.max(lonmesh)

    xind = np.where((lon >= lon1m) & (lon <= lon2m))[0]
    yind = np.where((lat >= lat1m) & (lat <= lat2m))[0]
    
    ds_out = ds_in.isel(lon=xind, ncl1=xind, ncl3=xind, ncl5=xind, ncl7=xind,
                        lat=yind, ncl0=yind, ncl2=yind, ncl4=yind, ncl6=yind,)


    diff1 = np.abs(ds_out.longxy.values[0,:] - (lonmesh[1:]+lonmesh[0:-1])/2)
    diff2 = np.abs(ds_out.latixy.values[:,0] - (latmesh[1:]+latmesh[0:-1])/2)
    if np.any(diff1>0.001) or np.any(diff2>0.001):
        sys.exit('Mistmatch lat/lon')

    # New IDs
    dsmo_ID = ds_out.ID.values
    dsmo_dnID = ds_out.dnID.values
    new_ID = np.arange(1, dsmo_ID.size + 1).reshape(dsmo_ID.shape)
    
    # Mapping dnID to new IDs, assigning -9999 for unknowns
    new_dnID = np.full(dsmo_dnID.shape, -9999)
    for i in range(dsmo_dnID.shape[0]):
        for j in range(dsmo_dnID.shape[1]):
            original_dnID = dsmo_dnID[i, j]
            if original_dnID in dsmo_ID:
                new_dnID[i, j] = new_ID[np.where(dsmo_ID == original_dnID)][0]  # Find corresponding new ID
    
    ds_out.ID.values = new_ID
    ds_out.dnID.values = new_dnID
    
    ds_out.to_netcdf(outfile_mosart)
    
