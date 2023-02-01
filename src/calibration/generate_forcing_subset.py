# Create a regional subset for the study area can accelerate modeling
import shutil, toml
import xarray as xr
import numpy as np
import xmltodict, subprocess
import os, time, sys, glob, pathlib
import timemerge

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

def subset_allfiles(datm_xml_dict, keyword_data, outpathSubset, ncformat, subset_length):
    # e.g., datm_xml_dict['file']['stream_info'][0]['datafiles']['file']
    tstart = time.time()
    inout_maplist = []
    keyword_data_complete = []
    for kyd in keyword_data:
        flag = False
        for dicti in datm_xml_dict['file']['stream_info']:
            if kyd in dicti['@name']:
                keyword_data_complete.append(dicti['@name'])
                flag = True
                infilelist = dicti['datafiles']['file']
                meshfile = dicti['meshfile']
                if isinstance(infilelist, str):
                    infilelist = [infilelist]
                else:
                    if subset_length == 'existing':
                        pass
                    elif subset_length == 'all':
                        infile0 = infilelist[0]
                        inpath0 = pathlib.Path(infile0).parent
                        inname0 = pathlib.Path(infile0).name
                        infilelist0 = glob.glob(f'{inpath0}/*{inname0[-3:]}')
                        infilelist0.sort()
                        infilelist = []
                        for i in range(len(infilelist0)):
                            if len(pathlib.Path(infilelist0[i]).name) == len(inname0): # make sure this is real forcing
                                infilelist.append(infilelist0[i])
                    else:
                        sys.exit(f'subset_length must be existing or all. {subset_length} is unknown.')
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
                print('Outfile exists:', outfilei)
                pass
            else:
                print('Generating outfile:', outfilei)
                ds_data_out = subset_forcing(infilei, lat_range, lon_range)
                ds_data_out.to_netcdf(outfilei, format=ncformat)
            t2 = time.time()
            print('Time cost (sec):', t2-t1)
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
    tend = time.time()
    print('Time cost (sec):', tend-tstart)
    return inout_maplist, keyword_data_complete


# mergetime
def get_inputfiles_from_datm_streams(file_datmstreams, keywords = ['Solar:datafiles', 'Precip:datafiles', 'TPQW:datafiles']):
    # back up
    file_datmstreams_backup = f'{file_datmstreams}_beforeTimeMerge'
    if not os.path.isfile(file_datmstreams_backup):
        _ = subprocess.run(f'cp {file_datmstreams} {file_datmstreams_backup}', shell=True)

    # get file lists
    with open(file_datmstreams, 'r') as f:
        datm_streams = f.readlines()

    infile_lists = []
    for kw in keywords:
        flag = False
        for line in datm_streams:
            if not line.strip().startswith('!'):
                if kw in line:
                    files = line.strip().split('=')[1].split(',')
                    files = [f.strip() for f in files]
                    infile_lists.append(files)
                    flag = True
                    break
        if flag == False:
            sys.exit(f'Error! Cannot find {kw} in {file_datmstreams}.')

    return infile_lists

# generate new user_nl_datm_streams
def update_datastreams_datafiles(file_datastreams, all_outfiles, keyword):

    with open(file_datastreams, 'r') as f:
        datm_streams = f.readlines()

    for i in range(len(datm_streams)):
        linei = datm_streams[i]
        if keyword in linei:
            lineis = linei.strip().split('=')
            datm_streams[i] = lineis[0] + '=' + all_outfiles + '\n'
            break

    with open(file_datastreams, 'w') as f:
        for line in datm_streams:
            _ = f.write(line)

if __name__ == '__main__':

    config_file_SubForc = sys.argv[1]

    print('Create datm subset settings ...')
    print('Reading configuration from:', config_file_SubForc)

    ########################################################################################################################
    # settings

    ##############
    # parse settings

    config_SubForc = toml.load(config_file_SubForc)

    path_CTSM_case = config_SubForc['path_CTSM_case']
    subset_length = config_SubForc['subset_length']
    forcing_YearStep = config_SubForc['forcing_YearStep'] # number of years (e.g., 2 means that files are saved at a 2-y step. <=0: no time merging

    ##############
    # default settings

    outpathSubset = path_CTSM_case + '_SubsetForcing'
    os.makedirs(outpathSubset, exist_ok=True)

    ncformat = 'NETCDF3_CLASSIC'

    ########################################################################################################################

    # back up files (generated when creating new case). only back up for the first-time run
    datm_streams_xml = f'{path_CTSM_case}/Buildconf/datmconf/datm.streams.xml'
    initial_datm_streams_xml = datm_streams_xml + '-initial'
    if not os.path.isfile(initial_datm_streams_xml):
        _ = shutil.copy(datm_streams_xml, initial_datm_streams_xml)

    user_nl_datm_streams = f'{path_CTSM_case}/user_nl_datm_streams'
    initial_user_nl_datm_streams = user_nl_datm_streams + '-initial'
    if not os.path.isfile(initial_user_nl_datm_streams):
        _ = shutil.copy(user_nl_datm_streams, initial_user_nl_datm_streams)

    ########################################################################################################################
    # basin boundary

    # get mesh file
    cwd = os.getcwd()
    os.chdir(path_CTSM_case)
    out = subprocess.run('./xmlquery LND_DOMAIN_MESH', shell=True, capture_output=True)
    infileMESH = out.stdout.decode().strip().split(':')[1].strip()
    os.chdir(cwd)

    # find coordinates of basins with mask == 1
    ds_basin = xr.load_dataset(infileMESH)
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

    lat_range = [np.min(nodeCoords_valid[:, 1]), np.max(nodeCoords_valid[:, 1])]
    lon_range = [np.min(nodeCoords_valid[:, 0]), np.max(nodeCoords_valid[:, 0])]


    ########################################################################################################################
    # create subset forcing / mesh files
    with open(initial_datm_streams_xml) as f:
        datm_xml_dict = xmltodict.parse(f.read())

    keyword_data = ['Solar', 'Precip', 'TPQW']
    inout_maplist, keyword_data_complete = subset_allfiles(datm_xml_dict, keyword_data, outpathSubset, ncformat, subset_length)

    ########################################################################################################################
    # update user_nl_datm_streams

    with open(initial_user_nl_datm_streams, 'r') as f:
        contents = f.readlines()

    contents.append('\n')
    for i in range(len(keyword_data_complete)):
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
        contents.append(f'{keyword_data_complete[i]}:meshfile={meshfile}\n')
        if not keyword_data_complete[i] == 'topo.observed':
            contents.append(f'{keyword_data_complete[i]}:mapalgo=nn\n')
        contents.append(f'{keyword_data_complete[i]}:datafiles={datafilesi}\n')

    # write to file
    with open(user_nl_datm_streams, 'w') as f:
        for l in contents:
            _ = f.write(l)

    ########################################################################################################################
    # post-processing: merge files in a folder to the target time step to reduce the number of files

    if forcing_YearStep > 0:

        print(f'Start time mering to forcing_YearStep {forcing_YearStep}')

        keywords = ['Solar:datafiles', 'Precip:datafiles', 'TPQW:datafiles']
        infile_lists = get_inputfiles_from_datm_streams(user_nl_datm_streams, keywords)

        for i in range(len(infile_lists)):
            infile_list = np.array(infile_lists[i])
            # subset
            all_outfiles, maplist_infile, maplist_outfile = timemerge.highlevel_cdo_mergetime(infile_list, forcing_YearStep)
            # update datastreams files
            update_datastreams_datafiles(user_nl_datm_streams, all_outfiles, keywords[i])
            # finish
            print('Sucessful mergetime!')
