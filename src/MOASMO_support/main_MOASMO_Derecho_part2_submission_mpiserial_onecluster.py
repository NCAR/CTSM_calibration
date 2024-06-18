# group different basins in one node
# generate submission scripts. don't submit real jobs

# if using MPILIB=mpi-serial, using this script which is much simpler

import numpy as np
import os, glob, sys
import pandas as pd


# iter = 0 # iteration number
# basin_on_one_node = 6 # number of basins in one node
iter = int(sys.argv[1])
basin_on_one_node = int(sys.argv[2])
print('Proecssing iteration', iter)
print('basin_on_one_node', basin_on_one_node)



# select one cluster
infile_basin_info = f"/glade/work/guoqiang/CTSM_CAMELS/data_mesh_surf/HillslopeHydrology/CAMELS_level1_basin_info.csv"
df_info = pd.read_csv(infile_basin_info)

infile = "/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/camels_cluster/Manuela_Brunner_2020/flood_cluster_memberships_CAMELS.txt"
df_cluster = pd.read_csv(infile)
df_cluster = df_cluster.rename(
    columns={"Camels_IDs": "hru_id", "flood_cluster": "clusters"}
)
df_cluster2 = pd.DataFrame()

for id in df_info["hru_id"].values:
    dfi = df_cluster.loc[df_cluster["hru_id"] == id]
    df_cluster2 = pd.concat([df_cluster2, dfi])

df_cluster2.sel_index = np.arange(len(df_cluster2))
df_cluster = df_cluster2
del df_cluster2
df_cluster["clusters"] = df_cluster["clusters"] - 1  # starting from 0

if np.any(df_info["hru_id"].values - df_cluster['hru_id'].values != 0):
    sys.exit("Mistmatch between basins and clusters")
else:
    print("basins and clusters match")

sel_cluster = 2
sel_index = np.where(df_cluster["clusters"].values == sel_cluster)[0]



inpath = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange'
outpath = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/run_model_mpiserial/iter{iter}'
os.makedirs(outpath, exist_ok=True)


# basins = [f'level1_{i}' for i in range(627)] + [f'level2_{i}' for i in range(40)] + [f'level3_{i}' for i in range(4)]
basins = [f'level1_{i}' for i in range(627) if i in sel_index] # Just level-1

bnum = len(basins)

if np.mod(bnum, basin_on_one_node) == 0:
    nbatch = int(bnum/basin_on_one_node)
else:
    nbatch = int(bnum/basin_on_one_node) + 1

cpu_per_node = 128

jobparams = [ "#PBS -N MOAcalib", "#PBS -q main", 
              "#PBS -l select=1:ncpus=128", "#PBS -l walltime=12:00:00", 
              #"#PBS -l job_priority=economy",
              "#PBS -A P08010000",
             "\n",
              "module load conda nco cdo",
             "conda activate npl-2024a",
             "\n",
            ]

# for i in range(nbatch):
#     bstart = i*basin_on_one_node
#     bend = (i+1)*basin_on_one_node
#     if bend>bnum:
#         bend = bnum

#     outpathi = f'{outpath}/batch{i}'
#     os.makedirs(outpathi, exist_ok=True)
    
#     # generate command line file
#     newcommands = []
#     for j in range(bstart, bend):
#         infileij = f'{inpath}/{basins[j]}_MOASMOcalib/run_model/iter{iter}/commands_run_iter{iter}.txt'
#         with open(infileij, 'r') as f:
#             linesj = f.readlines()
#         for l in linesj:
#             l = l.strip().split(' ')
#             newcommands.append(' '.join(l[:-1]))

#     outfile_newcom = f'{outpathi}/batch_{i}.txt'
#     with open(outfile_newcom, 'w') as f:
#         for l in newcommands:
#             _ = f.write(l+'\n')

#     # generate submission list
#     filesub = f'{outpathi}/submission.sh'
#     with open(filesub, 'w') as f:
#         for l in jobparams:
#             _ = f.write(l+'\n')
#         command = f"parallel -j {cpu_per_node} < {outfile_newcom}"
#         _ = f.write(command+'\n')

# group different basins in one node
# generate submission scripts. don't submit real jobs

# if using MPILIB=mpi-serial, using this script which is much simpler




# iter = 0 # iteration number
# basin_on_one_node = 6 # number of basins in one node
iter = int(sys.argv[1])
basin_on_one_node = int(sys.argv[2])
print('Proecssing iteration', iter)
print('basin_on_one_node', basin_on_one_node)



# select one cluster
infile_basin_info = f"/glade/work/guoqiang/CTSM_CAMELS/data_mesh_surf/HillslopeHydrology/CAMELS_level1_basin_info.csv"
df_info = pd.read_csv(infile_basin_info)

infile = "/glade/u/home/guoqiang/CTSM_repos/CTSM_calibration/src/camels_cluster/Manuela_Brunner_2020/flood_cluster_memberships_CAMELS.txt"
df_cluster = pd.read_csv(infile)
df_cluster = df_cluster.rename(
    columns={"Camels_IDs": "hru_id", "flood_cluster": "clusters"}
)
df_cluster2 = pd.DataFrame()

for id in df_info["hru_id"].values:
    dfi = df_cluster.loc[df_cluster["hru_id"] == id]
    df_cluster2 = pd.concat([df_cluster2, dfi])

df_cluster2.sel_index = np.arange(len(df_cluster2))
df_cluster = df_cluster2
del df_cluster2
df_cluster["clusters"] = df_cluster["clusters"] - 1  # starting from 0

if np.any(df_info["hru_id"].values - df_cluster['hru_id'].values != 0):
    sys.exit("Mistmatch between basins and clusters")
else:
    print("basins and clusters match")

sel_cluster = 2
sel_index = np.where(df_cluster["clusters"].values == sel_cluster)[0]

inpath = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange'
outpath = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO_bigrange/run_model_mpiserial/iter{iter}'
os.makedirs(outpath, exist_ok=True)


# basins = [f'level1_{i}' for i in range(627)] + [f'level2_{i}' for i in range(40)] + [f'level3_{i}' for i in range(4)]
basins = [f'level1_{i}' for i in range(627) if i in sel_index] # Just level-1

bnum = len(basins)

if np.mod(bnum, basin_on_one_node) == 0:
    nbatch = int(bnum/basin_on_one_node)
else:
    nbatch = int(bnum/basin_on_one_node) + 1

cpu_per_node = 128

jobparams = [ "#PBS -N MOAcalib", "#PBS -q main", 
              "#PBS -l select=1:ncpus=128", "#PBS -l walltime=12:00:00", 
              #"#PBS -l job_priority=economy",
              "#PBS -A P08010000",
             "\n",
              "module load conda nco cdo",
             "conda activate npl-2024a",
             "\n",
            ]


allcommands = []
for j in range(len(basins)):
    infileij = f'{inpath}/{basins[j]}_MOASMOcalib/run_model/iter{iter}/commands_run_iter{iter}.txt'
    with open(infileij, 'r') as f:
        linesj = f.readlines()
    for l in linesj:
        l = l.strip().split(' ')
        allcommands.append(' '.join(l[:-1]))

allcommands = np.array(allcommands)

trials_on_one_node = cpu_per_node
nbatch = int(len(allcommands)/trials_on_one_node) + 1



for i in range(nbatch):
    bstart = i*trials_on_one_node
    bend = (i+1)*trials_on_one_node
    if bend>len(allcommands):
        bend = len(allcommands)

    outpathi = f'{outpath}/batch{i}'
    os.makedirs(outpathi, exist_ok=True)
    
    # generate command line file
    newcommands = allcommands[bstart:bend]


    outfile_newcom = f'{outpathi}/batch_{i}.txt'
    with open(outfile_newcom, 'w') as f:
        for l in newcommands:
            _ = f.write(l+'\n')

    # generate submission list
    filesub = f'{outpathi}/submission.sh'
    with open(filesub, 'w') as f:
        for l in jobparams:
            _ = f.write(l+'\n')
        command = f"parallel -j {cpu_per_node} < {outfile_newcom}"
        _ = f.write(command+'\n')

