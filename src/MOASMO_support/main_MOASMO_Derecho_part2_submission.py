# group different basins in one node

import numpy as np
import os, glob, sys



# iter = 0 # iteration number
# basin_on_one_node = 6 # number of basins in one node
iter = int(sys.argv[1])
basin_on_one_node = int(sys.argv[2])
print('Proecssing iteration', iter)
print('basin_on_one_node', basin_on_one_node)


inpath = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO'
outpath = f'/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_MOASMO/run_model/iter{iter}'
os.makedirs(outpath, exist_ok=True)


# basins = [f'level1_{i}' for i in range(627)] + [f'level2_{i}' for i in range(40)] + [f'level3_{i}' for i in range(4)]
basins = [f'level1_{i}' for i in range(627)] # Just level-1

bnum = len(basins)

nbatch = int(bnum/basin_on_one_node) + 1

cpu_per_node = 128

jobparams = [ "#PBS -N MOAcalib", "#PBS -q main", 
              "#PBS -l select=1:ncpus=128", "#PBS -l walltime=12:00:00", 
              "#PBS -l job_priority=economy",
              "#PBS -A P08010000",
             "\n",
              "module load conda nco cdo",
             "conda activate npl-2024a",
             "\n",
            ]

for i in range(nbatch):
    bstart = i*basin_on_one_node
    bend = (i+1)*basin_on_one_node
    if bend>bnum:
        bend = bnum

    outpathi = f'{outpath}/batch{i}'
    os.makedirs(outpathi, exist_ok=True)
    
    # generate command line file
    newcommands = []
    for j in range(bstart, bend):
        infileij = f'{inpath}/{basins[j]}_MOASMOcalib/run_model/iter{iter}/commands_run_iter{iter}.txt'
        with open(infileij, 'r') as f:
            linesj = f.readlines()
        for l in linesj:
            l = l.strip().split(' ')
            l[-1] = outpathi
            newcommands.append(' '.join(l))

    for j in range(len(newcommands)): # which CPU to use
        if j < cpu_per_node:
            newcommands[j] = newcommands[j] + f' {j}'
        else:
            newcommands[j] = newcommands[j] + f' NA'

    outfile_newcom = f'{outpathi}/batch_{i}.txt'
    with open(outfile_newcom, 'w') as f:
        for l in newcommands:
            _ = f.write(l+'\n')

    # generate cpu list
    for j in range(cpu_per_node):
        filej = f'{outpathi}/idlecpu_{j}'
        with open(filej, 'w') as f:
            pass

    # generate submission list
    filesub = f'{outpathi}/submission.sh'
    with open(filesub, 'w') as f:
        for l in jobparams:
            _ = f.write(l+'\n')
        command = f"parallel -j {cpu_per_node} < {outfile_newcom}"
        _ = f.write(command+'\n')