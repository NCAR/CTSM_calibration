
import subprocess, os, pathlib

def xmlquery_output(pathCTSM, keyword):
    os.chdir(pathCTSM)
    out = subprocess.run(f'./xmlquery {keyword}', shell=True, capture_output=True)
    out = out.stdout.decode().strip().split(' ')[-1]
    return out


path_CTSM = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Lump_calib_split_nest_LMWG/CAMELS_100_iter0_trial0'
clone_case_name = pathlib.Path(path_CTSM).name

output_dir = xmlquery_output(path_CTSM, 'DOUT_S_ROOT')
cimeoutroot = xmlquery_output(path_CTSM, 'CIME_OUTPUT_ROOT')
rundir = xmlquery_output(path_CTSM, 'RUNDIR')
_ = subprocess.run(f'rm -r f{cimeoutroot}/{clone_case_name}', shell=True)
_ = subprocess.run(f'rm -r {path_CTSM}', shell=True)
_ = subprocess.run(f'rm -r {output_dir}', shell=True)
_ = subprocess.run(f'rm -r {rundir}', shell=True)