# Delete CTSM cases and relevant folders
import subprocess, os, pathlib, glob

def xmlquery_output(pathCTSM, keyword):
    os.chdir(pathCTSM)
    out = subprocess.run(f'./xmlquery {keyword}', shell=True, capture_output=True)
    out = out.stdout.decode().strip().split(' ')[-1]
    return out


# path_CTSMall = ['/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/test/CAMELS_100']
path_CTSMall = glob.glob('/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/test/*/', recursive = True)


for path_CTSM in path_CTSMall:
    clone_case_name = pathlib.Path(path_CTSM).name
    output_dir = xmlquery_output(path_CTSM, 'DOUT_S_ROOT')
    cimeoutroot = xmlquery_output(path_CTSM, 'CIME_OUTPUT_ROOT')
    rundir = xmlquery_output(path_CTSM, 'RUNDIR')
    print(f'Delete CIME_OUTPUT_ROOT: {cimeoutroot}/{clone_case_name}')
    _ = subprocess.run(f'rm -r {cimeoutroot}/{clone_case_name}', shell=True)
    print(f'Delete case: {path_CTSM}')
    _ = subprocess.run(f'rm -r {path_CTSM}', shell=True)
    print(f'Delete output dir: {output_dir}')
    _ = subprocess.run(f'rm -r {output_dir}', shell=True)
    print(f'Delete run dir: {rundir}')
    _ = subprocess.run(f'rm -r {rundir}', shell=True)