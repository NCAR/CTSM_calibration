import numpy as np
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def process_basin(basin, inpath_moasmo, nmet=24):
    if np.mod(basin, 50) == 0:
        print(f'Processing basin {basin}')

    allsuffix = ['many_metrics', 'many_metrics_mizuroute_s-1', 'many_metrics_mosart_s-1']

    for suffix in allsuffix:
    
        iterflag = 0
        totnum = 200    
        metrics = np.nan * np.zeros([totnum, nmet])
    
    
        for trialflag in range(totnum):
            path_archive = f'{inpath_moasmo}/level1_{basin}_calib/ctsm_outputs'
            caseflag = f'iter{iterflag}_trial{trialflag}'
            outfile_metric = f'{path_archive}/{caseflag}/evaluation_{suffix}.csv'
    
            if os.path.isfile(outfile_metric):
                try:
                    df = pd.read_csv(outfile_metric)
                    metrics[trialflag, :] = df.values[0]
                except Exception as e:
                    print(f'Failed reading {outfile_metric}: {e}')
            else:
                print('File not exist', outfile_metric)
    
        # Save metrics
        outfile = f'{path_archive}/iter{iterflag}_{suffix}.csv'
        if os.path.isfile(outfile):
        # if False: # always overwrite
            print(f'{outfile} already exists')
        else:
            dfout = pd.DataFrame(metrics, columns=df.columns)
            dfout.to_csv(outfile, index=False)
    
    return basin

def main(inpath_moasmo):
    basinnum = 627
    nmet = 24

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_basin, basin, inpath_moasmo, nmet) for basin in range(basinnum)]
        # futures = [executor.submit(process_basin, basin, inpath_moasmo, nmet) for basin in tarb]
        
        for future in futures:
            basin = future.result()
            if basin is not None:
                print(f'Completed processing for basin {basin}')

if __name__ == '__main__':
    inpath_moasmo = '/glade/campaign/cgd/tss/people/guoqiang/CTSM_CAMELS_proj/Calib_HH_emulator'  # Set your actual path here
    main(inpath_moasmo)
