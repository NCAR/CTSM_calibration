# generate a new datm stream file
# e.g., extended time period and new datasets

import glob, os, sys


file_datm_stream = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Distr_Calib_no_nest/hillslope_SS587_E5EMforc/Buildconf/datmconf/datm.streams.xml'
file_user_datm_stream = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Distr_Calib_no_nest/hillslope_SS587_E5EMforc/user_nl_datm_streams'


datm_mode = 'CLMGSWP3v1'

attrs = {'year_first': 1951, 
         'year_last': 2019, 
         'year_align': 1951,
         'meshfile': '/glade/scratch/guoqiang/era5land_eme/mesh_file/clmforc.E5LEME.c2023.010x010.Precip_ESMFmesh_cdf5_140823.nc',
        }

#forcformat = {'Solar': '/glade/campaign/cgd/tss/common/lm_forcing/hybrid/era5land_eme/Solar/clmforc.E5LEME.c2023.010x010.Solar.YYYY-MM.nc', 
#              'Precip': '/glade/campaign/cgd/tss/common/lm_forcing/hybrid/era5land_eme/Precip/clmforc.E5LEME.c2023.010x010.Precip.YYYY-MM.nc', 
#              'TPQW': '/glade/campaign/cgd/tss/common/lm_forcing/hybrid/era5land_eme/TPQWL/clmforc.E5LEME.c2023.010x010.TPQWL.YYYY-MM.nc', 
#             }
forcformat = {'Solar': '/glade/scratch/guoqiang/era5land_eme/Solar/clmforc.E5LEME.c2023.010x010.Solar.YYYY-MM.nc', 
              'Precip': '/glade/scratch/guoqiang/era5land_eme/Precip/clmforc.E5LEME.c2023.010x010.Precip.YYYY-MM.nc', 
              'TPQW': '/glade/scratch/guoqiang/era5land_eme/TPQWL/clmforc.E5LEME.c2023.010x010.TPQWL.YYYY-MM.nc', 
             }

os.system(f'cp {file_datm_stream} {file_datm_stream}-original')


lines = []
for s in ['Solar', 'Precip', 'TPQW']:
    # basic attrs
    for key, value in attrs.items():
        lines.append(f'{datm_mode}.{s}:{key}={value}\n')
    # datafiles
    forms = forcformat[s]
    datafiles = ''
    for y in range(attrs['year_first'], attrs['year_last']+1):
        for m in range(1, 13):
            filem = forms.replace('YYYY-MM', f'{y}-{m:02}') 
            if not os.path.isfile(filem):
                sys.exit(f'Cannot find file: {filem}')
            datafiles = datafiles + filem + ','
    datafiles = datafiles[:-1]
    lines.append(f'{datm_mode}.{s}:datafiles={datafiles}\n')
    lines.append('\n')


with open(file_user_datm_stream, 'w') as f:
    for l in lines:
        f.write(l)
