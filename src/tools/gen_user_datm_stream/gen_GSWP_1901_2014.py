# generate a new datm stream file
# e.g., extended time period and new datasets

import glob, os, sys


file_datm_stream = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Distr_Calib_no_nest/hillslope_SS587/Buildconf/datmconf/datm.streams.xml'
file_user_datm_stream = '/glade/work/guoqiang/CTSM_cases/CAMELS_Calib/Distr_Calib_no_nest/hillslope_SS587/user_nl_datm_streams'


datm_mode = 'CLMGSWP3v1'

attrs = {'year_first': 1901, 
         'year_last': 2014, 
         'year_align': 1901
        }

forcformat = {'Solar': '/glade/p/cgd/tss/CTSM_datm_forcing_data/atm_forcing.datm7.GSWP3.0.5d.v1.c170516/Solar/clmforc.GSWP3.c2011.0.5x0.5.Solr.YYYY-MM.nc', 
              'Precip': '/glade/p/cgd/tss/CTSM_datm_forcing_data/atm_forcing.datm7.GSWP3.0.5d.v1.c170516/Precip/clmforc.GSWP3.c2011.0.5x0.5.Prec.YYYY-MM.nc', 
              'TPQW': '/glade/p/cgd/tss/CTSM_datm_forcing_data/atm_forcing.datm7.GSWP3.0.5d.v1.c170516/TPHWL/clmforc.GSWP3.c2011.0.5x0.5.TPQWL.YYYY-MM.nc', 
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
