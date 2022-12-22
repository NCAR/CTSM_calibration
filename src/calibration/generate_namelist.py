# Actually, this script just appends some settings to namelist files
# In the future, this may turn to real generation

import toml, sys, os
config_file_NL = sys.argv[1]

print('Create namelist settings ...')
print('Reading configuration from:', config_file_NL)

########################################################################################################################
# settings

##############
# parse settings

config_NL = toml.load(config_file_NL)
path_CTSM_case = config_NL['path_CTSM_case']
AddToNamelist = config_NL['AddToNamelist'] # a dict

########################################################################################################################
# change settings

cwd = os.getcwd()
os.chdir(path_CTSM_case)

writelist = list(AddToNamelist.items())

for i in range(len(writelist)):
    key, value = list(AddToNamelist.items())[i] # key should be the name of namelist file
    if not os.path.isfile(key):
        sys.exit(f'namelist file {key} does not exist in {path_CTSM_case}!')
    else:
        with open(key, 'a') as f:
            for j in range(len(value)):
                if len(value[j]) > 0:
                    print(f'Appending {value[j]} to {key}')
                    _ = f.write(value[j] + '\n')

os.chdir(cwd)